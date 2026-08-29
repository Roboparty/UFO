"""Supervised training for temporal completion of projected terrain maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, random_split

from humanoidverse.perception.temporal_terrain import (
    TerrainCompletionLossConfig,
    TemporalTerrainCompletion,
    build_no_odometry_history,
    terrain_completion_loss,
    terrain_completion_metrics,
    sharpen_terrain_prediction,
    warp_terrain_history_to_current,
)
from humanoidverse.perception.terrain_dataset import TerrainPerceptionSequenceDataset


class ChunkGroupedShuffleSampler(Sampler[int]):
    """Shuffle samples without repeatedly reloading large on-disk chunks."""

    def __init__(self, dataset: Dataset, *, seed: int, num_samples: int | None = None) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        if num_samples is not None and not 0 < num_samples <= len(dataset):
            raise ValueError("num_samples must lie in [1, len(dataset)]")
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        if isinstance(dataset, Subset):
            base = dataset.dataset
            if not isinstance(base, TerrainPerceptionSequenceDataset):
                raise TypeError("subset must wrap TerrainPerceptionSequenceDataset")
            source_indices = list(dataset.indices)
        elif isinstance(dataset, TerrainPerceptionSequenceDataset):
            base = dataset
            source_indices = list(range(len(dataset)))
        else:
            raise TypeError("ChunkGroupedShuffleSampler requires TerrainPerceptionSequenceDataset")
        self._groups: dict[int, list[int]] = {}
        for local_index, source_index in enumerate(source_indices):
            chunk_index = base.chunk_index_for_sample(int(source_index))
            self._groups.setdefault(chunk_index, []).append(local_index)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        chunk_ids = list(self._groups)
        emitted = 0
        for group_position in torch.randperm(len(chunk_ids), generator=generator).tolist():
            group = self._groups[chunk_ids[group_position]]
            for sample_position in torch.randperm(len(group), generator=generator).tolist():
                yield group[sample_position]
                emitted += 1
                if emitted >= self.num_samples:
                    return


def _run_epoch(
    model: TemporalTerrainCompletion,
    loader: DataLoader,
    *,
    device: torch.device,
    history_seconds: float,
    optimizer: torch.optim.Optimizer | None,
    stairs_terrain_ids: tuple[int, ...],
    history_mode: str = "egomotion_warp",
    loss_config: TerrainCompletionLossConfig | None = None,
    terrain_names: tuple[str, ...] = (),
    report_metric_counts: bool = False,
    report_prediction_metrics: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    metric_counts: dict[str, float] = {}
    batches = 0
    for batch in loader:
        partial = batch["partial_map"].to(device)
        visible = batch["visible_mask"].to(device)
        timestamps = batch["timestamp_s"].to(device)
        proprio = batch["proprio"].to(device)
        target = batch["gt_terrain_actor"].to(device)
        if history_mode == "egomotion_warp":
            history = warp_terrain_history_to_current(
                partial,
                visible,
                batch["pelvis_pos_w"].to(device),
                batch["heading_yaw_w"].to(device),
                timestamps_s=timestamps,
                history_seconds=history_seconds,
                interpolation="bilinear",
            )
        elif history_mode == "no_odometry":
            history = build_no_odometry_history(
                partial,
                visible,
                timestamps_s=timestamps,
                frame_valid=batch.get("frame_valid", None).to(device) if "frame_valid" in batch else None,
                history_seconds=history_seconds,
            )
        else:
            raise ValueError(f"unsupported history_mode: {history_mode}")
        with torch.set_grad_enabled(training):
            output = model(history, proprio=proprio)
            loss = terrain_completion_loss(
                output.predicted_clearance,
                target,
                current_visible=output.current_visible,
                config=loss_config,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        values = {"loss": float(loss.detach().item())}
        weights = {"loss": float(partial.shape[0])}

        def add_metrics(prefix: str, payload: dict[str, torch.Tensor]) -> None:
            for name, value in payload.items():
                if name.endswith("__count"):
                    continue
                output_name = f"{prefix}{name}"
                values[output_name] = float(value.item())
                weights[output_name] = float(payload[f"{name}__count"].item())

        completed = output.completed_clearance.detach()
        predicted = output.predicted_clearance.detach()
        metric_variants = {"": completed}
        if report_prediction_metrics:
            metric_variants.update(
                {
                    "blend25_": torch.lerp(completed, predicted, 0.25),
                    "blend50_": torch.lerp(completed, predicted, 0.50),
                    "blend75_": torch.lerp(completed, predicted, 0.75),
                    "predicted_": predicted,
                    "sharpen50_": sharpen_terrain_prediction(predicted, strength=0.50),
                    "sharpen100_": sharpen_terrain_prediction(predicted, strength=1.00),
                    "sharpen200_": sharpen_terrain_prediction(predicted, strength=2.00),
                }
            )
        for variant_prefix, terrain_output in metric_variants.items():
            variant_metrics = terrain_completion_metrics(
                terrain_output,
                target,
                current_visible=output.current_visible,
                history_visible=history.visible_masks.any(dim=1),
                include_counts=True,
            )
            add_metrics(variant_prefix, variant_metrics)
        stairs = torch.isin(
            batch["terrain_type"].to(device),
            torch.tensor(stairs_terrain_ids, device=device),
        )
        if torch.any(stairs):
            for variant_prefix, terrain_output in metric_variants.items():
                stair_metrics = terrain_completion_metrics(
                    terrain_output[stairs],
                    target[stairs],
                    current_visible=output.current_visible[stairs],
                    history_visible=history.visible_masks.any(dim=1)[stairs],
                    include_counts=True,
                )
                add_metrics(f"{variant_prefix}stairs_", stair_metrics)
        terrain_ids = batch["terrain_type"].to(device)
        for terrain_id, terrain_name in enumerate(terrain_names):
            selected = terrain_ids == terrain_id
            if not torch.any(selected):
                continue
            prefix = str(terrain_name).replace("-", "_")
            for variant_prefix, terrain_output in metric_variants.items():
                terrain_metrics = terrain_completion_metrics(
                    terrain_output[selected],
                    target[selected],
                    current_visible=output.current_visible[selected],
                    history_visible=history.visible_masks.any(dim=1)[selected],
                    include_counts=True,
                )
                add_metrics(f"{variant_prefix}terrain_{prefix}_", terrain_metrics)
        for name, value in values.items():
            weight = weights[name]
            totals[name] = totals.get(name, 0.0) + value * weight
            metric_counts[name] = metric_counts.get(name, 0.0) + weight
        batches += 1
    if batches == 0:
        raise ValueError("terrain perception data loader contains no batches")
    result = {
        name: (value / metric_counts[name] if metric_counts[name] > 0.0 else 0.0)
        for name, value in totals.items()
    }
    if report_metric_counts:
        result.update({f"{name}__count": count for name, count in metric_counts.items()})
    return result


def train_terrain_perception(
    *,
    dataset_dir: Path,
    output_dir: Path,
    validation_dataset_dir: Path | None,
    sequence_steps: int,
    history_seconds: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_channels: int,
    proprio_channels: int = 8,
    device: str,
    seed: int,
    stairs_terrain_ids: tuple[int, ...] | None = None,
    history_mode: str = "egomotion_warp",
    loss_mode: str = "baseline",
    num_workers: int = 0,
    samples_per_epoch: int | None = None,
    resume_checkpoint: Path | None = None,
    init_checkpoint: Path | None = None,
    train_terrain_ids: tuple[int, ...] | None = None,
    use_grid_coordinates: bool = False,
    global_context_dim: int = 0,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if history_mode not in {"egomotion_warp", "no_odometry"}:
        raise ValueError("history_mode must be 'egomotion_warp' or 'no_odometry'")
    if loss_mode not in {"baseline", "phase2i_v2"}:
        raise ValueError("loss_mode must be 'baseline' or 'phase2i_v2'")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if global_context_dim < 0:
        raise ValueError("global_context_dim must be non-negative")
    if resume_checkpoint is not None and init_checkpoint is not None:
        raise ValueError("resume_checkpoint and init_checkpoint are mutually exclusive")
    if train_terrain_ids is not None:
        train_terrain_ids = tuple(sorted({int(value) for value in train_terrain_ids}))
        if not train_terrain_ids or any(value < 0 for value in train_terrain_ids):
            raise ValueError("train_terrain_ids must contain non-negative terrain IDs")
    loss_config = TerrainCompletionLossConfig() if loss_mode == "phase2i_v2" else None
    full_dataset = TerrainPerceptionSequenceDataset(
        dataset_dir,
        sequence_steps=sequence_steps,
        history_seconds=history_seconds,
    )
    if full_dataset.odometry_free != (history_mode == "no_odometry"):
        expected = "no_odometry" if full_dataset.odometry_free else "egomotion_warp"
        raise ValueError(f"dataset schema requires history_mode={expected!r}; got {history_mode!r}")
    component_names = tuple(str(name) for name in full_dataset.metadata.get("terrain_component_names", ()))
    if stairs_terrain_ids is None:
        stairs_terrain_ids = tuple(index for index, name in enumerate(component_names) if str(name).startswith("stairs"))
        if not stairs_terrain_ids:
            stairs_terrain_ids = (2,)
    if not stairs_terrain_ids or any(index < 0 for index in stairs_terrain_ids):
        raise ValueError("stairs_terrain_ids must contain non-negative terrain IDs")
    if validation_dataset_dir is not None:
        train_dataset: Dataset = full_dataset
        validation_dataset: Dataset = TerrainPerceptionSequenceDataset(
            validation_dataset_dir,
            sequence_steps=sequence_steps,
            history_seconds=history_seconds,
        )
        if validation_dataset.odometry_free != full_dataset.odometry_free:
            raise ValueError("training and validation datasets must use the same odometry schema")
        validation_source = "independent_dataset"
    else:
        if train_terrain_ids is not None:
            raise ValueError("train_terrain_ids requires an independent validation dataset")
        if len(full_dataset) < 10:
            raise ValueError("at least 10 sequences are required for an internal train/validation split")
        validation_size = max(1, round(len(full_dataset) * 0.1))
        train_dataset, validation_dataset = random_split(
            full_dataset,
            [len(full_dataset) - validation_size, validation_size],
            generator=torch.Generator().manual_seed(seed),
        )
        validation_source = "deterministic_10_percent_sample_split"
    if train_terrain_ids is not None:
        selected_indices = full_dataset.sample_indices_for_terrain_ids(train_terrain_ids)
        if not selected_indices:
            raise ValueError(f"training dataset has no sequences for terrain IDs {train_terrain_ids}")
        train_dataset = Subset(full_dataset, selected_indices)
    if samples_per_epoch is not None and not 0 < samples_per_epoch <= len(train_dataset):
        raise ValueError("samples_per_epoch must lie in [1, len(train_dataset)]")

    worker_options = {
        "num_workers": num_workers,
        **(
            {"persistent_workers": True, "prefetch_factor": 2}
            if num_workers > 0
            else {}
        ),
    }
    train_sampler = ChunkGroupedShuffleSampler(
        train_dataset,
        seed=seed,
        num_samples=samples_per_epoch,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        **worker_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        **worker_options,
    )
    torch_device = torch.device(device)
    model = TemporalTerrainCompletion(
        hidden_channels=hidden_channels,
        proprio_dim=full_dataset.proprio_dim,
        proprio_channels=proprio_channels,
        motion_feature_dim=1 if history_mode == "no_odometry" else 6,
        use_grid_coordinates=use_grid_coordinates,
        global_context_dim=global_context_dim,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_validation = float("inf")
    start_epoch = 1
    resumed_from: str | None = None
    initialized_from: str | None = None
    resume_best_validation_loss: float | None = None
    if init_checkpoint is not None:
        init_checkpoint = init_checkpoint.resolve()
        if not init_checkpoint.is_file():
            raise FileNotFoundError(f"initialization checkpoint does not exist: {init_checkpoint}")
        checkpoint = torch.load(init_checkpoint, map_location=torch_device, weights_only=False)
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError("initialization checkpoint is missing its training config")
        expected_config = {
            "hidden_channels": hidden_channels,
            "proprio_channels": proprio_channels,
            "proprio_dim": full_dataset.proprio_dim,
            "sequence_steps": sequence_steps,
            "history_seconds": history_seconds,
            "history_mode": history_mode,
            "loss_mode": loss_mode,
        }
        mismatches = {
            name: {"checkpoint": checkpoint_config.get(name), "requested": expected}
            for name, expected in expected_config.items()
            if checkpoint_config.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"initialization checkpoint config mismatch: {mismatches}")
        source_uses_coordinates = bool(checkpoint_config.get("use_grid_coordinates", False))
        source_global_context_dim = int(checkpoint_config.get("global_context_dim", 0))
        source_state = checkpoint["model"]
        expanded_state = dict(source_state)
        if source_uses_coordinates == use_grid_coordinates:
            pass
        elif use_grid_coordinates and not source_uses_coordinates:
            target_state = model.state_dict()
            hidden = model.recurrent.hidden_channels
            for name in ("recurrent.gates.weight", "recurrent.candidate.weight"):
                source_weight = source_state[name]
                target_weight = target_state[name].clone().zero_()
                source_inputs = source_weight.shape[1] - hidden
                target_inputs = target_weight.shape[1] - hidden
                if target_inputs != source_inputs + 2:
                    raise ValueError("coordinate expansion expected exactly two new recurrent input channels")
                target_weight[:, :source_inputs] = source_weight[:, :source_inputs]
                target_weight[:, target_inputs:] = source_weight[:, source_inputs:]
                expanded_state[name] = target_weight
        else:
            raise ValueError("cannot initialize a coordinate-free model from a coordinate-aware checkpoint")
        if source_global_context_dim == global_context_dim:
            model.load_state_dict(expanded_state)
        elif source_global_context_dim == 0 and global_context_dim > 0:
            incompatible = model.load_state_dict(expanded_state, strict=False)
            expected_missing = {name for name in model.state_dict() if name.startswith("global_head.")}
            if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise ValueError(
                    "global-context expansion had unexpected state mismatch: "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
        else:
            raise ValueError(
                "cannot change a nonzero global-context dimension during model-only initialization"
            )
        initialized_from = str(init_checkpoint)
        print(
            json.dumps(
                {
                    "initialize": initialized_from,
                    "source_epoch": int(checkpoint.get("epoch", 0)),
                    "optimizer": "fresh_adamw",
                    "train_terrain_ids": train_terrain_ids,
                    "coordinate_expansion": use_grid_coordinates and not source_uses_coordinates,
                    "global_context_expansion": global_context_dim > 0 and source_global_context_dim == 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=torch_device, weights_only=False)
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError("resume checkpoint is missing its training config")
        expected_config = {
            "hidden_channels": hidden_channels,
            "proprio_channels": proprio_channels,
            "proprio_dim": full_dataset.proprio_dim,
            "sequence_steps": sequence_steps,
            "history_seconds": history_seconds,
            "history_mode": history_mode,
            "loss_mode": loss_mode,
            "samples_per_epoch": samples_per_epoch,
            "train_terrain_ids": train_terrain_ids,
            "use_grid_coordinates": use_grid_coordinates,
            "global_context_dim": global_context_dim,
        }
        mismatches = {
            name: {"checkpoint": checkpoint_config.get(name), "requested": expected}
            for name, expected in expected_config.items()
            if checkpoint_config.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"resume checkpoint config mismatch: {mismatches}")
        completed_epoch = int(checkpoint.get("epoch", 0))
        if completed_epoch <= 0:
            raise ValueError("resume checkpoint must contain a positive completed epoch")
        if completed_epoch >= epochs:
            raise ValueError(
                f"resume checkpoint already reached epoch {completed_epoch}, requested total epochs={epochs}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = completed_epoch + 1
        train_sampler.epoch = completed_epoch
        resumed_from = str(resume_checkpoint)

        best_checkpoint_path = output_dir / "best.pt"
        if best_checkpoint_path.is_file():
            best_checkpoint = torch.load(best_checkpoint_path, map_location=torch_device, weights_only=False)
            model.load_state_dict(best_checkpoint["model"])
            with torch.no_grad():
                best_metrics = _run_epoch(
                    model,
                    validation_loader,
                    device=torch_device,
                    history_seconds=history_seconds,
                    optimizer=None,
                    stairs_terrain_ids=stairs_terrain_ids,
                    history_mode=history_mode,
                    loss_config=loss_config,
                    terrain_names=component_names,
                )
            best_validation = float(best_metrics["loss"])
            resume_best_validation_loss = best_validation
            model.load_state_dict(checkpoint["model"])
        elif "best_validation_loss" in checkpoint:
            best_validation = float(checkpoint["best_validation_loss"])
            resume_best_validation_loss = best_validation
        print(
            json.dumps(
                {
                    "resume": resumed_from,
                    "completed_epoch": completed_epoch,
                    "next_epoch": start_epoch,
                    "best_validation_loss": resume_best_validation_loss,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=torch_device,
            history_seconds=history_seconds,
            optimizer=optimizer,
            stairs_terrain_ids=stairs_terrain_ids,
            history_mode=history_mode,
            loss_config=loss_config,
            terrain_names=component_names,
        )
        with torch.no_grad():
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device=torch_device,
                history_seconds=history_seconds,
                optimizer=None,
                stairs_terrain_ids=stairs_terrain_ids,
                history_mode=history_mode,
                loss_config=loss_config,
                terrain_names=component_names,
            )
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        is_best = validation_metrics["loss"] < best_validation
        if is_best:
            best_validation = validation_metrics["loss"]
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation,
            "config": {
                "hidden_channels": hidden_channels,
                "proprio_channels": proprio_channels,
                "proprio_dim": full_dataset.proprio_dim,
                "sequence_steps": sequence_steps,
                "history_seconds": history_seconds,
                "stairs_terrain_ids": stairs_terrain_ids,
                "history_mode": history_mode,
                "motion_feature_dim": 1 if history_mode == "no_odometry" else 6,
                "dataset_schema": "odometry_free_local" if full_dataset.odometry_free else "world_pose",
                "dataset_metadata": full_dataset.metadata,
                "loss_mode": loss_mode,
                "loss_config": None if loss_config is None else loss_config.__dict__,
                "num_workers": num_workers,
                "samples_per_epoch": samples_per_epoch,
                "train_terrain_ids": train_terrain_ids,
                "use_grid_coordinates": use_grid_coordinates,
                "global_context_dim": global_context_dim,
                "terrain_output_mode": "predicted" if global_context_dim > 0 else "completed",
            },
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if is_best:
            torch.save(checkpoint, output_dir / "best.pt")

    summary = {
        "dataset_dir": str(dataset_dir.resolve()),
        "validation_dataset_dir": (str(validation_dataset_dir.resolve()) if validation_dataset_dir is not None else None),
        "validation_source": validation_source,
        "train_sequences": len(train_dataset),
        "validation_sequences": len(validation_dataset),
        "history": history,
        "best_validation_loss": best_validation,
        "stairs_terrain_ids": list(stairs_terrain_ids),
        "history_mode": history_mode,
        "loss_mode": loss_mode,
        "num_workers": num_workers,
        "samples_per_epoch": samples_per_epoch or len(train_dataset),
        "train_terrain_ids": None if train_terrain_ids is None else list(train_terrain_ids),
        "use_grid_coordinates": use_grid_coordinates,
        "global_context_dim": global_context_dim,
        "terrain_output_mode": "predicted" if global_context_dim > 0 else "completed",
        "start_epoch": start_epoch,
        "resumed_from": resumed_from,
        "initialized_from": initialized_from,
        "resume_best_validation_loss": resume_best_validation_loss,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--validation-dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-steps", type=int, default=31)
    parser.add_argument("--history-seconds", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--proprio-channels", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--use-grid-coordinates",
        action="store_true",
        help="Append fixed normalized robot-centric X/Y channels internally; external inputs remain unchanged.",
    )
    parser.add_argument(
        "--global-context-dim",
        type=int,
        default=0,
        help="Add an internal low-rank whole-grid residual head; zero preserves the legacy local head.",
    )
    parser.add_argument(
        "--train-terrain-ids",
        type=int,
        nargs="+",
        default=None,
        help="Restrict training sequences by endpoint terrain ID while retaining full independent validation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--history-mode",
        choices=("egomotion_warp", "no_odometry"),
        default="egomotion_warp",
        help="Use explicit pelvis egomotion warp or leave robot-centric history unwarped.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("baseline", "phase2i_v2"),
        default="baseline",
        help="Keep the v1 loss for ablations or enable the task-aligned v2 objective.",
    )
    parser.add_argument(
        "--stairs-terrain-ids",
        type=int,
        nargs="+",
        default=None,
        help="Terrain IDs aggregated into stairs metrics; inferred from dataset metadata by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_terrain_perception(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        validation_dataset_dir=args.validation_dataset_dir,
        sequence_steps=args.sequence_steps,
        history_seconds=args.history_seconds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        proprio_channels=args.proprio_channels,
        device=args.device,
        seed=args.seed,
        stairs_terrain_ids=(tuple(args.stairs_terrain_ids) if args.stairs_terrain_ids is not None else None),
        history_mode=args.history_mode,
        loss_mode=args.loss_mode,
        num_workers=args.num_workers,
        samples_per_epoch=args.samples_per_epoch,
        resume_checkpoint=args.resume_checkpoint,
        init_checkpoint=args.init_checkpoint,
        train_terrain_ids=(tuple(args.train_terrain_ids) if args.train_terrain_ids is not None else None),
        use_grid_coordinates=args.use_grid_coordinates,
        global_context_dim=args.global_context_dim,
    )


if __name__ == "__main__":
    main()
