"""Supervised training for temporal completion of projected terrain maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, random_split

from humanoidverse.perception.temporal_terrain import (
    TemporalTerrainCompletion,
    terrain_completion_loss,
    terrain_completion_metrics,
    warp_terrain_history_to_current,
)
from humanoidverse.perception.terrain_dataset import TerrainPerceptionSequenceDataset


class ChunkGroupedShuffleSampler(Sampler[int]):
    """Shuffle samples without repeatedly reloading large on-disk chunks."""

    def __init__(self, dataset: Dataset, *, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
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
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        chunk_ids = list(self._groups)
        for group_position in torch.randperm(len(chunk_ids), generator=generator).tolist():
            group = self._groups[chunk_ids[group_position]]
            for sample_position in torch.randperm(len(group), generator=generator).tolist():
                yield group[sample_position]


def _run_epoch(
    model: TemporalTerrainCompletion,
    loader: DataLoader,
    *,
    device: torch.device,
    history_seconds: float,
    optimizer: torch.optim.Optimizer | None,
    stairs_terrain_ids: tuple[int, ...],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    batches = 0
    for batch in loader:
        partial = batch["partial_map"].to(device)
        visible = batch["visible_mask"].to(device)
        pelvis = batch["pelvis_pos_w"].to(device)
        yaw = batch["heading_yaw_w"].to(device)
        timestamps = batch["timestamp_s"].to(device)
        proprio = batch["proprio"].to(device)
        target = batch["gt_terrain_actor"].to(device)
        history = warp_terrain_history_to_current(
            partial,
            visible,
            pelvis,
            yaw,
            timestamps_s=timestamps,
            history_seconds=history_seconds,
            interpolation="bilinear",
        )
        with torch.set_grad_enabled(training):
            output = model(history, proprio=proprio)
            loss = terrain_completion_loss(output.predicted_clearance, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        metrics = terrain_completion_metrics(
            output.completed_clearance.detach(),
            target,
            current_visible=output.current_visible,
        )
        values = {"loss": float(loss.detach().item())}
        values.update({name: float(value.item()) for name, value in metrics.items()})
        stairs = torch.isin(
            batch["terrain_type"].to(device),
            torch.tensor(stairs_terrain_ids, device=device),
        )
        if torch.any(stairs):
            stair_metrics = terrain_completion_metrics(
                output.completed_clearance.detach()[stairs],
                target[stairs],
                current_visible=output.current_visible[stairs],
            )
            values.update({f"stairs_{name}": float(value.item()) for name, value in stair_metrics.items()})
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value
            metric_counts[name] = metric_counts.get(name, 0) + 1
        batches += 1
    if batches == 0:
        raise ValueError("terrain perception data loader contains no batches")
    return {name: value / metric_counts[name] for name, value in totals.items()}


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
    device: str,
    seed: int,
    stairs_terrain_ids: tuple[int, ...] | None = None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    full_dataset = TerrainPerceptionSequenceDataset(
        dataset_dir,
        sequence_steps=sequence_steps,
        history_seconds=history_seconds,
    )
    if stairs_terrain_ids is None:
        component_names = tuple(full_dataset.metadata.get("terrain_component_names", ()))
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
        validation_source = "independent_dataset"
    else:
        if len(full_dataset) < 10:
            raise ValueError("at least 10 sequences are required for an internal train/validation split")
        validation_size = max(1, round(len(full_dataset) * 0.1))
        train_dataset, validation_dataset = random_split(
            full_dataset,
            [len(full_dataset) - validation_size, validation_size],
            generator=torch.Generator().manual_seed(seed),
        )
        validation_source = "deterministic_10_percent_sample_split"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=ChunkGroupedShuffleSampler(train_dataset, seed=seed),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    torch_device = torch.device(device)
    model = TemporalTerrainCompletion(
        hidden_channels=hidden_channels,
        proprio_dim=full_dataset.proprio_dim,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_validation = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=torch_device,
            history_seconds=history_seconds,
            optimizer=optimizer,
            stairs_terrain_ids=stairs_terrain_ids,
        )
        with torch.no_grad():
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device=torch_device,
                history_seconds=history_seconds,
                optimizer=None,
                stairs_terrain_ids=stairs_terrain_ids,
            )
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": {
                "hidden_channels": hidden_channels,
                "proprio_dim": full_dataset.proprio_dim,
                "sequence_steps": sequence_steps,
                "history_seconds": history_seconds,
                "stairs_terrain_ids": stairs_terrain_ids,
            },
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
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
        device=args.device,
        seed=args.seed,
        stairs_terrain_ids=(tuple(args.stairs_terrain_ids) if args.stairs_terrain_ids is not None else None),
    )


if __name__ == "__main__":
    main()
