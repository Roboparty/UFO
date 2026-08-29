"""Evaluate a frozen temporal terrain completion checkpoint on one dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from humanoidverse.perception.temporal_terrain import (
    TerrainCompletionLossConfig,
    TemporalTerrainCompletion,
)
from humanoidverse.perception.terrain_dataset import TerrainPerceptionSequenceDataset
from humanoidverse.train_terrain_perception import _run_epoch


def evaluate_terrain_perception(
    *,
    checkpoint_path: Path,
    dataset_dir: Path,
    output_path: Path,
    batch_size: int,
    device: str,
    num_workers: int = 0,
) -> dict[str, object]:
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    sequence_steps = int(config["sequence_steps"])
    history_seconds = float(config["history_seconds"])
    history_mode = str(config.get("history_mode", "egomotion_warp"))
    dataset = TerrainPerceptionSequenceDataset(
        dataset_dir,
        sequence_steps=sequence_steps,
        history_seconds=history_seconds,
    )
    if dataset.odometry_free != (history_mode == "no_odometry"):
        raise ValueError("perception checkpoint history mode does not match dataset schema")
    component_names = tuple(dataset.metadata.get("terrain_component_names", ()))
    stairs_terrain_ids = tuple(index for index, name in enumerate(component_names) if str(name).startswith("stairs")) or tuple(
        int(index) for index in config.get("stairs_terrain_ids", (2,))
    )
    model = TemporalTerrainCompletion(
        hidden_channels=int(config["hidden_channels"]),
        proprio_dim=int(config["proprio_dim"]),
        proprio_channels=int(config.get("proprio_channels", 8)),
        motion_feature_dim=int(config.get("motion_feature_dim", 6)),
        use_grid_coordinates=bool(config.get("use_grid_coordinates", False)),
        global_context_dim=int(config.get("global_context_dim", 0)),
    )
    model.load_state_dict(checkpoint["model"])
    torch_device = torch.device(device)
    model.to(torch_device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **(
            {"persistent_workers": True, "prefetch_factor": 2}
            if num_workers > 0
            else {}
        ),
    )
    with torch.inference_mode():
        metrics = _run_epoch(
            model,
            loader,
            device=torch_device,
            history_seconds=history_seconds,
            optimizer=None,
            stairs_terrain_ids=stairs_terrain_ids,
            history_mode=history_mode,
            loss_config=(
                TerrainCompletionLossConfig(**dict(config.get("loss_config") or {}))
                if config.get("loss_mode") == "phase2i_v2"
                else None
            ),
            terrain_names=component_names,
            report_metric_counts=True,
            report_prediction_metrics=True,
        )
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "dataset_dir": str(dataset_dir.resolve()),
        "terrain": dataset.metadata.get("terrain"),
        "seed": dataset.metadata.get("seed"),
        "sequences": len(dataset),
        "sequence_steps": sequence_steps,
        "history_seconds": history_seconds,
        "history_mode": history_mode,
        "num_workers": num_workers,
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_terrain_perception(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
