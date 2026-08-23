"""Evaluate a frozen temporal terrain completion checkpoint on one dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from humanoidverse.perception.terrain_dataset import TerrainPerceptionSequenceDataset
from humanoidverse.perception.temporal_terrain import TemporalTerrainCompletion
from humanoidverse.train_terrain_perception import _run_epoch


def evaluate_terrain_perception(
    *,
    checkpoint_path: Path,
    dataset_dir: Path,
    output_path: Path,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    sequence_steps = int(config["sequence_steps"])
    history_seconds = float(config["history_seconds"])
    dataset = TerrainPerceptionSequenceDataset(
        dataset_dir,
        sequence_steps=sequence_steps,
        history_seconds=history_seconds,
    )
    component_names = tuple(dataset.metadata.get("terrain_component_names", ()))
    stairs_terrain_ids = tuple(
        index for index, name in enumerate(component_names) if str(name).startswith("stairs")
    ) or tuple(int(index) for index in config.get("stairs_terrain_ids", (2,)))
    model = TemporalTerrainCompletion(
        hidden_channels=int(config["hidden_channels"]),
        proprio_dim=int(config["proprio_dim"]),
    )
    model.load_state_dict(checkpoint["model"])
    torch_device = torch.device(device)
    model.to(torch_device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        metrics = _run_epoch(
            model,
            loader,
            device=torch_device,
            history_seconds=history_seconds,
            optimizer=None,
            stairs_terrain_ids=stairs_terrain_ids,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_terrain_perception(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
