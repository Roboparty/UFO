"""Aggregate six-terrain Phase 2D nominal-noise regression results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CONDITIONS = ("clean", "measurement", "dropout", "edge", "latency", "extrinsic", "combined")
TERRAINS = ("flat", "slope", "rough", "platforms", "stairs_down", "stairs_up")
METRICS = (
    "fell_rate",
    "traversal_success_rate",
    "mean_body_impact_mean",
    "max_body_impact_mean",
    "terrain_input_mae_mean",
    "underfoot_mae_mean",
    "stairs_edge_mae_mean",
    "current_visible_fraction_mean",
    "temporal_coverage_fraction_mean",
    "action_deviation_from_clean_mean",
    "sensor_noisy_valid_fraction_mean",
)


def aggregate(input_root: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    invariants: dict[str, set[Any]] = {
        key: set() for key in ("checkpoint_global_time", "z_checksum", "actor_checksum", "perception_checksum", "noise_seed", "episode_steps", "seed")
    }
    paired_initial_states: dict[str, str] = {}
    for condition in CONDITIONS:
        config_hashes: set[str] = set()
        for terrain in TERRAINS:
            path = input_root / condition / terrain / "summary.json"
            if not path.exists():
                raise ValueError(f"missing regression result: {path}")
            summary = json.loads(path.read_text())
            if set(summary["modes"]) != {"temporal"}:
                raise ValueError(f"{path}: expected temporal-only evaluation")
            for key in invariants:
                invariants[key].add(summary[key])
            config_hashes.add(summary["noise_config_hash"])
            initial_checksum = summary["diagnostics"][0]["initial_state_checksum"]
            if terrain in paired_initial_states and paired_initial_states[terrain] != initial_checksum:
                raise ValueError(f"initial state changed across noise conditions for {terrain}")
            paired_initial_states[terrain] = initial_checksum
            metrics = summary["modes"]["temporal"]
            row: dict[str, Any] = {
                "condition": condition,
                "terrain": terrain,
                "noise_config_hash": summary["noise_config_hash"],
            }
            for metric in METRICS:
                row[metric] = metrics.get(metric)
            rows.append(row)
        if len(config_hashes) != 1:
            raise ValueError(f"{condition}: noise config changed across terrains")
    bad = {key: sorted(values, key=str) for key, values in invariants.items() if len(values) != 1}
    if bad:
        raise ValueError(f"regression invariants differ: {bad}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input_root": str(input_root.resolve()),
        "conditions": CONDITIONS,
        "terrains": TERRAINS,
        "invariants": {key: next(iter(values)) for key, values in invariants.items()},
        "results": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
