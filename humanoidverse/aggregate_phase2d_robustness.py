"""Aggregate paired Phase 2D stairs-up sensing-robustness evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from humanoidverse.aggregate_stairs_up_closed_loop import _bootstrap_mean_ci

METRICS = (
    "fell",
    "impact_safe",
    "mean_body_impact",
    "max_body_impact",
    "terrain_input_mae",
    "underfoot_mae",
    "stairs_edge_mae",
    "current_visible_fraction",
    "temporal_coverage_fraction",
    "action_deviation_from_clean",
    "sensor_noisy_valid_fraction",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _condition_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and list(path.glob("seed_*/summary.json")))


def aggregate(root: Path, output_dir: Path, *, episodes_per_level: int, expected_seeds: int) -> dict[str, Any]:
    condition_dirs = _condition_dirs(root)
    if not condition_dirs:
        raise ValueError(f"no completed conditions found under {root}")

    paired_ids: dict[tuple[int, int], tuple[int, ...]] = {}
    paired_initial_states: dict[tuple[int, int], str] = {}
    per_seed_rows: list[dict[str, Any]] = []
    balanced_rows: list[dict[str, Any]] = []
    invariants: dict[str, set[Any]] = defaultdict(set)
    condition_metadata: dict[str, dict[str, Any]] = {}

    for condition_dir in condition_dirs:
        summaries = sorted(condition_dir.glob("seed_*/summary.json"))
        if len(summaries) != expected_seeds:
            raise ValueError(f"{condition_dir.name}: expected {expected_seeds} seeds, found {len(summaries)}")
        first = json.loads(summaries[0].read_text())
        condition_metadata[condition_dir.name] = {
            "noise_condition": first["noise_config"]["condition"],
            "noise_severity": first["noise_config"]["severity"],
            "noise_config_hash": first["noise_config_hash"],
        }
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text())
            if set(summary["modes"]) != {"temporal"}:
                raise ValueError(f"{summary_path}: expected temporal-only evaluation")
            for key in ("checkpoint_global_time", "z_checksum", "actor_checksum", "perception_checksum", "episode_steps"):
                invariants[key].add(summary[key])
            invariants["noise_seed"].add(summary["noise_seed"])
            if summary["noise_config_hash"] != condition_metadata[condition_dir.name]["noise_config_hash"]:
                raise ValueError(f"{condition_dir.name}: noise config changed across seeds")
            seed = int(summary["seed"])
            by_level: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in _read_csv(summary_path.parent / "metrics.csv"):
                by_level[int(row["terrain_level"])].append(row)
            if sorted(by_level) != list(range(10)):
                raise ValueError(f"{summary_path}: missing stairs difficulty levels")
            for level, rows in sorted(by_level.items()):
                rows.sort(key=lambda row: int(row["env_index"]))
                selected = rows[:episodes_per_level]
                if len(selected) != episodes_per_level:
                    raise ValueError(f"{summary_path}: level {level} has too few episodes")
                ids = tuple(int(row["env_index"]) for row in selected)
                pair_key = (seed, level)
                if pair_key in paired_ids and paired_ids[pair_key] != ids:
                    raise ValueError(f"unpaired environment ids for seed={seed}, level={level}")
                paired_ids[pair_key] = ids
                for row in selected:
                    state_key = (seed, int(row["env_index"]))
                    checksum = row["initial_state_checksum"]
                    if state_key in paired_initial_states and paired_initial_states[state_key] != checksum:
                        raise ValueError(f"initial state changed across conditions for seed/env={state_key}")
                    paired_initial_states[state_key] = checksum
                successes = sum(row["traversal_success"] == "True" for row in selected)
                aggregate_row: dict[str, Any] = {
                    "condition": condition_dir.name,
                    "seed": seed,
                    "terrain_level": level,
                    "step_height_m": statistics.fmean(float(row["stairs_step_height"]) for row in selected),
                    "episodes": episodes_per_level,
                    "successes": successes,
                    "strict_success_rate": successes / episodes_per_level,
                }
                for metric in METRICS:
                    if metric in {"fell", "impact_safe"}:
                        aggregate_row[metric] = statistics.fmean(row[metric] == "True" for row in selected)
                    else:
                        aggregate_row[metric] = statistics.fmean(float(row[metric]) for row in selected)
                per_seed_rows.append(aggregate_row)
                balanced_rows.extend({"condition": condition_dir.name, "seed": seed, **row} for row in selected)

    bad = {key: sorted(values, key=str) for key, values in invariants.items() if len(values) != 1}
    if bad:
        raise ValueError(f"cross-condition invariants differ: {bad}")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[(str(row["condition"]), int(row["terrain_level"]))].append(row)
    ci_rows: list[dict[str, Any]] = []
    for (condition, level), rows in sorted(grouped.items()):
        seed_rates = [float(row["strict_success_rate"]) for row in rows]
        low, high = _bootstrap_mean_ci(seed_rates, seed=20_000 + level)
        result: dict[str, Any] = {
            "condition": condition,
            "terrain_level": level,
            "step_height_m": statistics.fmean(float(row["step_height_m"]) for row in rows),
            "seed_count": len(rows),
            "episodes": sum(int(row["episodes"]) for row in rows),
            "successes": sum(int(row["successes"]) for row in rows),
            "mean_seed_strict_success_rate": statistics.fmean(seed_rates),
            "ci95_low": low,
            "ci95_high": high,
        }
        for metric in METRICS:
            result[metric] = statistics.fmean(float(row[metric]) for row in rows)
        ci_rows.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("balanced_episodes.csv", balanced_rows), ("per_seed.csv", per_seed_rows), ("step_height_ci.csv", ci_rows)):
        with (output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    import matplotlib.pyplot as plt

    def plot_height_curves(filename: str, selected_conditions: list[str]) -> None:
        figure, axis = plt.subplots(figsize=(7.4, 4.6))
        plotted = False
        for condition in selected_conditions:
            rows = [row for row in ci_rows if row["condition"] == condition]
            if not rows:
                continue
            axis.plot(
                [100.0 * float(row["step_height_m"]) for row in rows],
                [100.0 * float(row["mean_seed_strict_success_rate"]) for row in rows],
                marker="o",
                linewidth=2.0,
                label=condition.replace("_", " "),
            )
            plotted = True
        if not plotted:
            plt.close(figure)
            return
        axis.set_xlabel("Stair height (cm)")
        axis.set_ylabel("Strict traversal success (%)")
        axis.set_ylim(70, 101)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)

    plot_height_curves(
        "success_vs_stair_height_nominal.png",
        ["clean", "measurement_nominal", "dropout_nominal", "edge_nominal", "latency_nominal", "extrinsic_nominal", "combined_nominal"],
    )
    plot_height_curves(
        "success_vs_stair_height_combined.png",
        ["clean", "combined_mild", "combined_nominal", "combined_strong"],
    )

    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    severity_order = ("mild", "nominal", "strong")
    plotted = False
    for noise_type in ("measurement", "dropout", "edge", "latency", "extrinsic", "combined"):
        labels = []
        values = []
        for severity in severity_order:
            rows = [row for row in ci_rows if row["condition"] == f"{noise_type}_{severity}"]
            if not rows:
                continue
            labels.append(severity)
            values.append(100.0 * sum(int(row["successes"]) for row in rows) / sum(int(row["episodes"]) for row in rows))
        if values:
            axis.plot(labels, values, marker="o", linewidth=2.0, label=noise_type)
            plotted = True
    clean_rows = [row for row in ci_rows if row["condition"] == "clean"]
    if clean_rows:
        clean_rate = 100.0 * sum(int(row["successes"]) for row in clean_rows) / sum(
            int(row["episodes"]) for row in clean_rows
        )
        axis.axhline(clean_rate, color="#222222", linestyle="--", linewidth=1.2, label="clean")
        plotted = True
    if plotted:
        axis.set_xlabel("Noise severity")
        axis.set_ylabel("Strict traversal success (%)")
        axis.set_ylim(94, 100.5)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8, ncol=2)
        figure.tight_layout()
        figure.savefig(output_dir / "success_vs_severity.png", dpi=180)
    plt.close(figure)
    summary = {
        "input_root": str(root.resolve()),
        "conditions": condition_metadata,
        "expected_seeds": expected_seeds,
        "episodes_per_seed_level": episodes_per_level,
        "confidence_interval": "seed-clustered nonparametric bootstrap percentile, 20,000 resamples",
        "invariants": {key: next(iter(values)) for key, values in invariants.items()},
        "step_height_results": ci_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-level", type=int, default=12)
    parser.add_argument("--expected-seeds", type=int, default=10)
    args = parser.parse_args()
    aggregate(args.input_root, args.output_dir, episodes_per_level=args.episodes_per_level, expected_seeds=args.expected_seeds)


if __name__ == "__main__":
    main()
