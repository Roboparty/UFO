"""Aggregate balanced multi-seed stairs-up closed-loop success with clustered CIs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 20_000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values or not 0.0 < confidence < 1.0 or samples <= 0:
        raise ValueError("invalid bootstrap inputs")
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    lower = means[max(0, int(tail * samples))]
    upper = means[min(samples - 1, int((1.0 - tail) * samples))]
    return lower, upper


def aggregate_stairs_up(
    *,
    input_root: Path,
    output_dir: Path,
    episodes_per_level: int,
    expected_seeds: int,
) -> dict[str, Any]:
    if episodes_per_level <= 0 or expected_seeds <= 0:
        raise ValueError("episodes_per_level and expected_seeds must be positive")
    run_dirs = sorted(path.parent for path in input_root.glob("seed_*/summary.json"))
    if len(run_dirs) != expected_seeds:
        raise ValueError(f"expected {expected_seeds} seed runs, found {len(run_dirs)}")
    summaries = [json.loads((run_dir / "summary.json").read_text()) for run_dir in run_dirs]
    invariants = {
        "checkpoint_global_time": {item["checkpoint_global_time"] for item in summaries},
        "z_checksum": {item["z_checksum"] for item in summaries},
        "perception_checkpoint": {item["perception_checkpoint"] for item in summaries},
        "perception_epoch": {item["perception_epoch"] for item in summaries},
        "camera": {json.dumps(item["camera"], sort_keys=True) for item in summaries},
        "episode_steps": {item["episode_steps"] for item in summaries},
        "action_selection": {item["action_selection"] for item in summaries},
        "terrain": {item["terrain"] for item in summaries},
    }
    bad = {name: sorted(values, key=str) for name, values in invariants.items() if len(values) != 1}
    if bad:
        raise ValueError(f"multi-seed invariants differ: {bad}")
    if next(iter(invariants["terrain"])) != "stairs_up":
        raise ValueError("aggregator only accepts stairs_up evaluations")
    if any(set(item["modes"]) != {"gt", "temporal"} for item in summaries):
        raise ValueError("every run must contain exactly GT and temporal modes")

    balanced: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for run_dir, summary in zip(run_dirs, summaries, strict=True):
        seed = int(summary["seed"])
        rows = _read_csv(run_dir / "metrics.csv")
        by_mode_level: dict[tuple[str, int], dict[int, dict[str, str]]] = defaultdict(dict)
        for row in rows:
            by_mode_level[(row["mode"], int(row["terrain_level"]))][int(row["env_index"])] = row
        levels = sorted({level for _mode, level in by_mode_level})
        if levels != list(range(10)):
            raise ValueError(f"seed {seed} does not cover all ten terrain levels: {levels}")
        for level in levels:
            paired_ids = sorted(
                set(by_mode_level[("gt", level)]) & set(by_mode_level[("temporal", level)])
            )
            if len(paired_ids) < episodes_per_level:
                raise ValueError(
                    f"seed {seed} level {level} has {len(paired_ids)} paired episodes; "
                    f"need {episodes_per_level}"
                )
            selected_ids = paired_ids[:episodes_per_level]
            for mode in ("gt", "temporal"):
                selected = [by_mode_level[(mode, level)][env_id] for env_id in selected_ids]
                successes = sum(row["traversal_success"] == "True" for row in selected)
                step_height = statistics.fmean(float(row["stairs_step_height"]) for row in selected)
                per_seed.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "terrain_level": level,
                        "step_height_m": step_height,
                        "episodes": episodes_per_level,
                        "successes": successes,
                        "strict_success_rate": successes / episodes_per_level,
                    }
                )
                for row in selected:
                    balanced.append({"seed": seed, **row})

    ci_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped[(row["mode"], row["terrain_level"])].append(row)
    for (mode, level), rows in sorted(grouped.items()):
        seed_rates = [float(row["strict_success_rate"]) for row in rows]
        low, high = _bootstrap_mean_ci(seed_rates, seed=10_000 + level + (0 if mode == "gt" else 100))
        total_successes = sum(int(row["successes"]) for row in rows)
        total_episodes = sum(int(row["episodes"]) for row in rows)
        ci_rows.append(
            {
                "mode": mode,
                "terrain_level": level,
                "step_height_m": statistics.fmean(float(row["step_height_m"]) for row in rows),
                "seed_count": len(rows),
                "episodes": total_episodes,
                "successes": total_successes,
                "pooled_strict_success_rate": total_successes / total_episodes,
                "mean_seed_strict_success_rate": statistics.fmean(seed_rates),
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("balanced_episodes.csv", balanced),
        ("per_seed.csv", per_seed),
        ("step_height_ci.csv", ci_rows),
    ):
        with (output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for mode, color, label in (("gt", "#222222", "GT map"), ("temporal", "#1677b8", "Temporal depth")):
        rows = [row for row in ci_rows if row["mode"] == mode]
        x = [100.0 * float(row["step_height_m"]) for row in rows]
        y = [100.0 * float(row["mean_seed_strict_success_rate"]) for row in rows]
        lower = [value - 100.0 * float(row["ci95_low"]) for value, row in zip(y, rows, strict=True)]
        upper = [100.0 * float(row["ci95_high"]) - value for value, row in zip(y, rows, strict=True)]
        axis.errorbar(x, y, yerr=(lower, upper), marker="o", linewidth=2.0, capsize=3, color=color, label=label)
    axis.set_xlabel("Stair height (cm)")
    axis.set_ylabel("Strict traversal success (%)")
    axis.set_ylim(-3, 103)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "stairs_up_success_ci.png", dpi=180)
    plt.close(figure)

    summary = {
        "input_root": str(input_root.resolve()),
        "expected_seeds": expected_seeds,
        "episodes_per_seed_level_mode": episodes_per_level,
        "balanced_episode_count": len(balanced),
        "confidence_interval": "seed-clustered nonparametric bootstrap percentile, 20,000 resamples",
        "invariants": {name: next(iter(values)) for name, values in invariants.items()},
        "step_height_results": ci_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-level", type=int, default=12)
    parser.add_argument("--expected-seeds", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate_stairs_up(
        input_root=args.input_root,
        output_dir=args.output_dir,
        episodes_per_level=args.episodes_per_level,
        expected_seeds=args.expected_seeds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
