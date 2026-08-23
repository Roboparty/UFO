#!/usr/bin/env python3
"""Evaluate Phase 2C Actor milestones without touching the training process."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

STAIRS_SEEDS = tuple(range(6840, 6850))
REGRESSION_TERRAINS = ("flat", "slope", "rough", "platforms", "stairs_down")


def _evaluation_command(args, *, actor: Path, output: Path, terrain: str, seed: int, num_envs: int) -> list[str]:
    return [
        str(args.python),
        "-m",
        "humanoidverse.terrain_perception_closed_loop",
        "--model-folder",
        str(args.model_folder),
        "--actor-checkpoint",
        str(actor),
        "--perception-checkpoint",
        str(args.perception_checkpoint),
        "--latent",
        str(args.latent),
        "--output-dir",
        str(output),
        "--terrain",
        terrain,
        "--num-envs",
        str(num_envs),
        "--episode-steps",
        "1000",
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--modes",
        "gt",
        "temporal",
    ]


def _run_parallel(jobs: list[tuple[list[str], Path]], gpus: tuple[int, ...]) -> None:
    processes: list[tuple[subprocess.Popen, object, Path]] = []
    for index, (command, log_path) in enumerate(jobs):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpus[index % len(gpus)]),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
            }
        )
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        processes.append((process, stream, log_path))
    failures = []
    for process, stream, log_path in processes:
        return_code = process.wait()
        stream.close()
        if return_code:
            failures.append((str(log_path), return_code))
    if failures:
        raise RuntimeError(f"milestone evaluations failed: {failures}")


def _wait_for_actor(path: Path, poll_seconds: float) -> None:
    while not path.exists():
        time.sleep(poll_seconds)


def _wait_for_adopted_evaluation(output: Path, poll_seconds: float) -> None:
    while len(list(output.glob("seed_*/summary.json"))) < len(STAIRS_SEEDS):
        time.sleep(poll_seconds)


def _evaluate_stairs(args, *, step: int, actor: Path, output: Path) -> None:
    if step in args.adopt_steps:
        _wait_for_adopted_evaluation(output, args.poll_seconds)
    else:
        jobs = []
        for seed in STAIRS_SEEDS:
            seed_output = output / f"seed_{seed}"
            if (seed_output / "summary.json").exists():
                continue
            jobs.append(
                (
                    _evaluation_command(
                        args,
                        actor=actor,
                        output=seed_output,
                        terrain="stairs_up",
                        seed=seed,
                        num_envs=256,
                    ),
                    seed_output / "evaluation.log",
                )
            )
        _run_parallel(jobs, args.gpus)
    aggregate = output / "aggregate"
    subprocess.run(
        [
            str(args.python),
            "-m",
            "humanoidverse.aggregate_stairs_up_closed_loop",
            "--input-root",
            str(output),
            "--output-dir",
            str(aggregate),
            "--episodes-per-level",
            "12",
            "--expected-seeds",
            "10",
        ],
        check=True,
    )


def _evaluate_regressions(args, *, actor: Path, output: Path) -> None:
    jobs = []
    for terrain in REGRESSION_TERRAINS:
        terrain_output = output / "regression" / terrain
        if (terrain_output / "summary.json").exists():
            continue
        jobs.append(
            (
                _evaluation_command(
                    args,
                    actor=actor,
                    output=terrain_output,
                    terrain=terrain,
                    seed=6840,
                    num_envs=64,
                ),
                terrain_output / "evaluation.log",
            )
        )
    _run_parallel(jobs, args.gpus)


def _write_comparison(args) -> None:
    stairs_rows: list[dict[str, object]] = []
    regression_rows: list[dict[str, object]] = []
    roots = [("frozen", args.baseline_stairs)] + [
        (str(step), args.output_root / f"step_{step:06d}" / "aggregate") for step in args.steps
    ]
    for checkpoint, root in roots:
        path = root / "step_height_ci.csv"
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["mode"] == "temporal" and float(row["step_height_m"]) >= 0.14:
                    stairs_rows.append({"checkpoint": checkpoint, **row})
    regression_roots = [("frozen", args.baseline_regression)] + [
        (str(step), args.output_root / f"step_{step:06d}" / "regression") for step in args.steps
    ]
    for checkpoint, root in regression_roots:
        for terrain in REGRESSION_TERRAINS:
            path = root / terrain / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text())
            temporal = summary["modes"]["temporal"]
            regression_rows.append(
                {
                    "checkpoint": checkpoint,
                    "terrain": terrain,
                    "fall_rate": temporal["fell_rate"],
                    "traversal_success_rate": temporal.get("traversal_success_rate"),
                    "mean_body_impact": temporal["mean_body_impact_mean"],
                }
            )
    for name, rows in (("stairs_up_milestones.csv", stairs_rows), ("regression.csv", regression_rows)):
        if rows:
            with (args.output_root / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--perception-checkpoint", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--baseline-stairs", type=Path, required=True)
    parser.add_argument("--baseline-regression", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    parser.add_argument("--adopt-steps", type=int, nargs="*", default=[])
    parser.add_argument("--gpus", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.steps = tuple(args.steps)
    args.adopt_steps = frozenset(args.adopt_steps)
    args.gpus = tuple(args.gpus)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        actor = args.run_dir / "milestones" / f"actor_step_{step:06d}.pt"
        _wait_for_actor(actor, args.poll_seconds)
        output = args.output_root / f"step_{step:06d}"
        output.mkdir(parents=True, exist_ok=True)
        _evaluate_stairs(args, step=step, actor=actor, output=output)
        _evaluate_regressions(args, actor=actor, output=output)
        _write_comparison(args)
        (args.output_root / "progress.json").write_text(
            json.dumps({"last_completed_step": step, "steps": args.steps}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
