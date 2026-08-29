#!/usr/bin/env python3
"""Run paired Actor2000 versus Phase 2H step5000 behavior-retention evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import torch

TRACKING_SCENARIOS = (
    ("tracking_walk", 159),
    ("tracking_run", 123),
    ("tracking_getup_prone", 2),
    ("tracking_getup_side", 373),
    ("tracking_getup_supine", 723),
    ("tracking_fight", 838),
)
REWARD_SCENARIOS = (
    ("reward_forward", "move-ego-0-0.7"),
    ("reward_lateral", "move-ego-90-0.3"),
    ("reward_rotate", "rotate-z-5-0.5"),
    ("reward_crouch", "crouch-0.25"),
)


@dataclass(frozen=True)
class Job:
    command: tuple[str, ...]
    output: Path
    log: Path


def _tensor_checksum(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MUJOCO_EGL_DEVICE_ID": "0",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "OMP_NUM_THREADS": "1",
        }
    )
    return environment


def _run_jobs(jobs: list[Job], gpus: tuple[int, ...]) -> None:
    pending = list(jobs)
    active: dict[int, tuple[subprocess.Popen, object, Job]] = {}
    failures: list[tuple[str, int]] = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            if job.output.exists():
                continue
            job.log.parent.mkdir(parents=True, exist_ok=True)
            stream = job.log.open("w")
            process = subprocess.Popen(
                job.command,
                cwd=Path(__file__).resolve().parents[1],
                env=_environment(gpu),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = (process, stream, job)
        for gpu, (process, stream, job) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del active[gpu]
            if return_code:
                failures.append((str(job.log), return_code))
        if failures:
            for process, stream, _job in active.values():
                process.terminate()
                process.wait()
                stream.close()
            raise RuntimeError(f"behavior-retention jobs failed: {failures}")
        if active:
            time.sleep(1.0)


def _model_view(output_root: Path, source_model: Path) -> Path:
    view = output_root / "reward_latent_model_view"
    view.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint", "config.json"):
        link = view / name
        target = source_model / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=target.is_dir())
    return view


def _start_reward_latent_preparation(args) -> tuple[subprocess.Popen | None, object | None, Path]:
    latent_dir = args.output_root / "reward_latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    if args.forward_latent is not None:
        target = latent_dir / "reward_forward.pt"
        if not target.exists():
            payload = torch.load(args.forward_latent, map_location="cpu", weights_only=True)
            if (
                not isinstance(payload, dict)
                or payload.get("prompt_type") != "reward"
                or payload.get("prompt_identifier") != "move-ego-0-0.7"
            ):
                raise ValueError(f"invalid fixed forward latent: {args.forward_latent}")
            z = payload.get("z")
            if not isinstance(z, torch.Tensor) or payload.get("z_checksum") != _tensor_checksum(z):
                raise ValueError(f"forward latent checksum mismatch: {args.forward_latent}")
            torch.save(payload, target)
    missing = [task for name, task in REWARD_SCENARIOS if not (latent_dir / f"{name}.pt").exists()]
    if not missing:
        return None, None, latent_dir
    view = _model_view(args.output_root, args.model_folder)
    log = args.output_root / "logs" / "reward_latent_preparation.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("w")
    command = (
        str(args.python),
        "-m",
        "humanoidverse.reward_inference",
        "--model-folder",
        str(view),
        "--device",
        "cuda:0",
        "--skip-rollouts",
        "true",
        "--disable-dr",
        "true",
        "--disable-obs-noise",
        "true",
        "--buffer-path",
        str(args.buffer_path),
        "--num-samples",
        str(args.reward_num_samples),
        "--max-workers",
        str(args.reward_workers),
        "--process-executor",
        "true",
        "--tasks",
        *missing,
    )
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=_environment(args.gpus[0]),
        stdout=stream,
        stderr=subprocess.STDOUT,
    )
    return process, stream, latent_dir


def _finish_reward_latent_preparation(
    args,
    process: subprocess.Popen | None,
    stream,
    latent_dir: Path,
) -> None:
    if process is not None:
        return_code = process.wait()
        assert stream is not None
        stream.close()
        if return_code:
            raise RuntimeError(
                f"reward latent preparation failed; see {args.output_root / 'logs' / 'reward_latent_preparation.log'}"
            )
    payload_path = args.output_root / "reward_latent_model_view" / "reward_inference" / "reward_locomotion.pkl"
    generated = joblib.load(payload_path) if payload_path.exists() else {}
    for name, task in REWARD_SCENARIOS:
        target = latent_dir / f"{name}.pt"
        if target.exists():
            continue
        values = generated.get(task)
        if not values:
            raise RuntimeError(f"reward latent was not generated for {task!r}")
        z = torch.as_tensor(values[0]).detach().cpu()
        torch.save(
            {
                "z": z,
                "prompt_type": "reward",
                "prompt_identifier": task,
                "z_checksum": _tensor_checksum(z),
            },
            target,
        )


def _common_command(args, *, actor: Path, comparison: Path, seed: int, output: Path) -> list[str]:
    return [
        str(args.python),
        "-m",
        "humanoidverse.terrain_transfer_inference",
        "--model-folder",
        str(args.model_folder),
        "--device",
        "cuda:0",
        "--terrains",
        "flat",
        "--seed",
        str(seed),
        "--fall-clearance",
        "0.45",
        "--actor-override",
        str(actor),
        "--comparison-actor",
        str(comparison),
        "--perception-checkpoint",
        str(args.perception_checkpoint),
        "--output",
        str(output),
    ]


def _tracking_jobs(args) -> list[Job]:
    jobs = []
    actors = (("actor2000", args.actor2000, args.actor5000), ("step5000", args.actor5000, args.actor2000))
    for actor_name, actor, comparison in actors:
        for scenario, motion_id in TRACKING_SCENARIOS:
            for seed in args.seeds:
                output = args.output_root / "rollouts" / actor_name / scenario / f"seed_{seed}.json"
                command = _common_command(
                    args,
                    actor=actor,
                    comparison=comparison,
                    seed=seed,
                    output=output,
                )
                command.extend(
                    [
                        "--prompt-type",
                        "tracking",
                        "--motion-id",
                        str(motion_id),
                        "--episode-length",
                        str(args.tracking_steps),
                    ]
                )
                jobs.append(Job(tuple(command), output, output.with_suffix(".log")))
    return jobs


def _reward_jobs(args, latent_dir: Path) -> list[Job]:
    jobs = []
    actors = (("actor2000", args.actor2000, args.actor5000), ("step5000", args.actor5000, args.actor2000))
    for actor_name, actor, comparison in actors:
        for scenario, task in REWARD_SCENARIOS:
            latent = latent_dir / f"{scenario}.pt"
            for seed in args.seeds:
                output = args.output_root / "rollouts" / actor_name / scenario / f"seed_{seed}.json"
                command = _common_command(
                    args,
                    actor=actor,
                    comparison=comparison,
                    seed=seed,
                    output=output,
                )
                command.extend(
                    [
                        "--prompt-type",
                        "reward",
                        "--reward-task",
                        task,
                        "--load-latent",
                        str(latent),
                        "--episode-length",
                        str(args.reward_steps),
                    ]
                )
                jobs.append(Job(tuple(command), output, output.with_suffix(".log")))
    return jobs


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _task_success(scenario: str, row: dict) -> bool:
    if bool(row["fell"]) or not bool(row["rollout_completed"]):
        return False
    if scenario.startswith("tracking_getup"):
        return float(row["final_ground_clearance"]) >= 0.55 and float(row["final_upright_score"]) >= 0.75
    return True


def _aggregate(args) -> dict:
    raw_rows = []
    scenarios = [name for name, _value in (*TRACKING_SCENARIOS, *REWARD_SCENARIOS)]
    for actor_name in ("actor2000", "step5000"):
        for scenario in scenarios:
            for seed in args.seeds:
                path = args.output_root / "rollouts" / actor_name / scenario / f"seed_{seed}.json"
                payload = json.loads(path.read_text())
                if len(payload) != 1:
                    raise RuntimeError(f"expected one flat-terrain result in {path}")
                row = payload[0]
                row.update({"actor": actor_name, "scenario": scenario, "task_success": _task_success(scenario, row)})
                raw_rows.append(row)

    for scenario in scenarios:
        for seed in args.seeds:
            pair = [row for row in raw_rows if row["scenario"] == scenario and row["seed"] == seed]
            if len(pair) != 2 or len({row["z_checksum"] for row in pair}) != 1:
                raise RuntimeError(f"paired latent mismatch for scenario={scenario}, seed={seed}")

    summary_rows = []
    for actor_name in ("actor2000", "step5000"):
        for scenario in scenarios:
            rows = [row for row in raw_rows if row["actor"] == actor_name and row["scenario"] == scenario]
            summary_rows.append(
                {
                    "actor": actor_name,
                    "scenario": scenario,
                    "episodes": len(rows),
                    "task_success_rate": sum(bool(row["task_success"]) for row in rows) / len(rows),
                    "fall_rate": sum(bool(row["fell"]) for row in rows) / len(rows),
                    "rollout_completion_rate": sum(bool(row["rollout_completed"]) for row in rows) / len(rows),
                    "mean_tracking_error": _mean(rows, "mean_tracking_error"),
                    "mean_prompt_value": _mean(rows, "mean_prompt_value"),
                    "mean_action_l2_deviation": _mean(rows, "mean_action_l2_deviation"),
                    "mean_action_abs_deviation": _mean(rows, "mean_action_abs_deviation"),
                    "mean_final_ground_clearance": _mean(rows, "final_ground_clearance"),
                    "mean_final_upright_score": _mean(rows, "final_upright_score"),
                    "mean_max_body_impact": _mean(rows, "max_body_impact"),
                }
            )

    paired_rows = []
    for scenario in scenarios:
        old = next(row for row in summary_rows if row["actor"] == "actor2000" and row["scenario"] == scenario)
        new = next(row for row in summary_rows if row["actor"] == "step5000" and row["scenario"] == scenario)
        paired_rows.append(
            {
                "scenario": scenario,
                "task_success_change": new["task_success_rate"] - old["task_success_rate"],
                "fall_rate_change": new["fall_rate"] - old["fall_rate"],
                "tracking_error_change": (
                    None
                    if old["mean_tracking_error"] is None
                    else new["mean_tracking_error"] - old["mean_tracking_error"]
                ),
                "prompt_value_change": new["mean_prompt_value"] - old["mean_prompt_value"],
                "step5000_action_l2_deviation": new["mean_action_l2_deviation"],
            }
        )

    tracking_rows = [row for row in paired_rows if row["scenario"].startswith("tracking_")]
    behavior_retained = all(row["fall_rate_change"] <= 0.0 and row["task_success_change"] >= 0.0 for row in paired_rows)
    behavior_retained = behavior_retained and all(
        row["tracking_error_change"] is None or row["tracking_error_change"] <= 0.02 for row in tracking_rows
    )
    return {
        "protocol": {
            "actors": {"actor2000": str(args.actor2000), "step5000": str(args.actor5000)},
            "perception_checkpoint": str(args.perception_checkpoint),
            "seeds": list(args.seeds),
            "terrain": "flat",
            "deterministic_mean_action": True,
            "paired_initial_conditions": True,
        },
        "behavior_retained": behavior_retained,
        "scenario_summary": summary_rows,
        "paired_summary": paired_rows,
        "raw_results": raw_rows,
    }


def _write_outputs(args, summary: dict) -> None:
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for name, rows in (
        ("scenario_summary.csv", summary["scenario_summary"]),
        ("paired_summary.csv", summary["paired_summary"]),
        ("raw_results.csv", summary["raw_results"]),
    ):
        with (args.output_root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--perception-checkpoint", type=Path, required=True)
    parser.add_argument("--actor2000", type=Path, required=True)
    parser.add_argument("--actor5000", type=Path, required=True)
    parser.add_argument("--buffer-path", type=Path, required=True)
    parser.add_argument("--forward-latent", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[9100, 9101, 9102])
    parser.add_argument("--tracking-steps", type=int, default=300)
    parser.add_argument("--reward-steps", type=int, default=500)
    parser.add_argument("--reward-num-samples", type=int, default=100000)
    parser.add_argument("--reward-workers", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_folder = args.model_folder.expanduser().resolve()
    args.perception_checkpoint = args.perception_checkpoint.expanduser().resolve()
    args.actor2000 = args.actor2000.expanduser().resolve()
    args.actor5000 = args.actor5000.expanduser().resolve()
    args.buffer_path = args.buffer_path.expanduser().resolve()
    args.forward_latent = args.forward_latent.expanduser().resolve() if args.forward_latent is not None else None
    args.output_root = args.output_root.expanduser().resolve()
    args.python = args.python.expanduser()
    if not args.python.is_absolute():
        args.python = (Path.cwd() / args.python).absolute()
    args.gpus = tuple(args.gpus)
    args.seeds = tuple(args.seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    required_paths = (
        args.model_folder,
        args.perception_checkpoint,
        args.actor2000,
        args.actor5000,
        args.buffer_path,
        args.python,
    )
    for path in (*required_paths, *(() if args.forward_latent is None else (args.forward_latent,))):
        if not path.exists():
            raise FileNotFoundError(path)

    prep_process, prep_stream, latent_dir = _start_reward_latent_preparation(args)
    tracking_gpus = args.gpus[1:] if prep_process is not None and len(args.gpus) > 1 else args.gpus
    _run_jobs(_tracking_jobs(args), tracking_gpus)
    _finish_reward_latent_preparation(args, prep_process, prep_stream, latent_dir)
    _run_jobs(_reward_jobs(args, latent_dir), args.gpus)
    summary = _aggregate(args)
    _write_outputs(args, summary)
    (args.output_root / "EVALUATION_COMPLETE").write_text("complete\n")
    print(json.dumps({"behavior_retained": summary["behavior_retained"], "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
