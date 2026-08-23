#!/usr/bin/env python3
"""Snapshot and evaluate fixed stairs milestones without stopping training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


MILESTONES = (
    ("20M", 19_202_048),
    ("40M", 38_404_096),
    ("80M", 76_808_192),
    ("120M", 115_212_288),
    ("192M", 192_020_480),
)
SNAPSHOT_FILES = (
    "config.json",
    "init_kwargs.json",
    "distributed_sync.json",
    "train_status.json",
)
EVALUATOR_SOURCES = (
    "humanoidverse/terrain_transfer_inference.py",
    "humanoidverse/stairs_region_evaluation.py",
    "scripts/watch_stairs_region_milestones.py",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_status(checkpoint: Path) -> dict[str, object] | None:
    path = checkpoint / "train_status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def source_hashes(repo: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in EVALUATOR_SOURCES
    }


def training_is_running(work_dir: Path) -> bool:
    marker = str(work_dir)
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc_dir / "cmdline").read_bytes().decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in command and "humanoidverse.train" in command:
            return True
    return False


def gpu_memory_used_mib(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def copy_reflink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "--archive", "--reflink=auto", str(source), str(destination)],
        check=True,
    )


def create_snapshot(work_dir: Path, output_dir: Path, expected_step: int) -> None:
    checkpoint = work_dir / "checkpoint"
    status_before = read_status(checkpoint)
    if status_before is None or int(status_before.get("global_time", -1)) != expected_step:
        raise RuntimeError(f"Checkpoint changed before snapshot: {status_before}")
    required = (
        work_dir / "config.json",
        checkpoint / "model" / "model.safetensors",
        checkpoint / "model" / "config.json",
        checkpoint / "model" / "init_kwargs.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete evaluation checkpoint: {missing}")

    temporary = output_dir.with_name(f".{output_dir.name}.snapshot-tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    copy_reflink(work_dir / "config.json", temporary / "config.json")
    for name in SNAPSHOT_FILES:
        source = checkpoint / name
        if source.exists():
            copy_reflink(source, temporary / "checkpoint" / name)
    copy_reflink(checkpoint / "model", temporary / "checkpoint" / "model")

    status_after = read_status(checkpoint)
    if status_after != status_before:
        shutil.rmtree(temporary)
        raise RuntimeError(
            f"Checkpoint changed during snapshot: before={status_before}, after={status_after}"
        )
    snapshot_status = read_status(temporary / "checkpoint")
    if snapshot_status != status_before:
        shutil.rmtree(temporary)
        raise RuntimeError("Snapshot train status does not match source")
    temporary.rename(output_dir)


def evaluation_command(args: argparse.Namespace, milestone_dir: Path) -> list[str]:
    return [
        str(args.python),
        "-m",
        "humanoidverse.stairs_region_evaluation",
        "--model-folder",
        str(milestone_dir),
        "--buffer-path",
        str(args.work_dir / "checkpoint" / "buffers" / "train_rank_0"),
        "--output-dir",
        str(milestone_dir),
        "--device",
        "cuda:0",
        "--reward-task",
        args.reward_task,
        "--seeds",
        args.seeds,
        "--episode-length",
        str(args.episode_length),
        "--fall-clearance",
        str(args.fall_clearance),
        "--max-body-impact",
        str(args.max_body_impact),
        "--fps",
        str(args.fps),
    ]


def run_evaluation(args: argparse.Namespace, milestone_dir: Path) -> None:
    log_path = milestone_dir / "evaluation.log"
    tmp_dir = args.repo / "cache" / "milestone_eval_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "MUJOCO_EGL_DEVICE_ID": "0",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "OMP_NUM_THREADS": "2",
            "PYTHONPATH": str(args.repo),
            "TMPDIR": str(tmp_dir),
            "UFO_CACHE_DIR": str(milestone_dir / "cache"),
        }
    )
    (milestone_dir / "cache").mkdir(exist_ok=True)
    command = evaluation_command(args, milestone_dir)
    log(f"Starting stacked evaluation on physical GPU {args.gpu}: {' '.join(command)}")
    with log_path.open("a", buffering=1) as stream:
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            if not training_is_running(args.work_dir):
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                raise RuntimeError("Training process disappeared during stacked evaluation")
            memory_used = gpu_memory_used_mib(args.gpu)
            if memory_used > args.max_used_mib:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                raise RuntimeError(
                    f"GPU {args.gpu} memory use {memory_used} MiB exceeded {args.max_used_mib} MiB"
                )
            time.sleep(10)
    if process.returncode != 0:
        raise RuntimeError(
            f"Milestone evaluation failed with code {process.returncode}; see {log_path}"
        )
    required = (
        milestone_dir / "metrics.csv",
        milestone_dir / "summary.json",
        milestone_dir / "stairs_up.mp4",
        milestone_dir / "stairs_down.mp4",
        milestone_dir / "forward_latent.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Evaluation completed without required outputs: {missing}")
    (milestone_dir / "EVALUATION_COMPLETE").write_text(
        time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/xue/UFO"))
    parser.add_argument("--python", type=Path, default=Path("/home/xue/UFO/.venv/bin/python"))
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--reward-task", default="move-ego-0-0.7")
    parser.add_argument("--seeds", default="4728,4729,4730,4731,4732")
    parser.add_argument("--episode-length", type=int, default=1500)
    parser.add_argument("--fall-clearance", type=float, default=0.45)
    parser.add_argument("--max-body-impact", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-used-mib", type=int, default=130000)
    return parser.parse_args()


def preserve_virtualenv_executable(path: Path) -> Path:
    """Make the path absolute without dereferencing a virtualenv symlink."""
    return Path(os.path.abspath(path.expanduser()))


def main() -> None:
    args = parse_args()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.python = preserve_virtualenv_executable(args.python)
    output_root = args.work_dir / "milestone_evaluations"
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "milestones": [{"label": label, "actual_global_step": step} for label, step in MILESTONES],
        "terrains": ["stairs_up", "stairs_down"],
        "reward_task": args.reward_task,
        "seeds": [int(value) for value in args.seeds.split(",") if value.strip()],
        "episode_length": args.episode_length,
        "fall_clearance": args.fall_clearance,
        "max_body_impact": args.max_body_impact,
        "fps": args.fps,
    }
    protocol_path = output_root / "PROTOCOL.json"
    if protocol_path.exists():
        expected_protocol = json.loads(protocol_path.read_text())
        if protocol != expected_protocol:
            raise RuntimeError(
                f"Evaluation protocol changed after freeze: expected={expected_protocol}, current={protocol}"
            )
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    source_manifest = output_root / "EVALUATOR_SOURCE.json"
    current_hashes = source_hashes(args.repo)
    if source_manifest.exists():
        expected_hashes = json.loads(source_manifest.read_text())["sha256"]
        if current_hashes != expected_hashes:
            raise RuntimeError(
                f"Evaluator source changed after protocol freeze: expected={expected_hashes}, current={current_hashes}"
            )
    else:
        source_manifest.write_text(
            json.dumps(
                {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "sha256": current_hashes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    log(f"Watching {args.work_dir}; stacked evaluation GPU={args.gpu}")

    while True:
        status = read_status(args.work_dir / "checkpoint")
        current_step = int(status.get("global_time", -1)) if status else -1
        pending = False
        for label, expected_step in MILESTONES:
            milestone_dir = output_root / label
            if (milestone_dir / "EVALUATION_COMPLETE").exists():
                continue
            pending = True
            if current_step != expected_step:
                continue
            if not milestone_dir.exists():
                log(f"Snapshotting {label} at exact checkpoint global_time={current_step}")
                create_snapshot(args.work_dir, milestone_dir, expected_step)
            try:
                run_evaluation(args, milestone_dir)
                log(f"Completed milestone {label} at global_time={current_step}")
            except Exception as error:
                (milestone_dir / "EVALUATION_FAILED").write_text(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} {error}\n",
                    encoding="utf-8",
                )
                log(f"FATAL for milestone {label}: {error}")
                raise
        if not pending:
            log("All milestone evaluations completed")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
