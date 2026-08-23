#!/usr/bin/env python3
"""Restart a checkpointed training command when all requested GPUs are idle."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_status(work_dir: Path) -> dict[str, object] | None:
    path = work_dir / "checkpoint" / "train_status.json"
    if not path.exists():
        return None
    try:
        with path.open() as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        log(f"Checkpoint status is temporarily unreadable: {error}")
        return None


def validate_checkpoint(work_dir: Path, expected_world_size: int) -> dict[str, object]:
    checkpoint = work_dir / "checkpoint"
    status = read_status(work_dir)
    if status is None:
        raise RuntimeError(f"No valid checkpoint status in {checkpoint}")
    required = (
        checkpoint / "model" / "model.safetensors",
        checkpoint / "optimizers.pth",
        checkpoint / "config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Checkpoint is incomplete: {missing}")
    world_size = int(status.get("world_size", -1))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Checkpoint world_size={world_size}, but watcher expects {expected_world_size} GPUs"
        )
    return status


def process_is_running(work_dir: Path) -> bool:
    marker = str(work_dir)
    own_pid = os.getpid()
    for proc_dir in Path("/proc").glob("[0-9]*"):
        if int(proc_dir.name) == own_pid:
            continue
        try:
            argv = (proc_dir / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        decoded = [item.decode(errors="replace") for item in argv if item]
        if marker in decoded and "humanoidverse.train" in decoded:
            return True
    return False


def gpu_rows() -> list[tuple[int, int, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, int, int]] = []
    for line in result.stdout.splitlines():
        if line.strip():
            index, memory_used, utilization = (int(value.strip()) for value in line.split(","))
            rows.append((index, memory_used, utilization))
    return rows


def all_gpus_idle(
    gpu_ids: tuple[int, ...], max_used_mib: int, max_utilization_percent: int
) -> tuple[bool, str]:
    rows = {index: (memory, utilization) for index, memory, utilization in gpu_rows()}
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in rows]
    if missing:
        return False, f"missing GPUs {missing}"
    busy = {
        gpu_id: rows[gpu_id]
        for gpu_id in gpu_ids
        if rows[gpu_id][0] > max_used_mib or rows[gpu_id][1] > max_utilization_percent
    }
    if busy:
        return False, f"busy GPUs (memory MiB, utilization %)={busy}"
    return True, "all requested GPUs idle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--target-global-steps", type=int, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--stable-polls", type=int, default=2)
    parser.add_argument("--max-used-mib", type=int, default=1500)
    parser.add_argument("--max-utilization-percent", type=int, default=5)
    parser.add_argument(
        "--allow-fresh",
        action="store_true",
        help="Allow one initial launch when the work directory has no checkpoint yet.",
    )
    parser.add_argument("--once", action="store_true", help="Report current state and exit without launching.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Training command after --.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.once and not args.command:
        parser.error("a training command is required after --")
    return args


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(","))
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Duplicate GPU IDs: {gpu_ids}")

    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / ".autoresume.lock"
    lock_stream = lock_path.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another watcher already owns {lock_path}") from None
    lock_stream.write(f"pid={os.getpid()}\n")
    lock_stream.flush()

    watcher_log = work_dir / "autoresume.log"
    log(f"Watching {work_dir}; GPUs={gpu_ids}; target={args.target_global_steps}")
    idle_streak = 0
    last_report = ""
    fresh_launch_attempted = False

    while True:
        status = read_status(work_dir)
        global_time = int(status.get("global_time", 0)) if status else 0
        if global_time >= args.target_global_steps:
            log(f"Training is complete at global_time={global_time}; watcher exiting")
            return 0

        if process_is_running(work_dir):
            report = f"training active; checkpoint global_time={global_time}"
            idle_streak = 0
        else:
            idle, detail = all_gpus_idle(
                gpu_ids, args.max_used_mib, args.max_utilization_percent
            )
            idle_streak = idle_streak + 1 if idle else 0
            report = (
                f"training absent; checkpoint global_time={global_time}; {detail}; "
                f"idle confirmation={idle_streak}/{args.stable_polls}"
            )

        if report != last_report:
            log(report)
            last_report = report
        if args.once:
            return 0

        if not process_is_running(work_dir) and idle_streak >= args.stable_polls:
            status = read_status(work_dir)
            if status is None:
                if not args.allow_fresh:
                    report = "resume deferred until checkpoint status exists"
                    if report != last_report:
                        log(report)
                        last_report = report
                    idle_streak = 0
                    time.sleep(args.poll_seconds)
                    continue
                if fresh_launch_attempted:
                    raise RuntimeError("Fresh training exited before writing a valid checkpoint")
                global_time = 0
                launch_label = "fresh training from step 0"
                fresh_launch_attempted = True
            else:
                status = validate_checkpoint(work_dir, len(gpu_ids))
                global_time = int(status["global_time"])
                launch_label = f"checkpoint resume from global_time={global_time}"
            log(f"Launching {launch_label}")
            with watcher_log.open("a", buffering=1) as stream:
                stream.write(
                    f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"LAUNCH {launch_label}\n"
                )
                process = subprocess.Popen(
                    args.command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=os.environ.copy(),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log(f"Started training PID={process.pid}; output={watcher_log}")
                return_code = process.wait()
            log(f"Training PID={process.pid} exited with code {return_code}; monitoring again")
            idle_streak = 0
            last_report = ""

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Watcher interrupted")
        raise SystemExit(130)
    except Exception as error:
        log(f"FATAL: {error}")
        raise
