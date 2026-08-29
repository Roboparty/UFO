#!/usr/bin/env python3
"""Collect idle GPUs, then resume the eight-GPU PBFM training run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid


REPO = Path("/home/xue/UFO")
WORK_DIR = Path("/data/xue/UFO/runs/PBFM_fb_terrain_groundrelative_8gpu_20260818_175334")
STATE_DIR = WORK_DIR / "gpu_collector"
EXPECTED_GLOBAL_TIME = 99_295_232
GPU_COUNT = 8
WANDB_RUN_ID = "r0is76mj"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run_nvidia_smi(query: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def gpu_inventory() -> dict[int, str]:
    rows = run_nvidia_smi("gpu=index,uuid")
    inventory = {int(index): gpu_uuid for index, gpu_uuid in rows}
    if sorted(inventory) != list(range(GPU_COUNT)):
        raise RuntimeError(f"Expected GPUs 0-{GPU_COUNT - 1}, found {sorted(inventory)}")
    return inventory


def compute_pids_by_uuid() -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for row in run_nvidia_smi("compute-apps=gpu_uuid,pid"):
        if len(row) != 2 or not row[1].isdigit():
            continue
        result.setdefault(row[0], set()).add(int(row[1]))
    return result


def process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, PermissionError):
        return ""


def terminate_owned_holder(pid: int) -> None:
    cmdline = process_cmdline(pid)
    if "pbfm_gpu_collector.py" not in cmdline or "--hold" not in cmdline:
        log(f"Refusing to signal unrecognized PID {pid}: {cmdline!r}")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and process_cmdline(pid):
        time.sleep(0.2)
    if process_cmdline(pid):
        os.kill(pid, signal.SIGKILL)


def clean_stale_holders() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for ready_file in STATE_DIR.glob("holder_gpu*.json"):
        try:
            pid = int(json.loads(ready_file.read_text())["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            ready_file.unlink(missing_ok=True)
            continue
        terminate_owned_holder(pid)
        ready_file.unlink(missing_ok=True)


def validate_checkpoint() -> None:
    checkpoint_dir = WORK_DIR / "checkpoint"
    required = [
        checkpoint_dir / "train_status.json",
        checkpoint_dir / "optimizers.pth",
        checkpoint_dir / "model" / "model.safetensors",
        checkpoint_dir / "buffers",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete checkpoint, missing: {missing}")
    status = json.loads(required[0].read_text())
    if int(status.get("global_time", -1)) != EXPECTED_GLOBAL_TIME:
        raise RuntimeError(
            f"Checkpoint global_time is {status.get('global_time')}, expected {EXPECTED_GLOBAL_TIME}"
        )
    if int(status.get("world_size", -1)) != GPU_COUNT:
        raise RuntimeError(f"Checkpoint world_size is {status.get('world_size')}, expected {GPU_COUNT}")
    log(
        "Validated complete checkpoint: "
        f"global_time={status['global_time']}, local_time={status['local_time']}, "
        f"optimizer_steps={status['optimizer_steps']}"
    )


def holder_main(args: argparse.Namespace) -> int:
    import torch

    ready_file = Path(args.ready_file)
    ready_file.unlink(missing_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Holder must see exactly one CUDA GPU")

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    leave_bytes = args.leave_free_mib * 1024 * 1024
    reserve_bytes = max(0, free_bytes - leave_bytes)
    if reserve_bytes < args.minimum_reserve_mib * 1024 * 1024:
        raise RuntimeError(
            f"GPU {args.physical_gpu} no longer has enough free memory: {free_bytes // 2**20} MiB"
        )

    blocks: list[torch.Tensor] = []
    remaining = reserve_bytes
    chunk_bytes = args.chunk_mib * 1024 * 1024
    while remaining > 0:
        size = min(remaining, chunk_bytes)
        try:
            block = torch.empty(size, dtype=torch.uint8, device="cuda")
            block[0] = 1
            blocks.append(block)
            remaining -= size
        except torch.OutOfMemoryError as error:
            raise RuntimeError(f"Lost allocation race on GPU {args.physical_gpu}") from error
    torch.cuda.synchronize()

    payload = {
        "pid": os.getpid(),
        "physical_gpu": args.physical_gpu,
        "token": args.token,
        "reserved_mib": sum(block.numel() for block in blocks) // 2**20,
        "total_mib": total_bytes // 2**20,
    }
    temporary = ready_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(ready_file)
    log(
        f"Holder ready on physical GPU {args.physical_gpu}: "
        f"reserved {payload['reserved_mib']} MiB, PID {payload['pid']}"
    )

    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGHUP, request_stop)
    while not stop:
        time.sleep(1)
    ready_file.unlink(missing_ok=True)
    return 0


def training_command() -> list[str]:
    return [
        str(REPO / ".venv" / "bin" / "python"),
        "-m",
        "humanoidverse.train",
        "--agent",
        "fb_terrain",
        "--terrain-mode",
        "mixed",
        "--data-path",
        "humanoidverse/data/lafan_29dof_10s-clipped.pkl",
        "--gpu-ids",
        "all",
        "--num-envs",
        "1024",
        "--num-env-steps",
        "192000000",
        "--work-dir",
        str(WORK_DIR),
        "--use-wandb",
        "--wandb-project",
        "PBFM",
        "--wandb-run-name",
        "PBFM_fb_terrain_groundrelative_8gpu_20260818_175334",
    ]


def watcher_main(args: argparse.Namespace) -> int:
    validate_checkpoint()
    clean_stale_holders()
    inventory = gpu_inventory()
    token = uuid.uuid4().hex
    holders: dict[int, subprocess.Popen[bytes]] = {}
    empty_streak = {index: 0 for index in inventory}
    stopping = False

    def cleanup() -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for gpu_index, process in list(holders.items()):
            if process.poll() is None:
                log(f"Releasing owned holder on GPU {gpu_index}, PID {process.pid}")
                terminate_owned_holder(process.pid)
        holders.clear()

    def on_signal(signum: int, _frame: object) -> None:
        log(f"Watcher received signal {signum}; cleaning up owned holders")
        cleanup()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    log(
        f"Watching GPUs 0-{GPU_COUNT - 1} every {args.poll_seconds}s; "
        f"need {args.free_confirmations} consecutive empty observations before claiming"
    )

    try:
        while len(holders) < GPU_COUNT:
            try:
                observed = compute_pids_by_uuid()
            except (subprocess.CalledProcessError, OSError) as error:
                log(f"nvidia-smi query failed ({error}); retrying")
                time.sleep(args.poll_seconds)
                continue

            for gpu_index, process in list(holders.items()):
                ready_file = STATE_DIR / f"holder_gpu{gpu_index}.json"
                if process.poll() is not None or not ready_file.exists():
                    log(f"Holder on GPU {gpu_index} exited; returning GPU to watch pool")
                    holders.pop(gpu_index)
                    ready_file.unlink(missing_ok=True)

            for gpu_index, gpu_uuid in inventory.items():
                if gpu_index in holders:
                    continue
                if observed.get(gpu_uuid):
                    empty_streak[gpu_index] = 0
                    continue
                empty_streak[gpu_index] += 1
                if empty_streak[gpu_index] < args.free_confirmations:
                    continue

                ready_file = STATE_DIR / f"holder_gpu{gpu_index}.json"
                ready_file.unlink(missing_ok=True)
                holder_log = (STATE_DIR / f"holder_gpu{gpu_index}.log").open("ab", buffering=0)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                command = [
                    str(REPO / ".venv" / "bin" / "python"),
                    str(Path(__file__).resolve()),
                    "--hold",
                    "--physical-gpu",
                    str(gpu_index),
                    "--ready-file",
                    str(ready_file),
                    "--token",
                    token,
                    "--leave-free-mib",
                    str(args.leave_free_mib),
                ]
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=holder_log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + args.holder_ready_timeout
                while time.monotonic() < deadline and process.poll() is None and not ready_file.exists():
                    time.sleep(0.2)
                holder_log.close()
                if process.poll() is None and ready_file.exists():
                    holders[gpu_index] = process
                    log(f"Claimed GPU {gpu_index} with owned holder PID {process.pid} ({len(holders)}/{GPU_COUNT})")
                else:
                    log(f"Could not claim GPU {gpu_index}; another process likely won the race")
                    terminate_owned_holder(process.pid)
                    ready_file.unlink(missing_ok=True)
                empty_streak[gpu_index] = 0

            status = ", ".join(
                f"GPU{i}={'ours' if i in holders else 'busy/waiting'}" for i in sorted(inventory)
            )
            log(status)
            if len(holders) < GPU_COUNT:
                time.sleep(args.poll_seconds)

        observed = compute_pids_by_uuid()
        for gpu_index, process in holders.items():
            pids = observed.get(inventory[gpu_index], set())
            foreign = pids - {process.pid}
            if foreign:
                raise RuntimeError(f"GPU {gpu_index} also has foreign compute PIDs {sorted(foreign)}")

        log("All 8 GPUs are held exclusively by this watcher; handing them to PBFM training")
        cleanup()
        time.sleep(2)
        remaining = compute_pids_by_uuid()
        if any(remaining.get(gpu_uuid) for gpu_uuid in inventory.values()):
            raise RuntimeError(f"GPU handoff race: compute processes appeared after releasing holders: {remaining}")

        resume_log = STATE_DIR / "resume_from_99295232.log"
        command = training_command()
        banner = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Resuming checkpoint {EXPECTED_GLOBAL_TIME}\n"
            f"Command: {shlex.join(command)}\n"
        )
        with resume_log.open("a") as stream:
            stream.write(banner)
        log(f"Launching training; output: {resume_log}")
        shell_command = f"exec {shlex.join(command)} >> {shlex.quote(str(resume_log))} 2>&1"
        os.chdir(REPO)
        train_env = os.environ.copy()
        train_env["WANDB_RUN_ID"] = WANDB_RUN_ID
        train_env["WANDB_RESUME"] = "must"
        os.execvpe("bash", ["bash", "-lc", shell_command], train_env)
    finally:
        cleanup()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--ready-file", default="", help=argparse.SUPPRESS)
    parser.add_argument("--token", default="", help=argparse.SUPPRESS)
    parser.add_argument("--leave-free-mib", type=int, default=6144)
    parser.add_argument("--minimum-reserve-mib", type=int, default=4096, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-mib", type=int, default=1024, help=argparse.SUPPRESS)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--free-confirmations", type=int, default=2)
    parser.add_argument("--holder-ready-timeout", type=int, default=30, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.hold:
        if parsed.physical_gpu < 0 or not parsed.ready_file or not parsed.token:
            raise SystemExit("Incomplete holder arguments")
        raise SystemExit(holder_main(parsed))
    raise SystemExit(watcher_main(parsed))
