#!/usr/bin/env python3
"""Run four useful terrain evaluations while waiting, then resume 8-GPU PBFM."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


REPO = Path("/home/xue/UFO")
PYTHON = REPO / ".venv" / "bin" / "python"
WORK_DIR = Path("/data/xue/UFO/runs/PBFM_fb_terrain_groundrelative_8gpu_20260818_175334")
STATE_DIR = WORK_DIR / "eval_then_resume"
EXPECTED_STEP = 99_295_232
WANDB_RUN_ID = "r0is76mj"
TERRAINS = ("flat", "slope", "stairs", "rough")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def nvidia_rows(query: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [[part.strip() for part in line.split(",")] for line in result.stdout.splitlines() if line.strip()]


def inventory() -> dict[int, str]:
    result = {int(index): gpu_uuid for index, gpu_uuid in nvidia_rows("gpu=index,uuid")}
    if sorted(result) != list(range(8)):
        raise RuntimeError(f"Expected GPUs 0-7, found {sorted(result)}")
    return result


def compute_pids() -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for row in nvidia_rows("compute-apps=gpu_uuid,pid"):
        if len(row) == 2 and row[1].isdigit():
            result.setdefault(row[0], set()).add(int(row[1]))
    return result


def validate_checkpoint() -> None:
    checkpoint = WORK_DIR / "checkpoint"
    status = json.loads((checkpoint / "train_status.json").read_text())
    required = [
        checkpoint / "optimizers.pth",
        checkpoint / "model" / "model.safetensors",
        checkpoint / "buffers",
    ]
    if int(status.get("global_time", -1)) != EXPECTED_STEP or int(status.get("world_size", -1)) != 8:
        raise RuntimeError(f"Unexpected checkpoint status: {status}")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete checkpoint: {missing}")
    log(f"Validated 8-GPU checkpoint at global_time={EXPECTED_STEP}")


def evaluation_command(terrain: str, output: Path) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "humanoidverse.terrain_transfer_inference",
        "--model-folder",
        str(WORK_DIR),
        "--prompt-type",
        "tracking",
        "--data-path",
        str(REPO / "humanoidverse" / "data" / "lafan_29dof.pkl"),
        "--motion-id",
        "7",
        "--terrains",
        terrain,
        "--device",
        "cuda:0",
        "--episode-length",
        "6000",
        "--save-mp4",
        "--render-size",
        "480",
        "--fps",
        "50",
        "--dense-terrain",
        "--output",
        str(output),
    ]


def training_command() -> list[str]:
    return [
        str(PYTHON),
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


def stop_process(process: subprocess.Popen[bytes], label: str) -> None:
    if process.poll() is not None:
        return
    log(f"Stopping owned {label}, PID {process.pid}")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def main() -> None:
    validate_checkpoint()
    gpu_inventory = inventory()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    evaluations: dict[int, tuple[str, subprocess.Popen[bytes]]] = {}
    free_streak = 0
    shutting_down = False

    def cleanup() -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        for _gpu, (terrain, process) in evaluations.items():
            stop_process(process, f"{terrain} evaluation")

    def handle_signal(signum: int, _frame: object) -> None:
        log(f"Received signal {signum}")
        cleanup()
        raise SystemExit(128 + signum)

    for handled in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled, handle_signal)

    log("Waiting for any four genuinely idle GPUs; no holder processes will be used")
    try:
        while not evaluations:
            observed = compute_pids()
            free = [index for index, gpu_uuid in gpu_inventory.items() if not observed.get(gpu_uuid)]
            if len(free) >= 4:
                free_streak += 1
            else:
                free_streak = 0
            log(f"Idle GPUs={free}; four-GPU confirmation={free_streak}/2")
            if free_streak < 2:
                time.sleep(5)
                continue

            selected = free[:4]
            for gpu_index, terrain in zip(selected, TERRAINS, strict=True):
                output = STATE_DIR / f"dance1_subject2_step{EXPECTED_STEP}_120s_dense_{terrain}.json"
                stream = (STATE_DIR / f"{terrain}.log").open("ab", buffering=0)
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu_index),
                        "MUJOCO_GL": "egl",
                        "PYOPENGL_PLATFORM": "egl",
                        "OMP_NUM_THREADS": "2",
                        "PYTHONPATH": str(REPO),
                    }
                )
                process = subprocess.Popen(
                    evaluation_command(terrain, output),
                    cwd=REPO,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                stream.close()
                evaluations[gpu_index] = (terrain, process)
                log(f"Started real {terrain} evaluation on GPU {gpu_index}, PID {process.pid}")

        while True:
            observed = compute_pids()
            owned_pids = {process.pid for _terrain, process in evaluations.values() if process.poll() is None}
            foreign: dict[int, set[int]] = {}
            for index, gpu_uuid in gpu_inventory.items():
                pids = observed.get(gpu_uuid, set()) - owned_pids
                if pids:
                    foreign[index] = pids
            completed = [terrain for terrain, process in evaluations.values() if process.poll() == 0]
            failed = [f"{terrain}:{process.returncode}" for terrain, process in evaluations.values() if process.poll() not in (None, 0)]
            log(f"Foreign busy GPUs={sorted(foreign)}; completed evals={completed}; failed evals={failed}")
            if not foreign:
                break
            time.sleep(5)

        log("All non-evaluation GPU jobs have exited; switching immediately to 8-GPU training")
        cleanup()
        time.sleep(2)
        remaining = compute_pids()
        if any(remaining.get(gpu_uuid) for gpu_uuid in gpu_inventory.values()):
            raise RuntimeError(f"GPU handoff race before training: {remaining}")

        resume_log = STATE_DIR / "resume_from_99295232.log"
        command = training_command()
        with resume_log.open("a") as stream:
            stream.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {shlex.join(command)}\n")
        env = os.environ.copy()
        env["WANDB_RUN_ID"] = WANDB_RUN_ID
        env["WANDB_RESUME"] = "must"
        shell_command = f"exec {shlex.join(command)} >> {shlex.quote(str(resume_log))} 2>&1"
        log(f"Launching 8-GPU training; log={resume_log}")
        os.chdir(REPO)
        os.execvpe("bash", ["bash", "-lc", shell_command], env)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
