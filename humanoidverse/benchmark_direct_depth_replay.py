"""Allocate the formal direct-depth replay and complete one optimizer update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from humanoidverse.train import build_ufo_mjlab_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--buffer-size", type=int, default=5_120_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_ufo_mjlab_config(
        device=args.device,
        work_dir=str(args.output_dir),
        num_envs=args.num_envs,
        num_env_steps=args.num_envs * 3,
        seed=args.seed,
        use_wandb=False,
        wandb_run_name=None,
        disable_eval_prioritization=True,
        smoke=False,
        agent="fb_depth",
        terrain_mode="mixed",
        buffer_size=args.buffer_size,
        disable_dr=False,
        disable_obs_noise=False,
        num_agent_updates=1,
    )
    cfg = cfg.model_copy(
        update={
            "agent": cfg.agent.model_copy(update={"compile": False}),
            "num_seed_steps": args.num_envs,
            "update_agent_every": args.num_envs,
            "log_every_updates": args.num_envs,
            "checkpoint_buffer": False,
            "checkpoint_every_steps": 10**12,
            "disable_tqdm": True,
        }
    )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    workspace = cfg.build()
    # Rank 0 keeps this 1024-env flat direct-depth evaluator alive between
    # EMD evaluations, so include it in the formal peak-memory audit.
    workspace._get_priority_eval_env()
    workspace.train()
    torch.cuda.synchronize(device)
    result = {
        "status": "passed",
        "num_envs": args.num_envs,
        "buffer_size": args.buffer_size,
        "trajectory_steps": args.buffer_size // args.num_envs,
        "optimizer_steps": int(workspace._optimizer_steps),
        "peak_allocated_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_memory_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    if result["optimizer_steps"] < 1:
        raise RuntimeError(f"replay smoke did not complete an optimizer update: {result}")
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
