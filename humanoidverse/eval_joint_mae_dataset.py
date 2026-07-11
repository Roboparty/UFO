"""Evaluate joint-position MAE for a model on a motion dataset.

This script computes per-step MAE on 29-DoF joint positions by comparing
policy rollout joint positions against motion targets generated from the
same motion dataset. It reports global mean/std and per-motion stats.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg
from humanoidverse.tracking_inference import _target_states_from_obs
from humanoidverse.utils.helpers import get_backward_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate joint position MAE/STD on a motion dataset."
    )
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-envs", type=int, default=600)
    parser.add_argument(
        "--agent",
        choices=["auto", "fb", "tldr"],
        default="auto",
        help="auto infers from model methods.",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument(
        "--disable-dr",
        action="store_true",
        default=True,
        help="Disable domain randomization during evaluation.",
    )
    parser.add_argument(
        "--disable-obs-noise",
        action="store_true",
        default=True,
        help="Disable observation noise during evaluation.",
    )
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    parser.add_argument("--log-every-prep", type=int, default=100)
    parser.add_argument("--log-every-step", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <model-folder>/tracking_inference/joint_pos_mae_stats.json",
    )
    return parser.parse_args()




@torch.no_grad()
def _tracking_z(model: torch.nn.Module, obs, agent: str) -> torch.Tensor:
    if agent == "tldr" and hasattr(model, "tracking_inference"):
        return model.tracking_inference(obs)
    z = model.backward_map(obs)
    for step in range(z.shape[0]):
        z[step] = z[step : step + 1].mean(dim=0)
    return model.project_z(z)

def _resolve_agent_flag(model: torch.nn.Module, arg_agent: str) -> str:
    if arg_agent in ("fb", "tldr"):
        return arg_agent
    return "tldr" if hasattr(model, "tracking_inference") else "fb"


def evaluate(args: argparse.Namespace) -> Path:
    model_folder = args.model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint dir: {checkpoint_dir}")

    data_path = args.data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else model_folder / "tracking_inference" / "joint_pos_mae_stats.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: single-env prep for per-motion z/targets.
    prep_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=None,
        device=args.device,
        headless=args.headless,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        max_episode_length_s=args.max_episode_length_s,
    )
    prep_env_wrapped, _ = prep_cfg.build(num_envs=1)
    prep_env = prep_env_wrapped._env

    model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(args.device))
    model.to(args.device)
    model.eval()
    agent_flag = _resolve_agent_flag(model, args.agent)

    num_dof = int(prep_env.num_dof)
    prep_env._motion_lib.load_all_motions()
    num_motions = int(prep_env._motion_lib._num_unique_motions)
    num_chunks = math.ceil(num_motions / args.chunk_envs)
    print(
        f"[EVAL] total_motions={num_motions}, chunk_envs={args.chunk_envs}, "
        f"num_chunks={num_chunks}, agent={agent_flag}"
    )

    all_errs: list[float] = []
    per_motion: dict[str, dict[str, float | int | None]] = {}

    try:
        with torch.no_grad():
            for chunk_idx, start in enumerate(
                range(0, num_motions, args.chunk_envs), start=1
            ):
                end = min(start + args.chunk_envs, num_motions)
                motion_ids = list(range(start, end))

                chunk_zs: list[torch.Tensor] = []
                chunk_target_dofs: list[torch.Tensor] = []
                chunk_root_states: list[torch.Tensor] = []
                chunk_dof_states: list[torch.Tensor] = []
                chunk_lengths: list[int] = []

                for i, motion_id in enumerate(motion_ids, start=1):
                    backward_obs, obs_dict = get_backward_observation(
                        prep_env, motion_id, use_root_height_obs=use_root_height_obs
                    )
                    z = _tracking_z(
                        model,
                        tree_map(
                            lambda x: x[1:].to(args.device)
                            if hasattr(x, "to")
                            else x,
                            backward_obs,
                        ),
                        agent_flag,
                    )
                    target_states = _target_states_from_obs(obs_dict, device=args.device, num_dof=num_dof)

                    chunk_zs.append(z)
                    chunk_target_dofs.append(
                        obs_dict["dof_pos"].to(device=args.device, dtype=torch.float32)
                    )
                    chunk_root_states.append(target_states["root_states"][0])
                    chunk_dof_states.append(target_states["dof_states"][0])
                    chunk_lengths.append(int(z.shape[0]))

                    if i % args.log_every_prep == 0 or i == len(motion_ids):
                        print(
                            f"[EVAL] chunk {chunk_idx}/{num_chunks} prep "
                            f"{i}/{len(motion_ids)}"
                        )

                run_cfg, _ = load_mjlab_env_cfg(
                    model_folder,
                    data_path=data_path,
                    robot_config=None,
                    device=args.device,
                    headless=args.headless,
                    disable_dr=args.disable_dr,
                    disable_obs_noise=args.disable_obs_noise,
                    max_episode_length_s=args.max_episode_length_s,
                )
                run_env_wrapped, _ = run_cfg.build(num_envs=len(motion_ids))
                run_env = run_env_wrapped._env
                try:
                    target_states = {
                        "root_states": torch.stack(chunk_root_states, dim=0),
                        "dof_states": torch.stack(chunk_dof_states, dim=0),
                    }
                    observation, _ = run_env_wrapped.reset(
                        to_numpy=False, target_states=target_states
                    )

                    err_lists = [[] for _ in motion_ids]
                    max_len = max(chunk_lengths)

                    for step in range(max_len):
                        ctx_batch = []
                        active = []
                        for i in range(len(motion_ids)):
                            is_active = step < chunk_lengths[i]
                            active.append(is_active)
                            ctx_batch.append(
                                chunk_zs[i][min(step, chunk_lengths[i] - 1)]
                            )
                        ctx_batch = torch.stack(ctx_batch, dim=0)

                        action = model.act(observation, ctx_batch, mean=True)
                        observation, _reward, _terminated, _truncated, _info = (
                            run_env_wrapped.step(action, to_numpy=False)
                        )

                        pred_all = run_env.simulator.dof_state[..., 0].to(torch.float32)
                        for i in range(len(motion_ids)):
                            if not active[i]:
                                continue
                            tgt = chunk_target_dofs[i][
                                min(step + 1, chunk_target_dofs[i].shape[0] - 1)
                            ]
                            err_lists[i].append((pred_all[i] - tgt).abs().mean().item())

                        if (
                            (step + 1) % args.log_every_step == 0
                            or step == 0
                            or step + 1 == max_len
                        ):
                            done = sum(
                                1
                                for i in range(len(motion_ids))
                                if step + 1 >= chunk_lengths[i]
                            )
                            print(
                                f"[EVAL] chunk {chunk_idx}/{num_chunks} step "
                                f"{step + 1}/{max_len}, finished={done}/{len(motion_ids)}"
                            )
                finally:
                    run_env_wrapped.close()

                for i, motion_id in enumerate(motion_ids):
                    arr = np.asarray(err_lists[i], dtype=np.float64)
                    per_motion[str(motion_id)] = {
                        "mean_mae": float(arr.mean()) if arr.size else None,
                        "std_mae": float(arr.std()) if arr.size else None,
                        "num_steps": int(arr.size),
                    }
                    all_errs.extend(err_lists[i])

                print(f"[EVAL] chunk {chunk_idx}/{num_chunks} done")
    finally:
        prep_env_wrapped.close()

    arr_all = np.asarray(all_errs, dtype=np.float64)
    if arr_all.size == 0:
        raise RuntimeError("No MAE samples collected.")

    summary = {
        "policy": model_folder.name,
        "model_folder": str(model_folder),
        "data_path": str(data_path),
        "metric": "joint_pos_mae_abs_mean_over_29dof",
        "chunk_envs": int(args.chunk_envs),
        "agent_flag": agent_flag,
        "global_mean": float(arr_all.mean()),
        "global_std": float(arr_all.std()),
        "num_total_steps": int(arr_all.size),
        "per_motion": per_motion,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"[EVAL] saved: {output_path}")
    print(
        json.dumps(
            {
                "policy": summary["policy"],
                "global_mean": summary["global_mean"],
                "global_std": summary["global_std"],
                "num_total_steps": summary["num_total_steps"],
                "chunk_envs": summary["chunk_envs"],
            }
        )
    )
    return output_path


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
