"""Evaluate goal-reaching joint-position MAE for UFO policies.

For each goal frame in a goal JSON, this script resets the robot to that
motion's first frame, rolls out a fixed goal latent for N steps, and reports
the final joint-position MAE against the target goal frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.goal_inference import _find_goal_json
from humanoidverse.mjlab_inference_utils import add_bool_arg, checkpoint_load_device, load_mjlab_env_cfg
from humanoidverse.tracking_inference import _target_states_from_obs
from humanoidverse.utils.helpers import get_backward_observation


def _goal_name(goal: dict[str, Any], frame_idx: int) -> str:
    return f"{goal.get('motion_name', 'motion')}_{int(frame_idx)}"


def _unique_name(name: str, existing: dict[str, Any]) -> str:
    if name not in existing:
        return name
    suffix = 1
    while f"{name}_{suffix}" in existing:
        suffix += 1
    return f"{name}_{suffix}"


@torch.no_grad()
def _goal_z(model: torch.nn.Module, goal_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    if not hasattr(model, "goal_inference"):
        raise TypeError("Loaded policy does not expose goal_inference(); goal MAE currently requires an FB-style goal encoder.")
    z = model.goal_inference(goal_obs)
    if z.ndim != 2:
        raise ValueError(f"Expected goal latent shape [batch, z_dim], got {tuple(z.shape)}")
    return z


def evaluate_goal_joint_mae(
    *,
    model_folder: Path,
    data_path: Path | None,
    goal_json: Path | None,
    device: str,
    steps: int,
    headless: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    max_episode_length_s: float,
    output: Path | None,
) -> Path:
    if steps <= 0:
        raise ValueError("--steps must be positive")

    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    if data_path is not None:
        data_path = data_path.expanduser().resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file: {data_path}")

    output_path = (
        output.expanduser().resolve()
        if output is not None
        else model_folder / "goal_inference" / f"goal_joint_mae_{int(steps)}steps.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    goal_json_path = _find_goal_json(goal_json)
    with goal_json_path.open("r") as f:
        goals_to_evaluate = json.load(f)
    if not goals_to_evaluate:
        raise RuntimeError(f"Goal JSON is empty: {goal_json_path}")

    model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(device))
    model.to(device)
    model.eval()

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=None,
        device=device,
        headless=headless,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=max_episode_length_s,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    num_dof = int(env.num_dof)

    print(f"[GOAL-MAE] model_folder={model_folder}", flush=True)
    print(f"[GOAL-MAE] data_path={env_cfg.lafan_tail_path}", flush=True)
    print(f"[GOAL-MAE] goal_json={goal_json_path}", flush=True)
    print(f"[GOAL-MAE] steps={steps}, num_dof={num_dof}, device={device}", flush=True)

    per_goal: dict[str, dict[str, Any]] = {}
    final_maes: list[float] = []

    try:
        with torch.no_grad():
            for goal_idx, goal in enumerate(goals_to_evaluate, start=1):
                motion_id = int(goal["motion_id"])
                env.set_is_evaluating(motion_id)
                backward_obs, obs_dict = get_backward_observation(
                    env,
                    0,
                    use_root_height_obs=use_root_height_obs,
                    velocity_multiplier=0,
                )
                num_frames = int(obs_dict["dof_pos"].shape[0])

                for frame_idx_raw in goal["frames"]:
                    frame_idx = int(frame_idx_raw)
                    name = _unique_name(_goal_name(goal, frame_idx), per_goal)
                    if frame_idx >= num_frames:
                        per_goal[name] = {
                            "motion_id": motion_id,
                            "motion_name": goal.get("motion_name"),
                            "frame_idx": frame_idx,
                            "num_frames": num_frames,
                            "skipped": True,
                            "reason": "frame_idx_out_of_range",
                        }
                        print(f"[GOAL-MAE] skip {name}: frame_idx={frame_idx}, num_frames={num_frames}", flush=True)
                        continue

                    goal_observation = {key: value[frame_idx][None, ...] for key, value in backward_obs.items()}
                    goal_observation = tree_map(
                        lambda x: torch.as_tensor(x, device=device, dtype=torch.float32),
                        goal_observation,
                    )
                    z = _goal_z(model, goal_observation)
                    target_dof = obs_dict["dof_pos"][frame_idx].to(device=device, dtype=torch.float32)
                    target_states = _target_states_from_obs(obs_dict, device=device, num_dof=num_dof)
                    observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)

                    reached_steps = 0
                    terminated_early = False
                    truncated_early = False
                    for step in range(int(steps)):
                        action = model.act(observation, z, mean=True)
                        observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                        reached_steps = step + 1
                        terminated_early = bool(torch.as_tensor(terminated).any())
                        truncated_early = bool(torch.as_tensor(truncated).any())
                        if terminated_early or truncated_early:
                            break

                    final_dof = env.simulator.dof_state[0, :, 0].to(device=device, dtype=torch.float32)
                    if final_dof.shape[0] != target_dof.shape[0]:
                        raise ValueError(f"final_dof shape {tuple(final_dof.shape)} != target_dof shape {tuple(target_dof.shape)}")
                    final_mae = float((final_dof - target_dof).abs().mean().detach().cpu().item())
                    final_maes.append(final_mae)

                    per_goal[name] = {
                        "motion_id": motion_id,
                        "motion_name": goal.get("motion_name"),
                        "frame_idx": frame_idx,
                        "num_frames": num_frames,
                        "steps_requested": int(steps),
                        "steps_executed": int(reached_steps),
                        "terminated_early": terminated_early,
                        "truncated_early": truncated_early,
                        "final_joint_mae": final_mae,
                    }
                    print(
                        f"[GOAL-MAE] {goal_idx}/{len(goals_to_evaluate)} {name}: "
                        f"final_joint_mae={final_mae:.6f}, steps={reached_steps}",
                        flush=True,
                    )
    finally:
        wrapped_env.close()

    if not final_maes:
        raise RuntimeError("No goal MAE samples collected.")

    arr = np.asarray(final_maes, dtype=np.float64)
    summary = {
        "policy": model_folder.name,
        "model_folder": str(model_folder),
        "data_path": str(data_path) if data_path is not None else str(env_cfg.lafan_tail_path),
        "goal_json": str(goal_json_path),
        "metric": "goal_final_joint_pos_mae_abs_mean_over_dof",
        "steps": int(steps),
        "num_goals": int(arr.size),
        "global_mean": float(arr.mean()),
        "global_std": float(arr.std()),
        "global_min": float(arr.min()),
        "global_max": float(arr.max()),
        "per_goal": per_goal,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"[GOAL-MAE] saved: {output_path}", flush=True)
    print(
        json.dumps(
            {
                "global_mean": summary["global_mean"],
                "global_std": summary["global_std"],
                "num_goals": summary["num_goals"],
                "steps": summary["steps"],
            }
        ),
        flush=True,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate goal-reaching final joint MAE.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--goal-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    add_bool_arg(parser, "--disable-dr", True, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", True, "Disable observation noise.")
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_goal_joint_mae(
        model_folder=args.model_folder,
        data_path=args.data_path,
        goal_json=args.goal_json,
        device=args.device,
        steps=args.steps,
        headless=args.headless,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        max_episode_length_s=args.max_episode_length_s,
        output=args.output,
    )


if __name__ == "__main__":
    main()
