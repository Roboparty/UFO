"""Near-riser FB/Aux/D action-sensitivity diagnostic for G1 direct-depth policies.

This script is read-only with respect to a training run.  It loads a frozen
checkpoint, computes one reward latent, runs a deterministic stairs rollout,
selects toe-riser crossing states, and probes small leg-action perturbations:

    action(alpha) = policy_action + alpha * high_knee_direction(foot)

It reports whether Q_FB, Q_aux, and Q_D actually prefer the cleaner
high-clearance direction around states where the policy scrapes stair edges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.direct_depth_actor_diagnostics import (
    MemoryMappedTrajectoryReplay,
    _encode_expert,
    _find_expert_cache,
    _load_expert_buffer,
    _sample_mixed_z,
    _to_device,
)
from humanoidverse.mjlab_inference_utils import (
    checkpoint_load_device,
    load_mjlab_env_cfg,
    resolve_inference_robot_config,
)
from humanoidverse.terrain_transfer import tensor_checksum
from humanoidverse.terrain_transfer_inference import (
    DEFAULT_INFERENCE_DATA_PATH,
    RP1_CENTER_PLATFORM_WIDTH,
    RP1_STAIR_LEVELS,
    RP1_STAIR_STEP_HEIGHT_RANGE,
    RP1_STAIR_STEP_WIDTH,
    _assign_rp1_training_tile,
    _compute_reward_z,
    _default_target_states,
    _is_rp1_training_family,
    _root_ground_clearance,
    _stair_mechanics_frame,
    _stairs_step_height,
    _summarize_stair_mechanics,
    _terrain_env_cfg,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import quat_rotate


JOINT_INDEX = {
    "left_hip_pitch_joint": 0,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "right_hip_pitch_joint": 6,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _checkpoint_step(checkpoint_dir: Path) -> int | None:
    status = checkpoint_dir / "train_status.json"
    if not status.exists():
        return None
    return int(_load_json(status).get("global_time", 0))


def _clone_tree(tree: Any) -> Any:
    return tree_map(lambda value: value.detach().clone() if torch.is_tensor(value) else deepcopy(value), tree)


def _repeat_tree(tree: Any, count: int) -> Any:
    def repeat(value: Any) -> Any:
        if not torch.is_tensor(value):
            return deepcopy(value)
        if value.shape[0] != 1:
            raise ValueError(f"Expected batch-1 tensor, got shape {tuple(value.shape)}")
        return value.expand((count, *value.shape[1:])).clone()

    return tree_map(repeat, tree)


def _pessimistic_value(predictions: torch.Tensor, penalty: float) -> torch.Tensor:
    predictions = predictions.float()
    mean = predictions.mean(dim=0)
    left = predictions.unsqueeze(0)
    right = predictions.unsqueeze(1)
    scale = predictions.shape[0] ** 2 - predictions.shape[0]
    if scale <= 0:
        uncertainty = torch.zeros_like(mean)
    else:
        uncertainty = (left - right).abs().sum(dim=(0, 1)) / scale
    return (mean - float(penalty) * uncertainty).reshape(-1)


def _q_components(
    model: torch.nn.Module,
    observation: Mapping[str, torch.Tensor],
    z: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if actions.ndim != 2:
        raise ValueError(f"Expected actions [B, A], got {tuple(actions.shape)}")
    batch_size = int(actions.shape[0])
    obs_batch = _repeat_tree(observation, batch_size)
    z_batch = z.expand(batch_size, -1).contiguous()
    obs_norm = model._normalize(obs_batch)
    device_type = torch.device(model.device).type
    train_cfg = getattr(getattr(model, "cfg", None), "train", None)
    penalty = 0.0
    # Model configs loaded from checkpoint usually do not include the Agent
    # train block, so read actor pessimism from the checkpoint config instead in
    # the caller and attach it to the model for this diagnostic.
    if hasattr(model, "_diagnostic_actor_pessimism_penalty"):
        penalty = float(model._diagnostic_actor_pessimism_penalty)
    elif train_cfg is not None and hasattr(train_cfg, "actor_pessimism_penalty"):
        penalty = float(train_cfg.actor_pessimism_penalty)
    with torch.autocast(device_type=device_type, dtype=model.amp_dtype, enabled=bool(model.cfg.amp)):
        forward = model._forward_map(obs_norm, z_batch, actions)
        q_fb_all = (forward * z_batch).sum(dim=-1)
        q_aux_all = model._aux_critic(obs_norm, z_batch, actions)
        q_d_all = model._critic(obs_norm, z_batch, actions)
    return {
        "Q_FB": _pessimistic_value(q_fb_all, penalty),
        "Q_Aux": _pessimistic_value(q_aux_all, penalty),
        "Q_D": _pessimistic_value(q_d_all, penalty),
    }


def _action_gradient_components(
    model: torch.nn.Module,
    observation: Mapping[str, torch.Tensor],
    z: torch.Tensor,
    action: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, float]:
    action_var = action.detach().clone().requires_grad_(True)
    q = _q_components(model, observation, z, action_var)
    output: dict[str, float] = {}
    direction = direction.to(action_var.device, dtype=action_var.dtype).reshape_as(action_var)
    direction_norm = torch.linalg.vector_norm(direction).clamp_min(1.0e-9)
    for name, value in q.items():
        gradient = torch.autograd.grad(value.mean(), action_var, retain_graph=True, allow_unused=False)[0]
        output[f"{name}/dq_dalpha_at0"] = float((gradient * direction).sum().item())
        output[f"{name}/grad_norm"] = float(torch.linalg.vector_norm(gradient).item())
        output[f"{name}/cos_with_highknee_direction"] = float(
            ((gradient * direction).sum() / (torch.linalg.vector_norm(gradient).clamp_min(1.0e-9) * direction_norm)).item()
        )
    return output


def _high_knee_direction(foot: str, device: str, action_dim: int = 29) -> torch.Tensor:
    direction = torch.zeros((1, action_dim), device=device, dtype=torch.float32)
    if foot not in {"left", "right", "both"}:
        raise ValueError(f"Unsupported foot {foot!r}")
    sides = ("left", "right") if foot == "both" else (foot,)
    for side in sides:
        prefix = f"{side}_"
        direction[:, JOINT_INDEX[prefix + "hip_pitch_joint"]] = -1.0
        direction[:, JOINT_INDEX[prefix + "knee_joint"]] = 1.0
        direction[:, JOINT_INDEX[prefix + "ankle_pitch_joint"]] = -0.5
    return direction


def _root_state_from_qpos_qvel(qpos: torch.Tensor, qvel: torch.Tensor) -> dict[str, torch.Tensor]:
    qpos = qpos.float()
    qvel = qvel.float()
    quaternion_wxyz = qpos[:, 3:7]
    quaternion_xyzw = quaternion_wxyz[:, (1, 2, 3, 0)]
    angular_body = qvel[:, 3:6]
    angular_world = quat_rotate(quaternion_xyzw, angular_body, w_last=True)
    root = torch.cat((qpos[:, :3], quaternion_xyzw, qvel[:, :3], angular_world), dim=-1)
    dof = torch.stack((qpos[:, 7:], qvel[:, 6:]), dim=-1)
    return {"root_states": root, "dof_states": dof}


def _parse_float_list(text: str) -> list[float]:
    values = [float(item) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Float list cannot be empty")
    return values


def _estimate_actor_reg_weight(
    *,
    model: torch.nn.Module,
    agent_config: Mapping[str, Any],
    buffer_dir: Path,
    buffer_rank: int,
    expert_cache: Path,
    batch_size: int,
    batches: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    train = agent_config["train"]
    reg_coeff = float(train.get("reg_coeff", 0.0))
    reg_coeff_aux = float(train.get("reg_coeff_aux", 0.0))
    if not bool(train.get("scale_reg", True)):
        weight = 1.0
        return {
            "available": True,
            "method": "scale_reg_false_constant",
            "weight": weight,
            "reg_coeff": reg_coeff,
            "reg_coeff_aux": reg_coeff_aux,
            "d_scale": reg_coeff * weight,
            "aux_scale": reg_coeff_aux * weight,
            "batch_size": 0,
            "batches": 0,
            "buffer_rank": int(buffer_rank),
            "expert_cache": str(expert_cache),
        }

    main_replay = MemoryMappedTrajectoryReplay(buffer_dir / f"train_rank_{buffer_rank}")
    expert_buffer = _load_expert_buffer(expert_cache)
    values: list[float] = []
    train_goal_ratio = float(train.get("train_goal_ratio", 0.2))
    expert_asm_ratio = float(train.get("expert_asm_ratio", 0.6))
    relabel_ratio = train.get("relabel_ratio")
    actor_pessimism = float(train.get("actor_pessimism_penalty", 0.0))
    stddev_clip = float(train.get("stddev_clip", 0.3))
    device_type = torch.device(device).type
    try:
        for batch_index in range(int(batches)):
            torch.manual_seed(int(seed) + 91_000 + batch_index)
            main_batch = main_replay.sample(int(batch_size))
            expert_batch = expert_buffer.sample(int(batch_size))
            main_obs = _to_device(main_batch["observation"], device)
            main_next_obs = _to_device(main_batch["next"]["observation"], device)
            expert_next_obs = _to_device(expert_batch["next"]["observation"], device)
            with torch.no_grad():
                main_obs = model._normalize(main_obs)
                main_next_obs = model._normalize(main_next_obs)
                expert_next_obs = model._normalize(expert_next_obs)
                expert_z = _encode_expert(model, expert_next_obs)
                sampled_z = _sample_mixed_z(
                    model,
                    main_next_obs,
                    expert_z,
                    p_goal=train_goal_ratio,
                    p_expert=expert_asm_ratio,
                )
                rollout_z = main_batch["z"].to(device)
                if relabel_ratio is None:
                    z_batch = rollout_z
                else:
                    mask = torch.rand((int(batch_size), 1), device=device) <= float(relabel_ratio)
                    z_batch = torch.where(mask, sampled_z, rollout_z)
                with torch.autocast(
                    device_type=device_type,
                    dtype=model.amp_dtype,
                    enabled=bool(model.cfg.amp),
                ):
                    distribution = model._actor(main_obs, z_batch, model.cfg.actor_std)
                    action = distribution.sample(clip=stddev_clip)
                    forward = model._forward_map(main_obs, z_batch, action)
                    q_fb = _pessimistic_value((forward * z_batch).sum(dim=-1), actor_pessimism)
                values.append(float(q_fb.abs().mean().item()))
    finally:
        del expert_buffer
    weight = statistics.fmean(values) if values else 1.0
    return {
        "available": True,
        "method": "main_replay_actor_objective_estimate",
        "weight": weight,
        "weight_per_batch": values,
        "reg_coeff": reg_coeff,
        "reg_coeff_aux": reg_coeff_aux,
        "d_scale": reg_coeff * weight,
        "aux_scale": reg_coeff_aux * weight,
        "batch_size": int(batch_size),
        "batches": int(batches),
        "buffer_rank": int(buffer_rank),
        "buffer_dir": str(buffer_dir),
        "expert_cache": str(expert_cache),
    }


def _fallback_actor_reg_weight(agent_config: Mapping[str, Any], reason: str) -> dict[str, Any]:
    train = agent_config["train"]
    reg_coeff = float(train.get("reg_coeff", 0.0))
    reg_coeff_aux = float(train.get("reg_coeff_aux", 0.0))
    weight = 1.0
    return {
        "available": False,
        "method": "fallback_unscaled",
        "reason": reason,
        "weight": weight,
        "reg_coeff": reg_coeff,
        "reg_coeff_aux": reg_coeff_aux,
        "d_scale": reg_coeff * weight,
        "aux_scale": reg_coeff_aux * weight,
    }


def _apply_actor_objective_scales(probes: Sequence[dict[str, Any]], scales: Mapping[str, Any]) -> None:
    d_scale = float(scales.get("d_scale", 0.0))
    aux_scale = float(scales.get("aux_scale", 0.0))
    for probe in probes:
        for row in probe.get("q_curve", []):
            q_fb = float(row["Q_FB"])
            q_d = float(row["Q_D"])
            q_aux = float(row["Q_Aux"])
            row["actor_objective"] = q_fb + d_scale * q_d + aux_scale * q_aux
        baseline = next((row for row in probe.get("q_curve", []) if abs(float(row["alpha"])) < 1.0e-9), None)
        if baseline is None:
            continue
        baseline_objective = float(baseline["actor_objective"])
        for row in probe.get("q_curve", []):
            row["delta_objective_FB"] = float(row["delta_Q_FB"])
            row["delta_objective_D_scaled"] = d_scale * float(row["delta_Q_D"])
            row["delta_objective_Aux_scaled"] = aux_scale * float(row["delta_Q_Aux"])
            row["delta_actor_objective"] = float(row["actor_objective"]) - baseline_objective
        gradients = probe.get("q_action_gradients", {})
        if gradients:
            gradients["ActorObjective/dj_dalpha_at0"] = (
                float(gradients["Q_FB/dq_dalpha_at0"])
                + d_scale * float(gradients["Q_D/dq_dalpha_at0"])
                + aux_scale * float(gradients["Q_Aux/dq_dalpha_at0"])
            )


def _radial_distance(xyz: Sequence[float], center_xy: Sequence[float]) -> float:
    return max(abs(float(xyz[0]) - float(center_xy[0])), abs(float(xyz[1]) - float(center_xy[1])))


def _edge_distance(xyz: Sequence[float], center_xy: Sequence[float]) -> float:
    rho = _radial_distance(xyz, center_xy)
    return min(
        abs(rho - (RP1_CENTER_PLATFORM_WIDTH / 2.0 + level * RP1_STAIR_STEP_WIDTH))
        for level in range(RP1_STAIR_LEVELS)
    )


def _build_eval_env(
    *,
    run_dir: Path,
    data_path: Path | None,
    robot_config: Path | None,
    device: str,
    seed: int,
    terrain: str,
    difficulty_row: int,
    episode_length: int,
):
    base_cfg, _use_root_height_obs = load_mjlab_env_cfg(
        run_dir,
        data_path=data_path,
        robot_config=robot_config,
        device=device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=max(30.0, episode_length / 50.0 + 2.0),
    )
    env_cfg = _terrain_env_cfg(base_cfg, terrain, seed)
    env_cfg = env_cfg.model_copy(
        update={
            "auto_reset": False,
            "disable_domain_randomization": True,
            "disable_obs_noise": True,
            "fixed_direct_depth_delay_frames": 0,
        }
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    if _is_rp1_training_family(terrain):
        _assign_rp1_training_tile(wrapped_env._env, family=terrain, difficulty_row=difficulty_row)
    return wrapped_env


def _rollout_and_collect(
    *,
    model: torch.nn.Module,
    env,
    z: torch.Tensor,
    episode_length: int,
) -> dict[str, Any]:
    target_states = _default_target_states(env)
    observation, _ = env.reset(to_numpy=False, target_states=target_states)
    core = env._env
    initial_root = core.robot_root_states[0, :3].clone()
    initial_ground_height = float(initial_root[2].item()) - _root_ground_clearance(env)
    step_height, terrain_difficulty, terrain_level = _stairs_step_height(env, step_height_range=RP1_STAIR_STEP_HEIGHT_RANGE)
    action_clip = float(core.config.robot.control.action_clip_value)
    tile_center_xy = core.env_origins[0, :2].detach().cpu().tolist()

    snapshots: list[dict[str, Any]] = []
    mechanics_frames: list[dict[str, Any]] = []
    terminated_flag = False
    truncated_flag = False
    for step in range(int(episode_length)):
        qpos, qvel = env._get_qpos_qvel(to_numpy=False)
        obs_snapshot = _clone_tree(observation)
        action = model.act(observation, z, mean=True)
        snapshots.append(
            {
                "step_before_action": step,
                "observation": obs_snapshot,
                "qpos": qpos.detach().clone(),
                "qvel": qvel.detach().clone(),
                "policy_action": action.detach().clone(),
            }
        )
        observation, _reward, terminated, truncated, info = env.step(action, to_numpy=False)
        mechanics_frames.append(_stair_mechanics_frame(core, step=step + 1, raw_action=action, info=info))
        terminated_flag = bool(torch.as_tensor(terminated).reshape(-1)[0].item())
        truncated_flag = bool(torch.as_tensor(truncated).reshape(-1)[0].item())
        if terminated_flag or truncated_flag:
            break

    summary = _summarize_stair_mechanics(
        mechanics_frames,
        tile_center_xy=tile_center_xy,
        initial_ground_height=initial_ground_height,
        step_height=step_height,
        action_clip=action_clip,
    )
    final_root = core.robot_root_states[0, :3].detach().clone()
    return {
        "snapshots": snapshots,
        "mechanics_frames": mechanics_frames,
        "mechanics_summary": summary,
        "tile_center_xy": tile_center_xy,
        "initial_ground_height": initial_ground_height,
        "step_height": step_height,
        "terrain_difficulty": terrain_difficulty,
        "terrain_level": terrain_level,
        "action_clip": action_clip,
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "completed_steps": len(mechanics_frames),
        "forward_displacement": float((final_root[0] - initial_root[0]).item()),
        "root_displacement": float(torch.linalg.vector_norm(final_root[:2] - initial_root[:2]).item()),
    }


def _select_events(
    rollout: Mapping[str, Any],
    *,
    max_events: int,
) -> list[dict[str, Any]]:
    snapshots = rollout["snapshots"]
    selected: list[dict[str, Any]] = []
    crossings = list(rollout["mechanics_summary"].get("crossing_events") or [])
    crossings.sort(key=lambda item: (float(item.get("toe_clearance_m", 1.0)), int(item.get("step", 0))))
    used_steps: set[int] = set()
    for event in crossings:
        snapshot_index = int(event["step"]) - 1
        if snapshot_index < 0 or snapshot_index >= len(snapshots) or snapshot_index in used_steps:
            continue
        enriched = dict(event)
        enriched["source"] = "crossing_event"
        enriched["snapshot_index"] = snapshot_index
        selected.append(enriched)
        used_steps.add(snapshot_index)
        if len(selected) >= max_events:
            return selected

    # Fallback: use contact-near-edge candidates if there were too few crossing
    # events.  They do not provide a clearance label, but are still useful for
    # local Q sensitivity around scraping contacts.
    contacts = list(rollout["mechanics_summary"].get("edge_contact_candidates") or [])
    contacts.sort(key=lambda item: -float(item.get("peak_horizontal_force_n", 0.0)))
    for event in contacts:
        snapshot_index = int(event["step"]) - 1
        if snapshot_index < 0 or snapshot_index >= len(snapshots) or snapshot_index in used_steps:
            continue
        enriched = dict(event)
        enriched["source"] = "edge_contact_candidate"
        enriched["snapshot_index"] = snapshot_index
        selected.append(enriched)
        used_steps.add(snapshot_index)
        if len(selected) >= max_events:
            return selected
    return selected


def _q_probe_for_event(
    *,
    model: torch.nn.Module,
    snapshot: Mapping[str, Any],
    event: Mapping[str, Any],
    z: torch.Tensor,
    alphas: Sequence[float],
    action_clip: float,
) -> dict[str, Any]:
    foot = str(event.get("foot", "both"))
    baseline_action = snapshot["policy_action"].to(model.device).float()
    direction = _high_knee_direction(foot, model.device, action_dim=baseline_action.shape[-1])
    alpha_tensor = torch.tensor(alphas, device=model.device, dtype=torch.float32).reshape(-1, 1)
    actions = baseline_action + alpha_tensor * direction
    actions = actions.clamp(min=-float(action_clip), max=float(action_clip))
    observation = tree_map(lambda value: value.to(model.device) if torch.is_tensor(value) else value, snapshot["observation"])
    z = z.to(model.device).float()
    with torch.no_grad():
        q = _q_components(model, observation, z, actions)
    baseline_index = min(range(len(alphas)), key=lambda i: abs(float(alphas[i])))
    rows: list[dict[str, float]] = []
    for i, alpha in enumerate(alphas):
        row = {"alpha": float(alpha)}
        for name, values in q.items():
            value = float(values[i].item())
            row[name] = value
            row[f"delta_{name}"] = value - float(values[baseline_index].item())
        rows.append(row)
    gradients = _action_gradient_components(model, observation, z, baseline_action, direction)
    return {
        "event": {k: v for k, v in event.items() if k not in {"raw_action", "torque_ratio", "aux"}},
        "foot": foot,
        "baseline_action_leg": {
            name: float(baseline_action[0, index].item())
            for name, index in JOINT_INDEX.items()
            if name.startswith(foot) or foot == "both"
        },
        "high_knee_direction": {
            name: float(direction[0, index].item())
            for name, index in JOINT_INDEX.items()
            if name.startswith(foot) or foot == "both"
        },
        "q_curve": rows,
        "q_action_gradients": gradients,
    }


def _short_rollout_from_snapshot(
    *,
    model: torch.nn.Module,
    env,
    snapshot: Mapping[str, Any],
    event: Mapping[str, Any],
    z: torch.Tensor,
    alpha: float,
    intervention_steps: int,
    rollout_steps: int,
    tile_center_xy: Sequence[float],
    initial_ground_height: float,
    step_height: float,
    action_clip: float,
) -> dict[str, Any]:
    foot = str(event.get("foot", "both"))
    direction = _high_knee_direction(foot, model.device, action_dim=snapshot["policy_action"].shape[-1])
    target_states = _root_state_from_qpos_qvel(
        snapshot["qpos"].to(model.device),
        snapshot["qvel"].to(model.device),
    )
    observation, _ = env.reset(to_numpy=False, target_states=target_states)
    core = env._env
    initial_root = core.robot_root_states[0, :3].detach().clone()
    mechanics_frames: list[dict[str, Any]] = []
    terminated_flag = False
    truncated_flag = False
    for step in range(int(rollout_steps)):
        action = model.act(observation, z, mean=True)
        if step < int(intervention_steps):
            action = (action + float(alpha) * direction).clamp(min=-float(action_clip), max=float(action_clip))
        observation, _reward, terminated, truncated, info = env.step(action, to_numpy=False)
        mechanics_frames.append(_stair_mechanics_frame(core, step=step + 1, raw_action=action, info=info))
        terminated_flag = bool(torch.as_tensor(terminated).reshape(-1)[0].item())
        truncated_flag = bool(torch.as_tensor(truncated).reshape(-1)[0].item())
        if terminated_flag or truncated_flag:
            break
    final_root = core.robot_root_states[0, :3].detach().clone()
    summary = _summarize_stair_mechanics(
        mechanics_frames,
        tile_center_xy=tile_center_xy,
        initial_ground_height=initial_ground_height,
        step_height=step_height,
        action_clip=action_clip,
    )
    clearance_values = [float(event["toe_clearance_m"]) for event in summary.get("crossing_events", [])]
    edge_contacts = summary.get("edge_contact_candidates", []) or []
    return {
        "source_step": int(event.get("step", -1)),
        "source_snapshot_index": int(event.get("snapshot_index", -1)),
        "source_event_type": str(event.get("source", "unknown")),
        "source_foot": str(event.get("foot", "both")),
        "source_toe_clearance_m": (
            None if event.get("toe_clearance_m") is None else float(event.get("toe_clearance_m"))
        ),
        "alpha": float(alpha),
        "intervention_steps": int(intervention_steps),
        "rollout_steps": len(mechanics_frames),
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "forward_displacement": float((final_root[0] - initial_root[0]).item()),
        "root_displacement": float(torch.linalg.vector_norm(final_root[:2] - initial_root[:2]).item()),
        "crossing_event_count": int(summary.get("crossing_event_count", 0)),
        "crossing_clearance_mean": statistics.fmean(clearance_values) if clearance_values else None,
        "crossing_clearance_min": min(clearance_values) if clearance_values else None,
        "below_zero_count": int((summary.get("crossing_clearance_m") or {}).get("below_zero_count", 0)),
        "below_3cm_count": int((summary.get("crossing_clearance_m") or {}).get("below_3cm_count", 0)),
        "edge_contact_candidate_count": int(summary.get("edge_contact_candidate_count", len(edge_contacts))),
    }


def _paired_short_rollout_summary(short_rollouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not short_rollouts:
        return {}
    by_event: dict[int, dict[float, Mapping[str, Any]]] = {}
    for row in short_rollouts:
        event_index = int(row.get("event_index", 0))
        alpha = float(row["alpha"])
        by_event.setdefault(event_index, {})[alpha] = row
    alphas = sorted({float(row["alpha"]) for row in short_rollouts if abs(float(row["alpha"])) > 1.0e-9})
    summary: dict[str, Any] = {}
    for alpha in alphas:
        progress_delta: list[float] = []
        root_delta: list[float] = []
        edge_contact_delta: list[float] = []
        clearance_mean_delta: list[float] = []
        clearance_min_delta: list[float] = []
        crossing_count_delta: list[float] = []
        improved_clearance_and_progress_not_worse = 0
        paired = 0
        for event_rows in by_event.values():
            baseline = event_rows.get(0.0)
            changed = event_rows.get(alpha)
            if baseline is None or changed is None:
                continue
            paired += 1
            dp = float(changed["forward_displacement"]) - float(baseline["forward_displacement"])
            dr = float(changed["root_displacement"]) - float(baseline["root_displacement"])
            de = float(changed["edge_contact_candidate_count"]) - float(baseline["edge_contact_candidate_count"])
            dc = float(changed["crossing_event_count"]) - float(baseline["crossing_event_count"])
            progress_delta.append(dp)
            root_delta.append(dr)
            edge_contact_delta.append(de)
            crossing_count_delta.append(dc)
            baseline_mean = baseline.get("crossing_clearance_mean")
            changed_mean = changed.get("crossing_clearance_mean")
            baseline_min = baseline.get("crossing_clearance_min")
            changed_min = changed.get("crossing_clearance_min")
            clearance_improved = False
            if baseline_mean is not None and changed_mean is not None:
                dm = float(changed_mean) - float(baseline_mean)
                clearance_mean_delta.append(dm)
                clearance_improved = dm > 0.0
            elif changed_mean is not None and baseline_mean is None:
                clearance_improved = float(changed_mean) > 0.0
            if baseline_min is not None and changed_min is not None:
                clearance_min_delta.append(float(changed_min) - float(baseline_min))
            if clearance_improved and dp >= -0.02:
                improved_clearance_and_progress_not_worse += 1

        def stats(values: Sequence[float]) -> dict[str, Any] | None:
            if not values:
                return None
            return {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "positive_fraction": sum(value > 0.0 for value in values) / len(values),
                "n": len(values),
            }

        summary[f"{alpha:g}"] = {
            "paired_events": paired,
            "forward_displacement_delta": stats(progress_delta),
            "root_displacement_delta": stats(root_delta),
            "edge_contact_count_delta": stats(edge_contact_delta),
            "crossing_event_count_delta": stats(crossing_count_delta),
            "crossing_clearance_mean_delta": stats(clearance_mean_delta),
            "crossing_clearance_min_delta": stats(clearance_min_delta),
            "clearance_improved_and_progress_not_worse_fraction": (
                improved_clearance_and_progress_not_worse / paired if paired else None
            ),
        }
    return summary


def _summarize_q_probe(probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not probes:
        return {}
    q_names = ("Q_FB", "Q_Aux", "Q_D")
    positive_alphas = sorted(
        {
            float(row["alpha"])
            for probe in probes
            for row in probe["q_curve"]
            if float(row["alpha"]) > 0.0
        }
    )
    summary: dict[str, Any] = {}
    for q_name in q_names:
        per_alpha: dict[str, Any] = {}
        for alpha in positive_alphas:
            deltas = [
                float(row[f"delta_{q_name}"])
                for probe in probes
                for row in probe["q_curve"]
                if abs(float(row["alpha"]) - alpha) < 1.0e-9
            ]
            if deltas:
                per_alpha[f"{alpha:g}"] = {
                    "mean_delta": statistics.fmean(deltas),
                    "min_delta": min(deltas),
                    "max_delta": max(deltas),
                    "positive_fraction": sum(value > 0.0 for value in deltas) / len(deltas),
                    "n": len(deltas),
                }
        gradients = [float(probe["q_action_gradients"][f"{q_name}/dq_dalpha_at0"]) for probe in probes]
        summary[q_name] = {
            "dq_dalpha_at0_mean": statistics.fmean(gradients),
            "dq_dalpha_at0_min": min(gradients),
            "dq_dalpha_at0_max": max(gradients),
            "dq_dalpha_positive_fraction": sum(value > 0.0 for value in gradients) / len(gradients),
            "positive_alpha_delta": per_alpha,
        }
    return summary


def _summarize_actor_objective_probe(probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not probes:
        return {}
    keys = (
        "delta_objective_FB",
        "delta_objective_D_scaled",
        "delta_objective_Aux_scaled",
        "delta_actor_objective",
    )
    positive_alphas = sorted(
        {
            float(row["alpha"])
            for probe in probes
            for row in probe["q_curve"]
            if float(row["alpha"]) > 0.0
        }
    )
    per_alpha: dict[str, Any] = {}
    for alpha in positive_alphas:
        alpha_output: dict[str, Any] = {}
        for key in keys:
            values = [
                float(row[key])
                for probe in probes
                for row in probe["q_curve"]
                if abs(float(row["alpha"]) - alpha) < 1.0e-9 and key in row
            ]
            if not values:
                continue
            alpha_output[key] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "positive_fraction": sum(value > 0.0 for value in values) / len(values),
                "n": len(values),
            }
        per_alpha[f"{alpha:g}"] = alpha_output
    gradients = [
        float(probe["q_action_gradients"]["ActorObjective/dj_dalpha_at0"])
        for probe in probes
        if "ActorObjective/dj_dalpha_at0" in probe.get("q_action_gradients", {})
    ]
    return {
        "positive_alpha_delta": per_alpha,
        "dj_dalpha_at0_mean": statistics.fmean(gradients) if gradients else None,
        "dj_dalpha_at0_min": min(gradients) if gradients else None,
        "dj_dalpha_at0_max": max(gradients) if gradients else None,
        "dj_dalpha_positive_fraction": (sum(value > 0.0 for value in gradients) / len(gradients) if gradients else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-task", default="move-ego-0-0.7")
    parser.add_argument("--terrain", default="low_stairs_down")
    parser.add_argument("--rp1-difficulty-row", type=int, default=4)
    parser.add_argument("--episode-length", type=int, default=750)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_INFERENCE_DATA_PATH)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--buffer-dir", type=Path, default=None)
    parser.add_argument("--buffer-path", type=Path, default=None)
    parser.add_argument("--buffer-rank", type=int, default=0)
    parser.add_argument("--expert-cache", type=Path, default=None)
    parser.add_argument("--expert-cache-root", type=Path, default=Path("/data/xue/UFO/cache/expert_buffers"))
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--process-executor", action="store_true")
    parser.add_argument("--alphas", default="-0.20,-0.10,-0.05,0,0.05,0.10,0.20")
    parser.add_argument("--short-alphas", default="0,0.05,0.10,0.20")
    parser.add_argument("--num-events", type=int, default=6)
    parser.add_argument("--short-event-count", type=int, default=1)
    parser.add_argument("--intervention-steps", type=int, default=8)
    parser.add_argument("--short-rollout-steps", type=int, default=30)
    parser.add_argument("--weight-batch-size", type=int, default=1024)
    parser.add_argument("--weight-batches", type=int, default=3)
    parser.add_argument("--skip-weight-estimate", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir is not None else run_dir / "checkpoint"
    buffer_dir = args.buffer_dir.expanduser().resolve() if args.buffer_dir is not None else checkpoint_dir / "buffers"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    robot_config = resolve_inference_robot_config(args.robot_config, None)
    robot_training = load_robot_training_spec(robot_config)

    model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(args.device))
    model.to(args.device).eval()
    agent_config = _load_json(checkpoint_dir / "config.json")
    model._diagnostic_actor_pessimism_penalty = float(agent_config["train"].get("actor_pessimism_penalty", 0.0))
    if args.expert_cache is not None:
        expert_cache = args.expert_cache.expanduser().resolve()
    else:
        try:
            expert_cache = _find_expert_cache(args.expert_cache_root)
        except Exception:
            expert_cache = None
    reward_args = argparse.Namespace(
        model_folder=run_dir,
        buffer_rank=int(args.buffer_rank),
        buffer_path=args.buffer_path,
        output=output_dir / "reward_latent_probe.json",
        robot_training=robot_training,
        num_samples=int(args.num_samples),
        max_workers=int(args.max_workers),
        process_executor=bool(args.process_executor),
        reward_task=str(args.reward_task),
    )
    z, identifier, _target_states = _compute_reward_z(reward_args, model)
    z = z[:1].to(args.device).float()
    torch.save(
        {
            "z": z.detach().cpu(),
            "prompt_type": "reward",
            "prompt_identifier": identifier,
            "z_checksum": tensor_checksum(z),
        },
        output_dir / "reward_z.pt",
    )

    env = _build_eval_env(
        run_dir=run_dir,
        data_path=args.data_path,
        robot_config=robot_config,
        device=args.device,
        seed=int(args.seed),
        terrain=str(args.terrain),
        difficulty_row=int(args.rp1_difficulty_row),
        episode_length=int(args.episode_length),
    )
    try:
        rollout = _rollout_and_collect(model=model, env=env, z=z, episode_length=int(args.episode_length))
        events = _select_events(rollout, max_events=int(args.num_events))
        probes = [
            _q_probe_for_event(
                model=model,
                snapshot=rollout["snapshots"][int(event["snapshot_index"])],
                event=event,
                z=z,
                alphas=_parse_float_list(args.alphas),
                action_clip=float(rollout["action_clip"]),
            )
            for event in events
        ]
    finally:
        env.close()

    short_rollouts: list[dict[str, Any]] = []
    if events:
        short_env = _build_eval_env(
            run_dir=run_dir,
            data_path=args.data_path,
            robot_config=robot_config,
            device=args.device,
            seed=int(args.seed),
            terrain=str(args.terrain),
            difficulty_row=int(args.rp1_difficulty_row),
            episode_length=max(int(args.short_rollout_steps) + 64, int(args.episode_length)),
        )
        try:
            short_events = events[: max(0, min(int(args.short_event_count), len(events)))]
            for event_index, event in enumerate(short_events):
                snapshot = rollout["snapshots"][int(event["snapshot_index"])]
                for alpha in _parse_float_list(args.short_alphas):
                    result = _short_rollout_from_snapshot(
                        model=model,
                        env=short_env,
                        snapshot=snapshot,
                        event=event,
                        z=z,
                        alpha=alpha,
                        intervention_steps=int(args.intervention_steps),
                        rollout_steps=int(args.short_rollout_steps),
                        tile_center_xy=rollout["tile_center_xy"],
                        initial_ground_height=float(rollout["initial_ground_height"]),
                        step_height=float(rollout["step_height"]),
                        action_clip=float(rollout["action_clip"]),
                    )
                    result["event_index"] = event_index
                    short_rollouts.append(result)
        finally:
            short_env.close()

    # Estimate the training-time regularization scale only after all reward-z,
    # baseline rollout, and local Q probes are fixed.  This keeps the diagnostic
    # extension from changing the random reward latent or selected events.
    if args.skip_weight_estimate:
        actor_objective_scales = _fallback_actor_reg_weight(agent_config, "skipped")
    elif expert_cache is None:
        actor_objective_scales = _fallback_actor_reg_weight(agent_config, f"no expert cache found under {args.expert_cache_root}")
    else:
        try:
            actor_objective_scales = _estimate_actor_reg_weight(
                model=model,
                agent_config=agent_config,
                buffer_dir=buffer_dir,
                buffer_rank=int(args.buffer_rank),
                expert_cache=expert_cache,
                batch_size=int(args.weight_batch_size),
                batches=int(args.weight_batches),
                seed=int(args.seed),
                device=str(args.device),
            )
        except Exception as exc:
            actor_objective_scales = _fallback_actor_reg_weight(agent_config, f"{type(exc).__name__}: {exc}")
    _apply_actor_objective_scales(probes, actor_objective_scales)

    # Avoid dumping the full observation snapshots; keep mechanics because they
    # are compact and useful for auditing selected events.
    baseline_mechanics_path = output_dir / "baseline_rollout_mechanics.json"
    baseline_mechanics_path.write_text(
        json.dumps(
            {
                "terrain": str(args.terrain),
                "tile_center_xy": rollout["tile_center_xy"],
                "initial_ground_height": rollout["initial_ground_height"],
                "summary": rollout["mechanics_summary"],
                "frames": rollout["mechanics_frames"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_global_step": _checkpoint_step(checkpoint_dir),
        "reward_task": str(args.reward_task),
        "reward_z_checksum": tensor_checksum(z),
        "terrain": str(args.terrain),
        "rp1_difficulty_row": int(args.rp1_difficulty_row),
        "deterministic_eval": {
            "disable_dr": True,
            "disable_obs_noise": True,
            "fixed_direct_depth_delay_frames": 0,
        },
        "baseline_rollout": {
            key: rollout[key]
            for key in (
                "completed_steps",
                "terminated",
                "truncated",
                "forward_displacement",
                "root_displacement",
                "step_height",
                "terrain_difficulty",
                "terrain_level",
                "action_clip",
            )
        },
        "baseline_mechanics_summary": rollout["mechanics_summary"],
        "selected_events": events,
        "actor_objective_scales": actor_objective_scales,
        "q_sensitivity_summary": _summarize_q_probe(probes),
        "actor_objective_sensitivity_summary": _summarize_actor_objective_probe(probes),
        "q_sensitivity_events": probes,
        "short_closed_loop": short_rollouts,
        "short_closed_loop_from_worst_event": [row for row in short_rollouts if int(row.get("event_index", -1)) == 0],
        "paired_short_closed_loop_summary": _paired_short_rollout_summary(short_rollouts),
        "artifacts": {
            "reward_z": str(output_dir / "reward_z.pt"),
            "baseline_rollout_mechanics": str(baseline_mechanics_path),
        },
    }
    report_path = output_dir / "stair_fb_action_sensitivity.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    compact = {
        "output": str(output_dir),
        "checkpoint_global_step": report["checkpoint_global_step"],
        "reward_z_checksum": report["reward_z_checksum"],
        "baseline": report["baseline_rollout"],
        "crossing_clearance_m": report["baseline_mechanics_summary"].get("crossing_clearance_m"),
        "actor_objective_scales": report["actor_objective_scales"],
        "q_sensitivity_summary": report["q_sensitivity_summary"],
        "actor_objective_sensitivity_summary": report["actor_objective_sensitivity_summary"],
        "paired_short_closed_loop_summary": report["paired_short_closed_loop_summary"],
        "short_closed_loop_from_worst_event": report["short_closed_loop_from_worst_event"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
