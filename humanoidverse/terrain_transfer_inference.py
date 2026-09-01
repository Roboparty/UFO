"""Same-z evaluation across the configured physical terrain families."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import mediapy as media
import torch
from torch.utils._pytree import tree_map

from humanoidverse.actor_override import load_actor_module_override, load_actor_override
from humanoidverse.agents.behavior_context import (
    align_heading_sequence,
    heading_observation,
    repeat_heading_sequence,
    root_heading_xy,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.goal_inference import _find_goal_json, load_and_validate_goal_json
from humanoidverse.mjlab_inference_utils import (
    DEFAULT_INFERENCE_DATA_PATH,
    MujocoQposRenderer,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    policy_qpos_from_env,
    replace_hydra_override,
    resolve_inference_robot_config,
)
from humanoidverse.perception.depth_terrain_runtime import TemporalDepthTerrainRuntime
from humanoidverse.terrain_transfer import clone_same_z_for_terrains, tensor_checksum
from humanoidverse.terrains.rp1_simple import (
    RP1_CENTER_PLATFORM_WIDTH,
    RP1_STAIR_LEVELS,
    RP1_STAIR_STEP_HEIGHT_RANGE,
    RP1_STAIR_STEP_WIDTH,
    RP1_TERRAIN_COMPONENT_NAMES,
    rp1_center_reset_profile,
)
from humanoidverse.tracking_inference import _target_states_from_obs, _tracking_z
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import my_quat_rotate

SUPPORTED_TERRAINS = (
    "flat",
    "slope",
    "stairs",
    "stairs_up",
    "stairs_down",
    "rough",
    "platforms",
    "course",
) + RP1_TERRAIN_COMPONENT_NAMES


def _is_rp1_training_family(terrain: str) -> bool:
    return terrain in RP1_TERRAIN_COMPONENT_NAMES


def _assign_rp1_training_tile(core, *, family: str, difficulty_row: int) -> None:
    """Place the inference environment on an exact RP1 training tile."""

    if tuple(core.terrain_component_names) != tuple(RP1_TERRAIN_COMPONENT_NAMES):
        raise RuntimeError(
            f"RP1 inference expected family columns {RP1_TERRAIN_COMPONENT_NAMES}, "
            f"got {core.terrain_component_names}"
        )
    terrain = core.mjlab_env.scene["terrain"]
    num_rows = int(terrain.terrain_origins.shape[0])
    if not 0 <= difficulty_row < num_rows:
        raise ValueError(f"RP1 difficulty row {difficulty_row} is outside [0, {num_rows})")
    family_id = RP1_TERRAIN_COMPONENT_NAMES.index(family)
    terrain.terrain_levels.fill_(int(difficulty_row))
    terrain.terrain_types.fill_(int(family_id))
    core.env_origins.copy_(terrain.terrain_origins[difficulty_row, family_id].expand_as(core.env_origins))
    profile, vertical_direction = rp1_center_reset_profile(family)
    print(
        f"[INFO] RP1 training terrain: asset_family={family}, profile={profile}, "
        f"direction={vertical_direction}, difficulty_row={difficulty_row}, reset=tile_center"
    )


def _stairs_step_center_offset(step: int, *, platform_width: float, step_depth: float) -> float:
    """Return the local radial offset to the center of a one-indexed stair band."""
    if step < 0:
        raise ValueError("stairs start step must be non-negative")
    if step == 0:
        return 0.0
    if platform_width <= 0.0 or step_depth <= 0.0:
        raise ValueError("stairs platform width and step depth must be positive")
    return platform_width / 2.0 + (step - 0.5) * step_depth


def _stairs_down_edge_offset(
    *,
    platform_width: float,
    step_depth: float,
    num_steps: int,
    plateau_width: float,
    edge_margin: float,
) -> float:
    """Return a point on the high plateau facing its first descending step."""
    if min(platform_width, step_depth, plateau_width) <= 0.0 or num_steps <= 0:
        raise ValueError("stairs dimensions and num_steps must be positive")
    if not 0.0 < edge_margin < plateau_width:
        raise ValueError("stairs down-edge margin must lie inside the high plateau")
    down_edge = platform_width / 2.0 + num_steps * step_depth + plateau_width
    return down_edge - edge_margin


def _stairs_pre_ascent_offset(*, platform_width: float, edge_margin: float) -> float:
    """Return a point on the center platform facing the first ascending step."""
    if platform_width <= 0.0 or not 0.0 < edge_margin < platform_width / 2.0:
        raise ValueError("stairs edge margin must lie inside the center platform")
    return platform_width / 2.0 - edge_margin


def _stairs_step_height(
    env,
    *,
    step_height_range: tuple[float, float] | None = None,
) -> tuple[float, float, int]:
    """Return generated step height, normalized difficulty, and terrain level."""
    core = env._env
    terrain = core.mjlab_env.scene["terrain"]
    level = int(terrain.terrain_levels[0].item())
    num_rows = int(terrain.terrain_origins.shape[0])
    difficulty_min, difficulty_max = (
        float(value) for value in core.config.terrain.difficulty_range
    )
    fraction = 0.0 if num_rows <= 1 else level / (num_rows - 1)
    difficulty = difficulty_min + fraction * (difficulty_max - difficulty_min)
    configured_range = (
        core.config.terrain.stairs.step_height_range
        if step_height_range is None
        else step_height_range
    )
    height_min, height_max = (float(value) for value in configured_range)
    return height_min + difficulty * (height_max - height_min), difficulty, level


def _scalar_aux_reward(info: dict[str, Any], name: str) -> float:
    value = info.get("aux_rewards", {}).get(name)
    if value is None:
        return 0.0
    return float(torch.as_tensor(value).reshape(-1)[0].item())


_STAIR_DIAGNOSTIC_JOINTS = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
)


def _stair_mechanics_frame(core, *, step: int, raw_action: torch.Tensor, info: dict[str, Any]) -> dict[str, Any]:
    """Capture inference-only swing, contact, action, and actuator diagnostics."""
    foot_indices = core.feet_indices
    foot_pos = core.body_pos[0, foot_indices]
    foot_rot = core.body_rot[0, foot_indices]
    # The G1 foot collision capsules end at x=0.132, z=-0.025 with radius 0.01 m.
    # Transform the capsule endpoint, then subtract the rotation-invariant sphere radius.
    toe_center_local = torch.tensor([0.132, 0.0, -0.025], device=core.device).expand(2, -1)
    heel_center_local = torch.tensor([-0.054, 0.0, -0.025], device=core.device).expand(2, -1)
    toe = foot_pos + my_quat_rotate(foot_rot, toe_center_local)
    heel = foot_pos + my_quat_rotate(foot_rot, heel_center_local)
    toe = toe.clone()
    heel = heel.clone()
    toe[:, 2] -= 0.01
    heel[:, 2] -= 0.01

    if hasattr(core, "contact_force_history"):
        force_history = core.contact_force_history[0, foot_indices]
        peak_index = torch.linalg.vector_norm(force_history, dim=-1).argmax(dim=-1)
        peak_force = force_history[torch.arange(2, device=core.device), peak_index]
    else:
        peak_force = core.contact_forces[0, foot_indices]
    foot_speed = torch.linalg.vector_norm(core.body_vel[0, foot_indices], dim=-1)

    selected_indices = [core.dof_names.index(name) for name in _STAIR_DIAGNOSTIC_JOINTS]
    clipped = core.actions[0]
    applied = core.applied_actions[0]
    pd_target = (
        core.default_dof_pos[0]
        + core.default_dof_pos_offset[0]
        + applied * core.action_target_scale[0]
    )
    torque_ratio = core.torques[0].abs() / torch.clamp(core.torque_limits, min=1.0e-6)

    def selected(values: torch.Tensor) -> dict[str, float]:
        return {
            name: float(values[index].detach().item())
            for name, index in zip(_STAIR_DIAGNOSTIC_JOINTS, selected_indices)
        }

    aux_names = (
        "penalty_action_rate",
        "limits_dof_pos",
        "penalty_body_impact",
        "penalty_slippage",
        "penalty_ankle_roll",
    )
    return {
        "step": int(step),
        "root_xyz": core.robot_root_states[0, :3].detach().cpu().tolist(),
        "feet": {
            side: {
                "toe_xyz": toe[index].detach().cpu().tolist(),
                "heel_xyz": heel[index].detach().cpu().tolist(),
                "body_xyz": foot_pos[index].detach().cpu().tolist(),
                "peak_contact_force_xyz": peak_force[index].detach().cpu().tolist(),
                "peak_contact_force_norm": float(torch.linalg.vector_norm(peak_force[index]).item()),
                "peak_contact_force_horizontal": float(torch.linalg.vector_norm(peak_force[index, :2]).item()),
                "foot_speed": float(foot_speed[index].item()),
            }
            for index, side in enumerate(("left", "right"))
        },
        "raw_action": selected(raw_action[0]),
        "clipped_action": selected(clipped),
        "applied_action": selected(applied),
        "dof_pos": selected(core.dof_pos[0]),
        "pd_target": selected(pd_target),
        "torque": selected(core.torques[0]),
        "torque_ratio": selected(torque_ratio),
        "raw_action_abs_max_all_joints": float(raw_action[0].abs().max().item()),
        "torque_ratio_max_all_joints": float(torque_ratio.max().item()),
        "aux": {name: _scalar_aux_reward(info, name) for name in aux_names},
    }


def _summarize_stair_mechanics(
    frames: list[dict[str, Any]],
    *,
    tile_center_xy: list[float],
    initial_ground_height: float,
    step_height: float,
    action_clip: float,
) -> dict[str, Any]:
    """Summarize toe clearance at riser crossings and actuator headroom."""
    if not frames:
        raise ValueError("stair mechanics diagnostics require at least one frame")
    center_x, center_y = tile_center_xy
    crossing_events: list[dict[str, Any]] = []
    edge_contact_candidates: list[dict[str, Any]] = []
    previous_edge_candidate = {"left": False, "right": False}

    def radial_distance(xyz: list[float]) -> float:
        return max(abs(float(xyz[0]) - center_x), abs(float(xyz[1]) - center_y))

    for side in ("left", "right"):
        for level in range(1, RP1_STAIR_LEVELS + 1):
            boundary = RP1_CENTER_PLATFORM_WIDTH / 2.0 + (level - 1) * RP1_STAIR_STEP_WIDTH
            for previous, current in zip(frames[:-1], frames[1:]):
                previous_rho = radial_distance(previous["feet"][side]["toe_xyz"])
                current_rho = radial_distance(current["feet"][side]["toe_xyz"])
                if previous_rho < boundary <= current_rho and current_rho > previous_rho:
                    alpha = (boundary - previous_rho) / max(current_rho - previous_rho, 1.0e-9)
                    previous_z = float(previous["feet"][side]["toe_xyz"][2])
                    current_z = float(current["feet"][side]["toe_xyz"][2])
                    crossing_z = previous_z + alpha * (current_z - previous_z)
                    tread_height = initial_ground_height + level * step_height
                    crossing_events.append(
                        {
                            "foot": side,
                            "level": level,
                            "step": int(current["step"]),
                            "toe_clearance_m": crossing_z - tread_height,
                            "toe_world_z_m": crossing_z,
                            "next_tread_world_z_m": tread_height,
                            "pre_crossing_vertical_force_n": abs(
                                float(previous["feet"][side]["peak_contact_force_xyz"][2])
                            ),
                            "peak_contact_force_n": float(current["feet"][side]["peak_contact_force_norm"]),
                            "peak_horizontal_force_n": float(
                                current["feet"][side]["peak_contact_force_horizontal"]
                            ),
                            "foot_speed_mps": float(current["feet"][side]["foot_speed"]),
                            "raw_action": current["raw_action"],
                            "torque_ratio": current["torque_ratio"],
                            "aux": current["aux"],
                        }
                    )
                    break

    for frame in frames:
        for side in ("left", "right"):
            rho = radial_distance(frame["feet"][side]["toe_xyz"])
            edge_distance = min(
                abs(rho - (RP1_CENTER_PLATFORM_WIDTH / 2.0 + level * RP1_STAIR_STEP_WIDTH))
                for level in range(RP1_STAIR_LEVELS)
            )
            foot = frame["feet"][side]
            candidate = (
                edge_distance <= 0.08
                and float(foot["peak_contact_force_horizontal"]) >= 25.0
                and float(foot["foot_speed"]) >= 0.10
            )
            if candidate and not previous_edge_candidate[side]:
                edge_contact_candidates.append(
                    {
                        "foot": side,
                        "step": int(frame["step"]),
                        "edge_distance_m": edge_distance,
                        "peak_contact_force_n": float(foot["peak_contact_force_norm"]),
                        "peak_horizontal_force_n": float(foot["peak_contact_force_horizontal"]),
                        "foot_speed_mps": float(foot["foot_speed"]),
                        "aux": frame["aux"],
                    }
                )
            previous_edge_candidate[side] = candidate

    def distribution(values: list[float]) -> dict[str, float]:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "max": float(tensor.max().item()),
            "p95": float(torch.quantile(tensor, 0.95).item()),
            "mean": float(tensor.mean().item()),
        }

    action_ratios: dict[str, list[float]] = {name: [] for name in _STAIR_DIAGNOSTIC_JOINTS}
    torque_ratios: dict[str, list[float]] = {name: [] for name in _STAIR_DIAGNOSTIC_JOINTS}
    for frame in frames:
        for name in _STAIR_DIAGNOSTIC_JOINTS:
            action_ratios[name].append(abs(float(frame["raw_action"][name])) / action_clip)
            torque_ratios[name].append(abs(float(frame["torque_ratio"][name])))

    clearance_values = [float(event["toe_clearance_m"]) for event in crossing_events]
    return {
        "frames": len(frames),
        "step_height_m": step_height,
        "action_clip": action_clip,
        "crossing_event_count": len(crossing_events),
        "crossing_clearance_m": (
            {
                "min": min(clearance_values),
                "mean": sum(clearance_values) / len(clearance_values),
                "max": max(clearance_values),
                "below_zero_count": sum(value < 0.0 for value in clearance_values),
                "below_3cm_count": sum(value < 0.03 for value in clearance_values),
                "below_5cm_count": sum(value < 0.05 for value in clearance_values),
            }
            if clearance_values
            else None
        ),
        "crossing_events": crossing_events,
        "edge_contact_candidate_count": len(edge_contact_candidates),
        "edge_contact_candidates": edge_contact_candidates,
        "action_clip_ratio": {name: distribution(values) for name, values in action_ratios.items()},
        "action_near_clip_fraction": {
            name: sum(value >= 0.90 for value in values) / len(values)
            for name, values in action_ratios.items()
        },
        "torque_limit_ratio": {name: distribution(values) for name, values in torque_ratios.items()},
        "torque_near_limit_fraction": {
            name: sum(value >= 0.90 for value in values) / len(values)
            for name, values in torque_ratios.items()
        },
    }


def _stairs_progress_metrics(
    *,
    ground_heights: list[float],
    ground_clearances: list[float],
    body_impacts: list[float],
    step_height: float,
    num_steps: int,
    fall_clearance: float,
    min_descent_steps: int,
    max_allowed_body_impact: float,
) -> dict[str, Any]:
    """Measure stair-level progress and reject impact-driven false successes."""
    if not ground_heights or not ground_clearances:
        raise ValueError("stairs progress requires ground heights and clearances")
    if step_height <= 0.0 or num_steps <= 0 or min_descent_steps <= 0:
        raise ValueError("stairs progress dimensions and thresholds must be positive")
    initial_ground_height = ground_heights[0]
    max_ground_height = max(ground_heights)
    min_ground_height = min(ground_heights)
    ascent_steps = min(
        num_steps,
        max(0, int((max_ground_height - initial_ground_height + 0.5 * step_height) // step_height)),
    )
    descent_steps = min(
        num_steps,
        max(0, int((initial_ground_height - min_ground_height + 0.5 * step_height) // step_height)),
    )
    max_body_impact = max(body_impacts, default=0.0)
    normal_final_clearance = ground_clearances[-1] >= fall_clearance
    impact_safe = max_body_impact <= max_allowed_body_impact
    return {
        "initial_ground_height": initial_ground_height,
        "max_ground_height": max_ground_height,
        "min_ground_height": min_ground_height,
        "ascent_initiated": ascent_steps >= 1,
        "ascending_steps_completed": ascent_steps,
        "high_platform_reached": ascent_steps >= num_steps,
        "descent_initiated": descent_steps >= 1,
        "descending_steps_completed": descent_steps,
        "low_platform_reached": descent_steps >= num_steps,
        "mean_body_impact": sum(body_impacts) / max(len(body_impacts), 1),
        "max_body_impact": max_body_impact,
        "impact_safe": impact_safe,
        "min_ground_clearance": min(ground_clearances),
        "normal_final_clearance": normal_final_clearance,
        "ascent_success": ascent_steps >= num_steps and impact_safe and normal_final_clearance,
        "descent_success": (
            descent_steps >= min_descent_steps and impact_safe and normal_final_clearance
        ),
    }


def _max_consecutive_stair_transitions(levels: list[int]) -> int:
    """Count the longest adjacent, ordered stair-level progression."""
    if not levels:
        return 0
    current_run = 0
    longest_run = 0
    previous = levels[0]
    for level in levels[1:]:
        if level == previous:
            continue
        if level == previous + 1:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
        previous = level
    return longest_run


def _separated_stairs_progress_metrics(
    *,
    terrain: str,
    ground_heights: list[float],
    ground_clearances: list[float],
    body_impacts: list[float],
    planar_radii: list[float],
    cumulative_planar_path: float,
    step_height: float,
    num_steps: int,
    center_width: float,
    fall_clearance: float,
    max_allowed_body_impact: float,
) -> dict[str, Any]:
    """Measure center-to-outer traversal on one separated stair family."""
    if terrain not in {"stairs_up", "stairs_down"}:
        raise ValueError(f"Unsupported separated stairs terrain: {terrain!r}")
    if not ground_heights or len(ground_heights) != len(planar_radii):
        raise ValueError("Separated stairs metrics require aligned height/radius trajectories")
    if step_height <= 0.0 or num_steps <= 0 or center_width <= 0.0:
        raise ValueError("Separated stairs dimensions must be positive")

    direction = 1.0 if terrain == "stairs_up" else -1.0
    initial_height = ground_heights[0]
    signed_progress = [direction * (height - initial_height) for height in ground_heights]
    levels = [
        min(num_steps, max(0, int((progress + 0.5 * step_height) // step_height)))
        for progress in signed_progress
    ]
    center_departure_step = next(
        (index for index, radius in enumerate(planar_radii) if radius >= center_width / 2.0),
        None,
    )
    first_transition_step = next(
        (index for index, level in enumerate(levels) if level >= 1),
        None,
    )
    max_body_impact = max(body_impacts, default=0.0)
    normal_final_clearance = ground_clearances[-1] >= fall_clearance
    center_looped = first_transition_step is None and cumulative_planar_path >= 2.0 * center_width
    return {
        "stairs_direction": "up" if direction > 0.0 else "down",
        "initial_ground_height": initial_height,
        "center_departed": center_departure_step is not None,
        "center_departure_step": center_departure_step,
        "first_transition": first_transition_step is not None,
        "first_transition_step": first_transition_step,
        "consecutive_steps_completed": _max_consecutive_stair_transitions(levels),
        "outer_ground_reached": max(levels) >= num_steps,
        "max_stair_level_reached": max(levels),
        "stalled_at_center": first_transition_step is None,
        "center_looped": center_looped,
        "cumulative_planar_path": cumulative_planar_path,
        "mean_body_impact": sum(body_impacts) / max(len(body_impacts), 1),
        "max_body_impact": max_body_impact,
        "impact_safe": max_body_impact <= max_allowed_body_impact,
        "min_ground_clearance": min(ground_clearances),
        "normal_final_clearance": normal_final_clearance,
    }


def _course_completion_radius(course_cfg, *, final_flat_margin: float = 0.30) -> float:
    """Return the radius just inside the final flat annulus."""
    return (
        float(course_cfg.flat_run)
        + 2.0 * int(course_cfg.num_steps) * float(course_cfg.step_depth)
        + float(course_cfg.top_platform_length)
        + float(course_cfg.connector_length)
        + float(course_cfg.ramp_length)
        + final_flat_margin
    )


def _terrain_env_cfg(
    base_cfg,
    terrain: str,
    seed: int,
    *,
    dense_terrain: bool = False,
    patch_size: float | None = None,
):
    overrides = list(base_cfg.hydra_overrides)
    overrides = replace_hydra_override(overrides, "terrain", "terrain_ufo_v0")
    terrain_mode = "rp1_simple" if _is_rp1_training_family(terrain) else terrain
    overrides = replace_hydra_override(overrides, "terrain.terrain_type", terrain_mode)
    overrides = replace_hydra_override(overrides, "terrain.seed", seed)
    if patch_size is not None and not _is_rp1_training_family(terrain):
        overrides = replace_hydra_override(overrides, "terrain.patch_size", [patch_size, patch_size])
    if terrain == "course":
        overrides = replace_hydra_override(overrides, "terrain.num_rows", 1)
    if dense_terrain:
        # Keep the legacy flag as an explicit presentation preset while using
        # the same bounded obstacle semantics as training.
        overrides = replace_hydra_override(overrides, "terrain.stairs.num_steps", 6)
        overrides = replace_hydra_override(overrides, "terrain.stairs.step_depth", 0.30)
        overrides = replace_hydra_override(overrides, "terrain.stairs.platform_width", 1.0)
        overrides = replace_hydra_override(overrides, "terrain.stairs.plateau_width", 0.8)
    return base_cfg.model_copy(update={"hydra_overrides": overrides, "seed": seed})


def _default_target_states(env) -> dict[str, torch.Tensor]:
    core = env._env
    init = core.config.robot.init_state
    root_pos = torch.as_tensor(init.pos, device=core.device, dtype=torch.float32).unsqueeze(0) + core.env_origins
    root_rot = torch.as_tensor(init.rot, device=core.device, dtype=torch.float32).unsqueeze(0)
    root_state = torch.cat((root_pos, root_rot, torch.zeros((1, 6), device=core.device)), dim=-1)
    dof_state = torch.zeros((1, core.num_dof, 2), device=core.device)
    dof_state[..., 0] = core.default_dof_pos
    return {"root_states": root_state, "dof_states": dof_state}


def _expand_goal_sequence(
    goal_latents: torch.Tensor,
    *,
    episode_length: int,
    switch_interval: int,
) -> torch.Tensor:
    if goal_latents.ndim != 2 or goal_latents.shape[0] < 1:
        raise ValueError(f"Expected goal latents [num_goals, z_dim], got {tuple(goal_latents.shape)}")
    if episode_length < 1:
        raise ValueError(f"episode_length must be positive, got {episode_length}")
    if switch_interval < 1:
        raise ValueError(f"switch_interval must be positive, got {switch_interval}")
    indices = torch.div(
        torch.arange(episode_length, device=goal_latents.device),
        switch_interval,
        rounding_mode="floor",
    ).remainder(goal_latents.shape[0])
    return goal_latents.index_select(0, indices)


def _compute_goal_or_tracking_z(args, model, base_cfg):
    encoding_cfg = _terrain_env_cfg(
        base_cfg,
        "flat",
        args.seed,
        dense_terrain=args.dense_terrain,
        patch_size=args.patch_size,
    )
    wrapped_env, _ = encoding_cfg.build(num_envs=1)
    env = wrapped_env._env
    try:
        env._motion_lib.load_all_motions()
        env.is_evaluating = True
        if args.prompt_type == "tracking":
            motion_id = int(args.motion_id)
            backward_obs, obs_dict = get_backward_observation(env, motion_id, use_root_height_obs=args.use_root_height_obs)
            z = _tracking_z(
                model,
                tree_map(lambda x: x[1:].to(args.device) if hasattr(x, "to") else x, backward_obs),
            )
            identifier = f"motion:{motion_id}"
        elif getattr(args, "goal_sequence", False):
            goal_path = _find_goal_json(
                args.goal_json,
                num_dof=env.num_dof,
                robot_name=args.robot_training.robot.name,
            )
            goals = load_and_validate_goal_json(goal_path, num_dof=env.num_dof)
            goal_latents = []
            goal_names = []
            initial_obs_dict = None
            for goal in goals:
                motion_id = int(goal["motion_id"])
                backward_obs, obs_dict = get_backward_observation(
                    env,
                    motion_id,
                    use_root_height_obs=args.use_root_height_obs,
                    velocity_multiplier=0,
                )
                if initial_obs_dict is None:
                    initial_obs_dict = obs_dict
                num_frames = int(next(iter(backward_obs.values())).shape[0])
                for raw_frame_idx in goal["frames"]:
                    frame_idx = int(raw_frame_idx)
                    if frame_idx < 0 or frame_idx >= num_frames:
                        raise ValueError(
                            f"Goal frame {frame_idx} is outside motion {motion_id} with {num_frames} frames. "
                            "Use the full-motion LaFAN data file referenced by the goal JSON."
                        )
                    goal_obs = {
                        key: torch.as_tensor(
                            value[frame_idx : frame_idx + 1],
                            device=args.device,
                            dtype=torch.float32,
                        )
                        for key, value in backward_obs.items()
                    }
                    goal_latents.append(model.goal_inference(goal_obs))
                    goal_names.append(f"{goal.get('motion_name', motion_id)}:{frame_idx}")
            if initial_obs_dict is None or not goal_latents:
                raise RuntimeError(f"No goal frames were loaded from {goal_path}")
            z = _expand_goal_sequence(
                torch.cat(goal_latents, dim=0),
                episode_length=int(args.episode_length),
                switch_interval=int(args.goal_switch_interval),
            )
            identifier = (
                f"goal-sequence:{goal_path.stem}:{len(goal_names)}@{int(args.goal_switch_interval)}"
            )
            obs_dict = initial_obs_dict
            print(f"[INFO] goal sequence: identifier={identifier}, goals={goal_names}")
        else:
            goal_path = _find_goal_json(
                args.goal_json,
                num_dof=env.num_dof,
                robot_name=args.robot_training.robot.name,
            )
            goals = load_and_validate_goal_json(goal_path, num_dof=env.num_dof)
            goal = goals[int(args.goal_index)]
            motion_id = int(goal["motion_id"])
            frame_idx = int(args.goal_frame if args.goal_frame is not None else goal["frames"][0])
            backward_obs, obs_dict = get_backward_observation(
                env,
                motion_id,
                use_root_height_obs=args.use_root_height_obs,
                velocity_multiplier=0,
            )
            num_frames = int(next(iter(backward_obs.values())).shape[0])
            if frame_idx < 0 or frame_idx >= num_frames:
                raise ValueError(
                    f"Goal frame {frame_idx} is outside motion {motion_id} with {num_frames} frames. "
                    "Use the full-motion LaFAN data file referenced by the goal JSON."
                )
            goal_obs = {
                key: torch.as_tensor(value[frame_idx : frame_idx + 1], device=args.device, dtype=torch.float32)
                for key, value in backward_obs.items()
            }
            z = model.goal_inference(goal_obs)
            identifier = f"{goal.get('motion_name', motion_id)}:{frame_idx}"
        target_states = _target_states_from_obs(obs_dict, device=args.device, num_dof=env.num_dof)
        # Store the prompt pose in terrain-local coordinates. MotionLib root Z
        # is already the clearance above the canonical motion plane, so keep it
        # unchanged; each rollout adds its physical terrain origin exactly once.
        # Subtracting the encoding env's Z here makes depressed RP1 tile origins
        # count twice and spawns the robot far above the local ground.
        target_states["root_states"] = target_states["root_states"].clone()
        target_states["root_states"][:, :2] = 0.0
        return z.detach(), identifier, target_states
    finally:
        wrapped_env.close()


def _compute_reward_z(args, model):
    from humanoidverse.mjlab_inference_utils import write_g1_mjlab_relabel_xml
    from humanoidverse.mjlab_reward_relabel import RewardWrapperHV
    from humanoidverse.reward_inference import _load_replay_buffer

    dataset, _ = _load_replay_buffer(args.model_folder, buffer_rank=args.buffer_rank, buffer_path=args.buffer_path)
    output_dir = args.output.parent / "terrain_transfer_relabel"
    relabel_xml = write_g1_mjlab_relabel_xml(Path(args.robot_training.robot.xml_path), output_dir)
    wrapper = RewardWrapperHV(
        model=model,
        inference_dataset=dataset,
        num_samples_per_inference=args.num_samples,
        inference_function="reward_wr_inference",
        max_workers=args.max_workers,
        process_executor=args.process_executor,
        env_model=str(relabel_xml),
    )
    return wrapper.reward_inference(task=args.reward_task).detach(), args.reward_task, None


def _save_prompt_latent(path: Path, z: torch.Tensor, *, prompt_type: str, identifier: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "z": z.detach().cpu(),
        "prompt_type": prompt_type,
        "prompt_identifier": identifier,
        "z_checksum": tensor_checksum(z),
    }
    torch.save(payload, path)
    print(f"[INFO] saved prompt latent {path} z_checksum={payload['z_checksum']}")


def _load_prompt_latent(path: Path, *, prompt_type: str, identifier: str, device: str) -> torch.Tensor:
    path = path.expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "z" not in payload:
        raise ValueError(f"Invalid prompt latent payload: {path}")
    if payload.get("prompt_type") != prompt_type:
        raise ValueError(
            f"Prompt latent type mismatch: expected {prompt_type!r}, got {payload.get('prompt_type')!r}"
        )
    if payload.get("prompt_identifier") != identifier:
        raise ValueError(
            f"Prompt latent identifier mismatch: expected {identifier!r}, got {payload.get('prompt_identifier')!r}"
        )
    z = payload["z"]
    if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[0] < 1 or not torch.isfinite(z).all():
        raise ValueError(f"Prompt latent must be a finite rank-2 tensor, got {type(z)!r} shape={getattr(z, 'shape', None)}")
    checksum = tensor_checksum(z)
    if payload.get("z_checksum") != checksum:
        raise ValueError(
            f"Prompt latent checksum mismatch: stored={payload.get('z_checksum')!r}, computed={checksum!r}"
        )
    print(f"[INFO] loaded prompt latent {path} z_checksum={checksum}")
    return z.to(device)


def _load_reward_latent_for_goal_composition(path: Path, *, device: str) -> tuple[torch.Tensor, str]:
    path = path.expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("prompt_type") != "reward":
        raise ValueError(f"Expected a saved reward latent payload: {path}")
    z = payload.get("z")
    if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[0] != 1 or not torch.isfinite(z).all():
        raise ValueError(
            f"Forward reward latent must be a finite [1, z_dim] tensor, got "
            f"{type(z)!r} shape={getattr(z, 'shape', None)}"
        )
    checksum = tensor_checksum(z)
    if payload.get("z_checksum") != checksum:
        raise ValueError(
            f"Forward reward latent checksum mismatch: stored={payload.get('z_checksum')!r}, "
            f"computed={checksum!r}"
        )
    identifier = str(payload.get("prompt_identifier", path.stem))
    print(f"[INFO] loaded goal forward latent {path} identifier={identifier} z_checksum={checksum}")
    return z.to(device), identifier


def _compose_goal_and_forward_z(
    model,
    goal_z: torch.Tensor,
    forward_z: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    if goal_z.ndim != 2 or forward_z.ndim != 2 or forward_z.shape[0] != 1:
        raise ValueError(
            f"Expected goal_z [T, D] and forward_z [1, D], got "
            f"{tuple(goal_z.shape)} and {tuple(forward_z.shape)}"
        )
    if goal_z.shape[1] != forward_z.shape[1]:
        raise ValueError(f"Latent dimensions differ: {goal_z.shape[1]} != {forward_z.shape[1]}")
    if weight < 0:
        raise ValueError(f"goal forward weight must be non-negative, got {weight}")
    return model.project_z(goal_z + weight * forward_z)


def _prompt_value(model, observation, z: torch.Tensor) -> float:
    discriminator = getattr(model, "_discriminator", None)
    if discriminator is None:
        return float("nan")
    z_step = z if z.shape[0] == 1 else z[:1]
    value = discriminator.compute_reward(model._normalize(observation), z_step)
    return float(value.mean().item())


@torch.no_grad()
def _comparison_action(model, actor: torch.nn.Module, observation, z: torch.Tensor) -> torch.Tensor:
    device_type = torch.device(model.device).type
    with torch.autocast(
        device_type=device_type,
        dtype=model.amp_dtype,
        enabled=bool(model.cfg.amp),
    ):
        distribution = actor(model._normalize(observation), z, model.cfg.actor_std)
    return distribution.mean.float()


def _root_ground_clearance(env) -> float:
    sensor = env._env.mjlab_env.scene.sensors["terrain_height"]
    heights = sensor.data.heights
    if heights.ndim == 3:
        heights = heights[:, 0]
    reference_index = env._env._terrain_reference_index
    if reference_index is None:
        raise RuntimeError("terrain reference ray was not initialized")
    return float(heights[0, reference_index].item())


def _root_upright_score(env) -> float:
    """Return 1 for an upright pelvis and -1 for an inverted pelvis."""
    projected_gravity_z = env._env.projected_gravity[0, 2]
    return float((-projected_gravity_z).clamp(-1.0, 1.0).item())


def _run_rollout(
    args,
    model,
    base_cfg,
    terrain: str,
    z: torch.Tensor,
    target_states,
    comparison_actor: torch.nn.Module | None = None,
) -> dict[str, Any]:
    env_cfg = _terrain_env_cfg(
        base_cfg,
        terrain,
        args.seed,
        dense_terrain=args.dense_terrain,
        patch_size=args.patch_size,
    )
    perception_runtime = (
        TemporalDepthTerrainRuntime(
            env_cfg,
            perception_checkpoint=args.perception_checkpoint,
            device=args.device,
        )
        if args.perception_checkpoint is not None
        else None
    )
    wrapped_env = perception_runtime.wrapped_env if perception_runtime is not None else env_cfg.build(num_envs=1)[0]
    checksum = tensor_checksum(z)
    renderer = None
    try:
        if _is_rp1_training_family(terrain):
            _assign_rp1_training_tile(
                wrapped_env._env,
                family=terrain,
                difficulty_row=int(args.rp1_difficulty_row),
            )
        if target_states is None:
            target_states = _default_target_states(wrapped_env)
        else:
            target_states = {key: value.clone() for key, value in target_states.items()}
            target_states["root_states"][:, :3] += wrapped_env._env.env_origins[:1].to(args.device)
        stairs_reset_region = getattr(args, "stairs_reset_region", None)
        if stairs_reset_region is not None and terrain != "stairs":
            raise ValueError("--stairs-reset-region is only valid with stairs terrain")
        if stairs_reset_region is not None and (
            args.stairs_start_step > 0 or args.stairs_down_edge_margin is not None
        ):
            raise ValueError(
                "--stairs-reset-region cannot be combined with --stairs-start-step or --stairs-down-edge-margin"
            )
        if terrain == "stairs" and stairs_reset_region is not None:
            stairs_cfg = wrapped_env._env.config.terrain.stairs
            edge_margin = float(wrapped_env._env.config.terrain.reset.stairs_edge_margin)
            if stairs_reset_region == "center":
                start_offset = 0.0
            elif stairs_reset_region == "pre_ascent":
                start_offset = _stairs_pre_ascent_offset(
                    platform_width=float(stairs_cfg.platform_width),
                    edge_margin=edge_margin,
                )
            elif stairs_reset_region == "pre_descent":
                start_offset = _stairs_down_edge_offset(
                    platform_width=float(stairs_cfg.platform_width),
                    step_depth=float(stairs_cfg.step_depth),
                    num_steps=int(stairs_cfg.num_steps),
                    plateau_width=float(stairs_cfg.plateau_width),
                    edge_margin=edge_margin,
                )
            else:
                raise ValueError(f"Unsupported stairs reset region: {stairs_reset_region!r}")
            target_states["root_states"][:, 0] += start_offset
            print(
                f"[INFO] stairs reset region={stairs_reset_region} "
                f"local_xy=({start_offset:.3f}, 0.000)"
            )
        elif terrain == "stairs" and args.stairs_down_edge_margin is not None:
            if args.stairs_start_step > 0:
                raise ValueError("--stairs-start-step and --stairs-down-edge-margin are mutually exclusive")
            stairs_cfg = wrapped_env._env.config.terrain.stairs
            start_offset = _stairs_down_edge_offset(
                platform_width=float(stairs_cfg.platform_width),
                step_depth=float(stairs_cfg.step_depth),
                num_steps=int(stairs_cfg.num_steps),
                plateau_width=float(stairs_cfg.plateau_width),
                edge_margin=float(args.stairs_down_edge_margin),
            )
            target_states["root_states"][:, 0] += start_offset
            print(
                f"[INFO] stairs down-edge start: margin={args.stairs_down_edge_margin:.3f}, "
                f"local_xy=({start_offset:.3f}, 0.000)"
            )
        elif terrain in {"stairs", "stairs_up", "stairs_down"} and args.stairs_start_step > 0:
            stairs_cfg = wrapped_env._env.config.terrain.stairs
            if args.stairs_start_step > int(stairs_cfg.num_steps):
                raise ValueError(
                    f"--stairs-start-step={args.stairs_start_step} exceeds "
                    f"num_steps={int(stairs_cfg.num_steps)}"
                )
            start_offset = _stairs_step_center_offset(
                args.stairs_start_step,
                platform_width=float(stairs_cfg.platform_width),
                step_depth=float(stairs_cfg.step_depth),
            )
            target_states["root_states"][:, 0] += start_offset
            print(
                f"[INFO] stairs start: terrain={terrain}, step={args.stairs_start_step}, "
                f"local_xy=({start_offset:.3f}, 0.000)"
            )
        observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)
        if perception_runtime is not None:
            perception_runtime.reset()
        initial_root = wrapped_env._env.robot_root_states[0, :3].clone()
        initial_clearance = _root_ground_clearance(wrapped_env)
        initial_upright_score = _root_upright_score(wrapped_env)
        initial_ground_height = float(initial_root[2].item()) - initial_clearance
        max_forward_displacement = 0.0
        max_planar_displacement = 0.0
        course_completion_radius = (
            _course_completion_radius(wrapped_env._env.config.terrain.course)
            if terrain == "course"
            else None
        )
        course_completed = False
        velocities: list[float] = []
        prompt_values: list[float] = []
        tracking_errors: list[float] = []
        action_l2_deviations: list[float] = []
        action_abs_deviations: list[float] = []
        ground_heights = [initial_ground_height]
        ground_clearances = [initial_clearance]
        body_impacts: list[float] = []
        planar_radii = [0.0]
        cumulative_planar_path = 0.0
        previous_root_xy = initial_root[:2].clone()
        frames = []
        mechanics_frames: list[dict[str, Any]] = []
        mechanics_output = getattr(args, "stair_mechanics_output", None)
        collect_stair_mechanics = mechanics_output is not None and terrain == "low_stairs_down"
        if args.save_mp4:
            renderer = MujocoQposRenderer(
                None,
                render_size=args.render_size,
                scene_spec=wrapped_env._env.mjlab_env.scene.spec,
                source_xml_path=Path(args.robot_training.robot.xml_path),
                add_floor=False,
                camera_distance=args.camera_distance,
                camera_azimuth=args.camera_azimuth,
                camera_elevation=args.camera_elevation,
                expected_qpos_size=7 + wrapped_env._env.num_dof,
            )
        terminated_flag = False
        truncated_flag = False
        boundary_reset_flag = False
        steps = min(args.episode_length, int(z.shape[0])) if args.prompt_type == "tracking" else args.episode_length
        heading_targets = None
        heading_valid = False
        if bool(getattr(model.cfg, "heading_context_enabled", False)):
            current_heading = root_heading_xy(wrapped_env._env.base_quat.float())
            if args.prompt_type == "reward":
                # An explicit ego-forward reward command owns a fixed world
                # heading captured when the command starts.
                heading_targets = current_heading.clone()
                heading_valid = True
            elif args.prompt_type == "tracking":
                reference_heading = root_heading_xy(
                    target_states["root_states"][:, 3:7].to(args.device).float()
                )
                repeats = max(1, (steps + reference_heading.shape[0] - 1) // reference_heading.shape[0])
                reference_heading = repeat_heading_sequence(reference_heading, repeats)
                heading_targets = align_heading_sequence(
                    reference_heading.unsqueeze(0),
                    current_heading,
                    torch.zeros(1, device=args.device, dtype=torch.long),
                )[0]
                heading_valid = True
        completed = 0
        last_z_step = z[:1]
        for step in range(steps):
            z_step = z[step : step + 1] if z.shape[0] > 1 else z
            if perception_runtime is not None:
                observation["terrain_actor"] = perception_runtime.terrain_actor(
                    observation,
                    reset_mask=torch.ones(1, device=args.device, dtype=torch.bool) if step == 0 else None,
                )
            if heading_valid:
                target_heading = (
                    heading_targets
                    if heading_targets.ndim == 2 and heading_targets.shape[0] == 1
                    else heading_targets[min(step, heading_targets.shape[0] - 1)].unsqueeze(0)
                )
                observation["heading"] = heading_observation(
                    root_heading_xy(wrapped_env._env.base_quat.float()),
                    target_heading,
                    torch.ones((1, 1), device=args.device, dtype=torch.bool),
                )
            last_z_step = z_step
            action = model.act(observation, z_step, mean=True)
            if comparison_actor is not None:
                comparison_action = _comparison_action(model, comparison_actor, observation, z_step)
                difference = action - comparison_action
                action_l2_deviations.append(float(torch.linalg.vector_norm(difference, dim=-1).mean().item()))
                action_abs_deviations.append(float(difference.abs().mean().item()))
            observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
            if collect_stair_mechanics:
                mechanics_frames.append(
                    _stair_mechanics_frame(
                        wrapped_env._env,
                        step=step + 1,
                        raw_action=action,
                        info=_info,
                    )
                )
            if perception_runtime is not None:
                reset = torch.as_tensor(terminated, device=args.device).bool() | torch.as_tensor(truncated, device=args.device).bool()
                perception_runtime.after_step(reset)
            completed = step + 1
            forward_displacement = float(
                (wrapped_env._env.robot_root_states[0, 0] - initial_root[0]).item()
            )
            max_forward_displacement = max(max_forward_displacement, forward_displacement)
            planar_displacement = float(
                torch.linalg.vector_norm(wrapped_env._env.robot_root_states[0, :2] - initial_root[:2]).item()
            )
            max_planar_displacement = max(max_planar_displacement, planar_displacement)
            current_root_xy = wrapped_env._env.robot_root_states[0, :2].clone()
            cumulative_planar_path += float(torch.linalg.vector_norm(current_root_xy - previous_root_xy).item())
            previous_root_xy = current_root_xy
            planar_radii.append(planar_displacement)
            course_completed = bool(
                course_completion_radius is not None and planar_displacement >= course_completion_radius
            )
            velocities.append(float(torch.linalg.vector_norm(wrapped_env._env.robot_root_states[0, 7:9]).item()))
            prompt_values.append(_prompt_value(model, observation, z_step))
            clearance = _root_ground_clearance(wrapped_env)
            ground_clearances.append(clearance)
            ground_heights.append(
                float(wrapped_env._env.robot_root_states[0, 2].item()) - clearance
            )
            body_impacts.append(_scalar_aux_reward(_info, "penalty_body_impact"))
            if args.prompt_type == "tracking":
                tracking_errors.append(float(torch.linalg.vector_norm(wrapped_env._env.dif_global_body_pos, dim=-1).mean().item()))
            if renderer is not None:
                render_qpos = policy_qpos_from_env(wrapped_env, expected_qpos_size=renderer.input_nq)
                frames.append(renderer.render_qpos(render_qpos))
                if step == 0:
                    print(f"[INFO] terrain renderer state: {renderer.render_debug_state()}")
            terminated_flag = bool(torch.as_tensor(terminated).any().item())
            truncated_flag = bool(torch.as_tensor(truncated).any().item())
            boundary_reset_flag = bool(torch.as_tensor(_info.get("boundary_resets", False)).any().item())
            if course_completed or terminated_flag or truncated_flag:
                break
        final_root = wrapped_env._env.robot_root_states[0, :3].clone()
        final_ground_clearance = _root_ground_clearance(wrapped_env)
        final_upright_score = _root_upright_score(wrapped_env)
        stairs_metrics: dict[str, Any] = {}
        stair_step_height_for_diagnostics: float | None = None
        if terrain == "stairs":
            step_height, terrain_difficulty, terrain_level = _stairs_step_height(wrapped_env)
            num_steps = int(wrapped_env._env.config.terrain.stairs.num_steps)
            stairs_metrics = {
                "stairs_reset_region": stairs_reset_region,
                "terrain_level": terrain_level,
                "terrain_difficulty": terrain_difficulty,
                "stairs_step_height": step_height,
                **_stairs_progress_metrics(
                    ground_heights=ground_heights,
                    ground_clearances=ground_clearances,
                    body_impacts=body_impacts,
                    step_height=step_height,
                    num_steps=num_steps,
                    fall_clearance=float(args.fall_clearance),
                    min_descent_steps=int(args.min_descent_steps),
                    max_allowed_body_impact=float(args.max_body_impact),
                ),
            }
        elif terrain in {"stairs_up", "stairs_down"}:
            step_height, terrain_difficulty, terrain_level = _stairs_step_height(wrapped_env)
            stairs_cfg = wrapped_env._env.config.terrain.stairs
            stairs_metrics = {
                "terrain_level": terrain_level,
                "terrain_difficulty": terrain_difficulty,
                "stairs_step_height": step_height,
                **_separated_stairs_progress_metrics(
                    terrain=terrain,
                    ground_heights=ground_heights,
                    ground_clearances=ground_clearances,
                    body_impacts=body_impacts,
                    planar_radii=planar_radii,
                    cumulative_planar_path=cumulative_planar_path,
                    step_height=step_height,
                    num_steps=int(stairs_cfg.num_steps),
                    center_width=float(stairs_cfg.platform_width),
                    fall_clearance=float(args.fall_clearance),
                    max_allowed_body_impact=float(args.max_body_impact),
                ),
            }
        elif terrain in {"low_stairs_up", "low_stairs_down"}:
            step_height, terrain_difficulty, terrain_level = _stairs_step_height(
                wrapped_env,
                step_height_range=RP1_STAIR_STEP_HEIGHT_RANGE,
            )
            stair_step_height_for_diagnostics = step_height
            profile, vertical_direction = rp1_center_reset_profile(terrain)
            separated_direction = "stairs_up" if vertical_direction == "ascent" else "stairs_down"
            stairs_metrics = {
                "terrain_asset_family": terrain,
                "terrain_family": profile,
                "initial_vertical_direction": vertical_direction,
                "reset_region": "tile_center",
                "terrain_level": terrain_level,
                "terrain_difficulty": terrain_difficulty,
                "stairs_step_height": step_height,
                **_separated_stairs_progress_metrics(
                    terrain=separated_direction,
                    ground_heights=ground_heights,
                    ground_clearances=ground_clearances,
                    body_impacts=body_impacts,
                    planar_radii=planar_radii,
                    cumulative_planar_path=cumulative_planar_path,
                    step_height=step_height,
                    num_steps=RP1_STAIR_LEVELS,
                    center_width=RP1_CENTER_PLATFORM_WIDTH,
                    fall_clearance=float(args.fall_clearance),
                    max_allowed_body_impact=float(args.max_body_impact),
                ),
            }
        final_goal_error = None
        if args.prompt_type == "goal":
            achieved_z = model.goal_inference(observation)
            final_goal_error = float(
                torch.linalg.vector_norm(achieved_z - last_z_step, dim=-1).mean().item()
            )
        fell = terminated_flag or final_ground_clearance < args.fall_clearance
        rollout_completed = completed == steps and not terminated_flag and not truncated_flag
        video_path = None
        if renderer is not None:
            video_path = args.output.with_suffix("").with_name(f"{args.output.stem}_{terrain}.mp4")
            if not frames:
                raise RuntimeError(f"No frames rendered for terrain={terrain}")
            media.write_video(str(video_path), frames, fps=args.fps)
            print(f"[INFO] wrote terrain video {video_path}")
        stair_mechanics_summary = None
        if collect_stair_mechanics:
            if stair_step_height_for_diagnostics is None:
                raise RuntimeError("stair mechanics diagnostics did not resolve the RP1 step height")
            stair_mechanics_summary = _summarize_stair_mechanics(
                mechanics_frames,
                tile_center_xy=wrapped_env._env.env_origins[0, :2].detach().cpu().tolist(),
                initial_ground_height=initial_ground_height,
                step_height=stair_step_height_for_diagnostics,
                action_clip=float(wrapped_env._env.config.robot.control.action_clip_value),
            )
            mechanics_path = mechanics_output.expanduser().resolve()
            mechanics_path.parent.mkdir(parents=True, exist_ok=True)
            mechanics_payload = {
                "terrain": terrain,
                "prompt_identifier": args.prompt_identifier,
                "checkpoint": str(args.model_folder),
                "tile_center_xy": wrapped_env._env.env_origins[0, :2].detach().cpu().tolist(),
                "initial_ground_height": initial_ground_height,
                "summary": stair_mechanics_summary,
                "frames": mechanics_frames,
            }
            mechanics_path.write_text(json.dumps(mechanics_payload, indent=2) + "\n")
            print(f"[INFO] wrote stair mechanics diagnostics {mechanics_path}")
        result = {
            "terrain_type": terrain,
            "seed": args.seed,
            "prompt_type": args.prompt_type,
            "prompt_identifier": args.prompt_identifier,
            "z_shape": list(z.shape),
            "z_checksum": checksum,
            "episode_length": completed,
            "requested_episode_length": steps,
            "rollout_completed": rollout_completed,
            "terminated": terminated_flag,
            "truncated": truncated_flag,
            "boundary_reset": boundary_reset_flag,
            "fell": fell,
            "root_displacement": float(torch.linalg.vector_norm(final_root[:2] - initial_root[:2]).item()),
            "forward_displacement": float((final_root[0] - initial_root[0]).item()),
            "max_forward_displacement": max_forward_displacement,
            "max_planar_displacement": max_planar_displacement,
            "course_completion_radius": course_completion_radius,
            "course_completed": course_completed if terrain == "course" else None,
            "mean_root_velocity": sum(velocities) / max(len(velocities), 1),
            "final_root_height": float(final_root[2].item()),
            "initial_ground_clearance": initial_clearance,
            "final_ground_clearance": final_ground_clearance,
            "mean_ground_clearance": sum(ground_clearances) / len(ground_clearances),
            "min_ground_clearance": min(ground_clearances),
            "initial_upright_score": initial_upright_score,
            "final_upright_score": final_upright_score,
            "mean_body_impact": sum(body_impacts) / max(len(body_impacts), 1),
            "max_body_impact": max(body_impacts, default=0.0),
            "mean_prompt_value": sum(prompt_values) / max(len(prompt_values), 1),
            "final_goal_error": final_goal_error,
            "mean_tracking_error": sum(tracking_errors) / len(tracking_errors) if tracking_errors else None,
            "mean_action_l2_deviation": (
                sum(action_l2_deviations) / len(action_l2_deviations) if action_l2_deviations else None
            ),
            "max_action_l2_deviation": max(action_l2_deviations) if action_l2_deviations else None,
            "mean_action_abs_deviation": (
                sum(action_abs_deviations) / len(action_abs_deviations) if action_abs_deviations else None
            ),
            "actor_checksum": (
                args.actor_override_info["checksum"] if args.actor_override_info is not None else None
            ),
            "comparison_actor_checksum": (
                args.comparison_actor_info["checksum"] if args.comparison_actor_info is not None else None
            ),
            "video_path": str(video_path) if video_path is not None else None,
            "stair_mechanics_summary": stair_mechanics_summary,
            **stairs_metrics,
        }
        return result
    finally:
        if renderer is not None:
            renderer.close()
        if perception_runtime is not None:
            perception_runtime.close()
        else:
            wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reuse one UFO z exactly across physical terrains.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--prompt-type", choices=["reward", "goal", "tracking"], required=True)
    parser.add_argument("--terrains", default="flat,slope,stairs,rough")
    parser.add_argument(
        "--rp1-difficulty-row",
        type=int,
        default=4,
        help="Curriculum row [0, 9] used by exact RP1 training-family rollouts.",
    )
    parser.add_argument(
        "--patch-size",
        type=float,
        default=None,
        help="Optional square terrain patch size in meters for long inference rollouts.",
    )
    parser.add_argument(
        "--dense-terrain",
        action="store_true",
        help="Evaluation-only preset using the bounded six-step stair presentation.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_INFERENCE_DATA_PATH)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--fall-clearance", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=Path("terrain_transfer_results.json"))
    parser.add_argument("--reward-task", default="move-ego-0-0.7")
    parser.add_argument("--save-latent", type=Path, default=None)
    parser.add_argument("--load-latent", type=Path, default=None)
    parser.add_argument(
        "--actor-override",
        type=Path,
        default=None,
        help="Actor-only milestone loaded in memory; the source full checkpoint is never modified.",
    )
    parser.add_argument(
        "--comparison-actor",
        type=Path,
        default=None,
        help="Optional read-only Actor used only to measure action deviation on rollout states.",
    )
    parser.add_argument(
        "--perception-checkpoint",
        type=Path,
        default=None,
        help="Optional temporal terrain checkpoint; clean depth replaces GT terrain_actor.",
    )
    parser.add_argument("--buffer-path", type=Path, default=None)
    parser.add_argument("--buffer-rank", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--process-executor", action="store_true")
    parser.add_argument("--goal-json", type=Path, default=None)
    parser.add_argument("--goal-index", type=int, default=0)
    parser.add_argument("--goal-frame", type=int, default=None)
    parser.add_argument("--goal-sequence", action="store_true")
    parser.add_argument("--goal-switch-interval", type=int, default=100)
    parser.add_argument(
        "--goal-forward-latent",
        type=Path,
        default=None,
        help="Saved reward latent to blend into every goal latent before projection.",
    )
    parser.add_argument("--goal-forward-weight", type=float, default=0.5)
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument(
        "--tracking-repeats",
        type=int,
        default=1,
        help="Repeat a tracking latent sequence for a longer continuous rollout.",
    )
    parser.add_argument(
        "--stairs-start-step",
        type=int,
        default=0,
        help=(
            "Start a stairs/stairs_up/stairs_down rollout at the center of this one-indexed "
            "stair band; 0 keeps the center platform."
        ),
    )
    parser.add_argument(
        "--stairs-down-edge-margin",
        type=float,
        default=None,
        help="Start this many meters inside the high plateau's first descending edge, facing outward.",
    )
    parser.add_argument(
        "--stairs-reset-region",
        choices=("center", "pre_ascent", "pre_descent"),
        default=None,
        help="Start a stairs rollout from the matching training reset region.",
    )
    parser.add_argument(
        "--min-descent-steps",
        type=int,
        default=3,
        help="Minimum completed descending levels required by strict descent success.",
    )
    parser.add_argument(
        "--max-body-impact",
        type=float,
        default=1.0,
        help="Maximum continuous body-impact severity allowed by strict success metrics.",
    )
    parser.add_argument(
        "--stair-mechanics-output",
        type=Path,
        default=None,
        help=(
            "Optional inference-only JSON trace for low_stairs_down toe-clearance, "
            "contact, action, and torque diagnostics."
        ),
    )
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_folder = args.model_folder.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.robot_config = resolve_inference_robot_config(args.robot_config, None)
    args.robot_training = load_robot_training_spec(args.robot_config)
    checkpoint_dir = args.model_folder / "checkpoint"
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(args.device))
    model.to(args.device).eval()
    args.actor_override_info = load_actor_override(model, args.actor_override) if args.actor_override is not None else None
    if args.actor_override_info is not None:
        print(f"[INFO] Loaded read-only Actor override: {args.actor_override_info}")
    comparison_actor = None
    args.comparison_actor_info = None
    if args.comparison_actor is not None:
        comparison_actor = copy.deepcopy(model._actor)
        args.comparison_actor_info = load_actor_module_override(comparison_actor, args.comparison_actor)
        comparison_actor.to(args.device)
        print(f"[INFO] Loaded read-only comparison Actor: {args.comparison_actor_info}")
    if args.perception_checkpoint is not None:
        print(f"[INFO] Enabled clean depth/temporal terrain_actor: {args.perception_checkpoint}")
    base_cfg, args.use_root_height_obs = load_mjlab_env_cfg(
        args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,
        device=args.device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=max(10.0, args.episode_length / 50.0 + 1.0),
    )
    if args.prompt_type == "reward":
        identifier = args.reward_task
        target_states = None
        if args.load_latent is not None:
            z = _load_prompt_latent(
                args.load_latent,
                prompt_type=args.prompt_type,
                identifier=identifier,
                device=args.device,
            )
        else:
            z, identifier, target_states = _compute_reward_z(args, model)
    else:
        if args.load_latent is not None:
            raise ValueError("--load-latent currently supports reward prompts only")
        z, identifier, target_states = _compute_goal_or_tracking_z(args, model, base_cfg)
        if args.prompt_type == "tracking":
            if args.tracking_repeats < 1:
                raise ValueError("--tracking-repeats must be positive")
            if args.tracking_repeats > 1:
                z = z.repeat((int(args.tracking_repeats), 1))
                identifier = f"{identifier}:repeat{int(args.tracking_repeats)}"
        if args.prompt_type == "goal" and args.goal_forward_latent is not None:
            forward_z, forward_identifier = _load_reward_latent_for_goal_composition(
                args.goal_forward_latent,
                device=args.device,
            )
            z = _compose_goal_and_forward_z(
                model,
                z,
                forward_z,
                weight=float(args.goal_forward_weight),
            ).detach()
            identifier = f"{identifier}+{float(args.goal_forward_weight):g}*{forward_identifier}"
            print(f"[INFO] composed moving goal prompt: identifier={identifier}")
    if args.save_latent is not None:
        _save_prompt_latent(
            args.save_latent,
            z,
            prompt_type=args.prompt_type,
            identifier=identifier,
        )
    args.prompt_identifier = identifier
    terrains = [value.strip() for value in args.terrains.split(",") if value.strip()]
    unknown = sorted(set(terrains) - set(SUPPORTED_TERRAINS))
    if unknown:
        raise ValueError(f"Unsupported terrains: {unknown}")
    same_z = clone_same_z_for_terrains(z, terrains)
    checksum = tensor_checksum(z)
    print(f"[INFO] prompt_type={args.prompt_type} prompt_source={identifier} z_shape={tuple(z.shape)} z_checksum={checksum}")
    results = []
    for terrain in terrains:
        assert tensor_checksum(same_z[terrain]) == checksum
        print(f"[INFO] terrain={terrain} z_checksum={checksum}")
        results.append(
            _run_rollout(
                args,
                model,
                base_cfg,
                terrain,
                same_z[terrain],
                target_states,
                comparison_actor=comparison_actor,
            )
        )
    if {row["z_checksum"] for row in results} != {checksum}:
        raise AssertionError("same-z checksum changed across terrain rollouts")
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    fieldnames = list(dict.fromkeys(key for row in results for key in row))
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
