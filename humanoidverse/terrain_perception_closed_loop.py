"""Frozen-policy closed-loop comparison of GT, single-frame, and temporal terrain maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.depth_terrain_evaluation import (
    TERRAIN_NAMES,
    build_depth_evaluation_env,
    synchronize_depth_and_gt,
)
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg, replace_hydra_override
from humanoidverse.perception.depth_camera import DepthCameraConfig, depth_frame_from_raycast
from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter
from humanoidverse.perception.temporal_terrain import TemporalTerrainCompletion, TerrainHistoryBuffer
from humanoidverse.terrain_transfer import tensor_checksum
from humanoidverse.terrain_transfer_inference import _separated_stairs_progress_metrics
from humanoidverse.utils.torch_utils import calc_heading_quat, get_euler_xyz

OBSERVATION_MODES = ("gt", "single", "temporal")
BOOLEAN_METRICS = (
    "center_departed",
    "first_transition",
    "outer_ground_reached",
    "stalled_at_center",
    "fell",
    "impact_safe",
    "normal_final_clearance",
    "traversal_success",
)
NUMERIC_METRICS = (
    "consecutive_steps_completed",
    "max_stair_level_reached",
    "mean_body_impact",
    "max_body_impact",
    "min_ground_clearance",
    "final_ground_clearance",
    "forward_displacement",
    "planar_displacement",
    "mean_root_velocity",
    "terrain_input_mae",
    "current_visible_fraction",
)


def _load_latent(path: Path, device: str) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("z"), torch.Tensor):
        raise ValueError(f"Invalid saved latent: {path}")
    z = payload["z"]
    if z.ndim != 2 or z.shape[0] != 1 or not torch.isfinite(z).all():
        raise ValueError("closed-loop evaluation requires one finite saved latent [1, Z]")
    checksum = tensor_checksum(z)
    if payload.get("z_checksum") != checksum:
        raise ValueError(
            f"Saved latent checksum mismatch: stored={payload.get('z_checksum')!r}, computed={checksum!r}"
        )
    return z.to(device), {**payload, "z_checksum": checksum}


def _load_perception(path: Path, device: str) -> tuple[TemporalTerrainCompletion, dict[str, Any]]:
    checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = TemporalTerrainCompletion(
        hidden_channels=int(config["hidden_channels"]),
        proprio_dim=int(config["proprio_dim"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def _state_dict_checksum(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_actor_override(model, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    actor_state = payload.get("actor") if isinstance(payload, dict) else None
    if not isinstance(actor_state, dict) or not actor_state:
        raise ValueError(f"invalid Actor milestone checkpoint: {resolved}")
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in actor_state.values()):
        raise ValueError(f"Actor milestone contains non-finite or non-tensor state: {resolved}")
    model._actor.load_state_dict(actor_state, strict=True)
    model.eval().requires_grad_(False)
    return {
        "path": str(resolved),
        "step": int(payload["step"]),
        "checksum": _state_dict_checksum(actor_state),
    }


def _default_target_states(wrapped_env) -> dict[str, torch.Tensor]:
    core = wrapped_env._env
    count = core.num_envs
    init = core.config.robot.init_state
    root_pos = torch.as_tensor(init.pos, device=core.device, dtype=torch.float32).unsqueeze(0) + core.env_origins
    root_rot = torch.as_tensor(init.rot, device=core.device, dtype=torch.float32).unsqueeze(0).expand(count, -1)
    root_state = torch.cat((root_pos, root_rot, torch.zeros((count, 6), device=core.device)), dim=-1)
    dof_state = torch.zeros((count, core.num_dof, 2), device=core.device)
    dof_state[..., 0] = core.default_dof_pos
    return {"root_states": root_state, "dof_states": dof_state}


def _initial_state(wrapped_env) -> torch.Tensor:
    qpos, qvel = wrapped_env._get_qpos_qvel(to_numpy=False)
    return torch.cat((qpos, qvel), dim=-1).detach().cpu()


def _ground_clearance(core) -> torch.Tensor:
    clearances = core._terrain_sensor_clearances()
    if core._terrain_reference_index is None:
        raise RuntimeError("terrain reference ray is unavailable")
    return clearances[:, core._terrain_reference_index]


def _body_impact(info: dict[str, Any], *, num_envs: int, device: str) -> torch.Tensor:
    value = info.get("aux_rewards", {}).get("penalty_body_impact")
    if value is None:
        return torch.zeros(num_envs, device=device)
    return torch.as_tensor(value, device=device, dtype=torch.float32).reshape(num_envs)


def _step_heights(core) -> torch.Tensor:
    terrain = core.mjlab_env.scene["terrain"]
    levels = terrain.terrain_levels.to(torch.float32)
    rows = int(terrain.terrain_origins.shape[0])
    difficulty_min, difficulty_max = (float(value) for value in core.config.terrain.difficulty_range)
    fraction = torch.zeros_like(levels) if rows <= 1 else levels / (rows - 1)
    difficulty = difficulty_min + fraction * (difficulty_max - difficulty_min)
    height_min, height_max = (float(value) for value in core.config.terrain.stairs.step_height_range)
    return height_min + difficulty * (height_max - height_min)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"episodes": len(rows)}
    for field in BOOLEAN_METRICS:
        if field in rows[0]:
            summary[f"{field}_rate"] = statistics.fmean(float(bool(row[field])) for row in rows)
    for field in NUMERIC_METRICS:
        if field in rows[0]:
            values = [float(row[field]) for row in rows]
            summary[f"{field}_mean"] = statistics.fmean(values)
            summary[f"{field}_min"] = min(values)
            summary[f"{field}_max"] = max(values)
    return summary


def _rollout_mode(
    *,
    mode: str,
    terrain: str,
    model,
    perception: TemporalTerrainCompletion,
    perception_config: dict[str, Any],
    env_config,
    latent: torch.Tensor,
    latent_checksum: str,
    num_envs: int,
    episode_steps: int,
    camera: DepthCameraConfig,
    device: str,
    expected_initial_state: torch.Tensor | None,
    fall_clearance: float,
    max_body_impact: float,
) -> tuple[list[dict[str, Any]], torch.Tensor, dict[str, Any]]:
    if mode not in OBSERVATION_MODES:
        raise ValueError(f"Unknown observation mode: {mode}")
    wrapped_env, _ = build_depth_evaluation_env(env_config, num_envs=num_envs, camera=camera)
    core = wrapped_env._env
    adapter = DepthTerrainAdapter(camera.intrinsics(), camera.height, camera.width).to(device)
    history = TerrainHistoryBuffer(
        batch_size=num_envs,
        time_steps=int(perception_config["sequence_steps"]),
        proprio_dim=int(perception_config["proprio_dim"]),
        device=device,
    )
    episode_time = torch.zeros(num_envs, device=device)
    pending_reset = torch.ones(num_envs, device=device, dtype=torch.bool)
    z = latent.expand(num_envs, -1)
    if tensor_checksum(latent) != latent_checksum:
        raise AssertionError("latent checksum changed before rollout")

    try:
        observation, _ = wrapped_env.reset(to_numpy=False, target_states=_default_target_states(wrapped_env))
        initial_state = _initial_state(wrapped_env)
        if expected_initial_state is not None:
            torch.testing.assert_close(initial_state, expected_initial_state, atol=0.0, rtol=0.0)
        initial_state_checksum = tensor_checksum(initial_state)
        initial_root = core.robot_root_states[:, :3].clone()
        final_root = initial_root.clone()
        previous_xy = initial_root[:, :2].clone()
        cumulative_path = torch.zeros(num_envs, device=device)
        active = torch.ones(num_envs, device=device, dtype=torch.bool)
        terminated_any = torch.zeros_like(active)
        root_velocity_sum = torch.zeros(num_envs, device=device)
        valid_steps = torch.zeros(num_envs, device=device)
        impact_sum = torch.zeros(num_envs, device=device)
        impact_max = torch.zeros(num_envs, device=device)
        input_error_sum = torch.zeros(num_envs, device=device)
        visibility_sum = torch.zeros(num_envs, device=device)
        ground_history: list[torch.Tensor] = []
        clearance_history: list[torch.Tensor] = []
        radius_history: list[torch.Tensor] = []
        impact_history: list[torch.Tensor] = []

        synchronize_depth_and_gt(core, camera.name)
        initial_clearance = _ground_clearance(core).clone()
        ground_history.append((core.robot_root_states[:, 2] - initial_clearance).detach().cpu())
        clearance_history.append(initial_clearance.detach().cpu())
        radius_history.append(torch.zeros(num_envs))

        with torch.inference_mode():
            for _step in range(episode_steps):
                synchronize_depth_and_gt(core, camera.name)
                frame = depth_frame_from_raycast(core.mjlab_env.scene.sensors[camera.name], camera)
                heading_quat = calc_heading_quat(core.base_quat, w_last=True)
                partial_map, visible_mask = adapter(
                    frame.depth_z,
                    frame.camera_pos_w,
                    frame.camera_optical_quat_w,
                    core.robot_root_states[:, :3],
                    heading_quat,
                )
                gt = core._terrain_actor_obs().clone()
                yaw = get_euler_xyz(core.base_quat, w_last=True)[2]
                history.reset(pending_reset)
                history.append(
                    partial_map=partial_map,
                    visible_mask=visible_mask,
                    pelvis_pos_w=core.robot_root_states[:, :3],
                    heading_yaw_w=yaw,
                    timestamp_s=episode_time,
                    proprio=observation["state"],
                )
                if mode == "gt":
                    terrain_input = gt
                else:
                    selected = history.single_frame_view() if mode == "single" else history
                    warped = selected.warp(
                        history_seconds=float(perception_config["history_seconds"]),
                        interpolation="bilinear",
                    )
                    completion = perception(warped, proprio=selected.proprio)
                    terrain_input = completion.completed_clearance
                    if not torch.isfinite(terrain_input).all():
                        raise RuntimeError(f"{mode} terrain completion produced non-finite Actor input")
                observation["terrain_actor"] = terrain_input
                valid_gt = torch.isfinite(gt)
                per_env_error = torch.where(valid_gt, (terrain_input - gt).abs(), 0.0).sum(dim=1)
                per_env_error /= valid_gt.sum(dim=1).clamp_min(1)
                input_error_sum[active] += per_env_error[active]
                visibility_sum[active] += visible_mask.float().mean(dim=1)[active]

                action = model.act(observation, z, mean=True)
                observation, _reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
                reset = torch.as_tensor(terminated, device=device).bool() | torch.as_tensor(truncated, device=device).bool()
                impact = _body_impact(info, num_envs=num_envs, device=device)
                current_active = active.clone()
                impact_sum[current_active] += impact[current_active]
                impact_max[current_active] = torch.maximum(impact_max[current_active], impact[current_active])
                root_velocity_sum[current_active] += torch.linalg.vector_norm(
                    core.robot_root_states[:, 7:9], dim=-1
                )[current_active]
                valid_steps[current_active] += 1
                nonreset_active = current_active & ~reset
                current_xy = core.robot_root_states[:, :2]
                cumulative_path[nonreset_active] += torch.linalg.vector_norm(
                    current_xy[nonreset_active] - previous_xy[nonreset_active], dim=-1
                )
                previous_xy[nonreset_active] = current_xy[nonreset_active]
                final_root[nonreset_active] = core.robot_root_states[nonreset_active, :3]

                clearance = _ground_clearance(core).clone()
                ground = core.robot_root_states[:, 2] - clearance
                ground_history.append(torch.where(nonreset_active, ground, torch.nan).detach().cpu())
                clearance_history.append(torch.where(nonreset_active, clearance, torch.nan).detach().cpu())
                radius = torch.linalg.vector_norm(core.robot_root_states[:, :2] - initial_root[:, :2], dim=-1)
                radius_history.append(torch.where(nonreset_active, radius, torch.nan).detach().cpu())
                impact_history.append(torch.where(current_active, impact, torch.nan).detach().cpu())
                terminated_any |= torch.as_tensor(terminated, device=device).bool()
                active &= ~reset
                episode_time += core.dt
                episode_time[reset] = 0.0
                pending_reset = reset

        ground_tensor = torch.stack(ground_history)
        clearance_tensor = torch.stack(clearance_history)
        radius_tensor = torch.stack(radius_history)
        impact_tensor = torch.stack(impact_history) if impact_history else torch.empty((0, num_envs))
        rows: list[dict[str, Any]] = []
        step_heights = _step_heights(core).detach().cpu() if terrain.startswith("stairs") else None
        terrain_levels = (
            core.mjlab_env.scene["terrain"].terrain_levels.detach().cpu()
            if terrain.startswith("stairs")
            else None
        )
        stairs_cfg = core.config.terrain.stairs
        for env_index in range(num_envs):
            ground_values = ground_tensor[:, env_index]
            clearance_values = clearance_tensor[:, env_index]
            radius_values = radius_tensor[:, env_index]
            impact_values = impact_tensor[:, env_index]
            valid = torch.isfinite(ground_values) & torch.isfinite(clearance_values) & torch.isfinite(radius_values)
            grounds = ground_values[valid].tolist()
            clearances = clearance_values[valid].tolist()
            radii = radius_values[valid].tolist()
            impacts = impact_values[torch.isfinite(impact_values)].tolist()
            steps = max(int(valid_steps[env_index].item()), 1)
            row: dict[str, Any] = {
                "mode": mode,
                "terrain": terrain,
                "env_index": env_index,
                "z_checksum": latent_checksum,
                "initial_state_checksum": initial_state_checksum,
                "episode_steps": int(valid_steps[env_index].item()),
                "fell": bool(terminated_any[env_index].item()) or min(clearances) < fall_clearance,
                "forward_displacement": float(final_root[env_index, 0] - initial_root[env_index, 0]),
                "planar_displacement": float(
                    torch.linalg.vector_norm(final_root[env_index, :2] - initial_root[env_index, :2])
                ),
                "mean_root_velocity": float(root_velocity_sum[env_index] / steps),
                "mean_body_impact": float(impact_sum[env_index] / steps),
                "max_body_impact": float(impact_max[env_index]),
                "min_ground_clearance": min(clearances),
                "final_ground_clearance": clearances[-1],
                "terrain_input_mae": float(input_error_sum[env_index] / steps),
                "current_visible_fraction": float(visibility_sum[env_index] / steps),
            }
            if terrain in {"stairs_up", "stairs_down"}:
                row.update(
                    _separated_stairs_progress_metrics(
                        terrain=terrain,
                        ground_heights=grounds,
                        ground_clearances=clearances,
                        body_impacts=impacts,
                        planar_radii=radii,
                        cumulative_planar_path=float(cumulative_path[env_index]),
                        step_height=float(step_heights[env_index]),
                        num_steps=int(stairs_cfg.num_steps),
                        center_width=float(stairs_cfg.platform_width),
                        fall_clearance=fall_clearance,
                        max_allowed_body_impact=max_body_impact,
                    )
                )
                row["terrain_level"] = int(terrain_levels[env_index])
                row["stairs_step_height"] = float(step_heights[env_index])
                row["traversal_success"] = bool(
                    row["outer_ground_reached"]
                    and row["impact_safe"]
                    and row["normal_final_clearance"]
                    and not row["fell"]
                )
            rows.append(row)
        diagnostics = {
            "mode": mode,
            "initial_state_checksum": initial_state_checksum,
            "history_valid_after_final_step_min": int(history.frame_valid.sum(dim=1).min().item()),
            "history_valid_after_final_step_max": int(history.frame_valid.sum(dim=1).max().item()),
        }
        return rows, initial_state, diagnostics
    finally:
        wrapped_env.close()


def evaluate_closed_loop(
    *,
    model_folder: Path,
    perception_checkpoint: Path,
    latent_path: Path,
    output_dir: Path,
    terrain: str,
    num_envs: int,
    episode_steps: int,
    seed: int,
    device: str,
    camera: DepthCameraConfig,
    fall_clearance: float,
    max_body_impact: float,
    modes: tuple[str, ...] = OBSERVATION_MODES,
    actor_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if terrain not in TERRAIN_NAMES:
        raise ValueError(f"Unsupported terrain: {terrain!r}")
    if min(num_envs, episode_steps) <= 0:
        raise ValueError("num_envs and episode_steps must be positive")
    if not modes or len(set(modes)) != len(modes) or any(mode not in OBSERVATION_MODES for mode in modes):
        raise ValueError(f"modes must be a unique non-empty subset of {OBSERVATION_MODES}")
    model_folder = model_folder.expanduser().resolve()
    model = load_model_from_checkpoint_dir(
        model_folder / "checkpoint", device=checkpoint_load_device(device)
    )
    model.to(device).eval()
    actor_override = _load_actor_override(model, actor_checkpoint) if actor_checkpoint is not None else None
    perception, perception_checkpoint_data = _load_perception(perception_checkpoint, device)
    perception_config = perception_checkpoint_data["config"]
    latent, latent_payload = _load_latent(latent_path, device)
    latent_checksum = str(latent_payload["z_checksum"])
    env_config, _ = load_mjlab_env_cfg(
        model_folder,
        data_path=None,
        robot_config=None,
        device=device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=max(10.0, episode_steps / 50.0 + 1.0),
    )
    env_config = env_config.model_copy(
        update={
            "seed": seed,
            "hydra_overrides": replace_hydra_override(
                list(env_config.hydra_overrides), "terrain.terrain_type", terrain
            ),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    expected_initial_state = None
    for mode in modes:
        rows, initial_state, mode_diagnostics = _rollout_mode(
            mode=mode,
            terrain=terrain,
            model=model,
            perception=perception,
            perception_config=perception_config,
            env_config=env_config,
            latent=latent,
            latent_checksum=latent_checksum,
            num_envs=num_envs,
            episode_steps=episode_steps,
            camera=camera,
            device=device,
            expected_initial_state=expected_initial_state,
            fall_clearance=fall_clearance,
            max_body_impact=max_body_impact,
        )
        if expected_initial_state is None:
            expected_initial_state = initial_state
        all_rows.extend(rows)
        diagnostics.append(mode_diagnostics)

    metrics_path = output_dir / "metrics.csv"
    fieldnames = list(dict.fromkeys(key for row in all_rows for key in row))
    with metrics_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    status_path = model_folder / "checkpoint" / "train_status.json"
    train_status = json.loads(status_path.read_text()) if status_path.exists() else None
    summary = {
        "model_folder": str(model_folder),
        "checkpoint_global_time": int(train_status["global_time"]) if train_status else None,
        "actor_override": actor_override,
        "perception_checkpoint": str(perception_checkpoint.expanduser().resolve()),
        "perception_epoch": int(perception_checkpoint_data["epoch"]),
        "latent_path": str(latent_path.expanduser().resolve()),
        "latent_prompt_type": latent_payload.get("prompt_type"),
        "latent_prompt_identifier": latent_payload.get("prompt_identifier"),
        "z_checksum": latent_checksum,
        "terrain": terrain,
        "seed": seed,
        "num_envs": num_envs,
        "episode_steps": episode_steps,
        "action_selection": "deterministic mean=True",
        "camera": asdict(camera),
        "initial_state_identical_across_modes": True,
        "diagnostics": diagnostics,
        "modes": {
            mode: _aggregate([row for row in all_rows if row["mode"] == mode])
            for mode in modes
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_results.json").write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--perception-checkpoint", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terrain", choices=TERRAIN_NAMES, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=6840)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fall-clearance", type=float, default=0.45)
    parser.add_argument("--max-body-impact", type=float, default=1.0)
    parser.add_argument("--modes", nargs="+", choices=OBSERVATION_MODES, default=list(OBSERVATION_MODES))
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--horizontal-fov", type=float, default=89.0)
    parser.add_argument("--vertical-fov", type=float, default=58.0)
    parser.add_argument("--down-pitch", type=float, default=48.0)
    parser.add_argument("--min-range", type=float, default=0.10)
    parser.add_argument("--max-range", type=float, default=2.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = DepthCameraConfig(
        width=args.width,
        height=args.height,
        horizontal_fov_deg=args.horizontal_fov,
        vertical_fov_deg=args.vertical_fov,
        down_pitch_deg=args.down_pitch,
        min_range=args.min_range,
        max_range=args.max_range,
        include_geom_groups=(5,),
    )
    summary = evaluate_closed_loop(
        model_folder=args.model_folder,
        perception_checkpoint=args.perception_checkpoint,
        latent_path=args.latent,
        output_dir=args.output_dir,
        terrain=args.terrain,
        num_envs=args.num_envs,
        episode_steps=args.episode_steps,
        seed=args.seed,
        device=args.device,
        camera=camera,
        fall_clearance=args.fall_clearance,
        max_body_impact=args.max_body_impact,
        modes=tuple(args.modes),
        actor_checkpoint=args.actor_checkpoint,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
