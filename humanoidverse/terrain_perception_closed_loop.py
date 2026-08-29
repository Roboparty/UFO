"""Frozen-policy closed-loop comparison of GT, single-frame, and temporal terrain maps."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from humanoidverse.actor_override import load_actor_override as _load_actor_override
from humanoidverse.actor_override import state_dict_checksum as _state_dict_checksum
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.depth_terrain_evaluation import (
    TERRAIN_NAMES,
    build_depth_evaluation_env,
    synchronize_depth_and_gt,
)
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg, replace_hydra_override
from humanoidverse.perception.depth_augmentation import (
    CameraFrameScheduler,
    DepthCalibrationAugmentationConfig,
    DepthTimingAugmentationConfig,
    LocalCalibrationAugmentation,
    MetricDepthAugmentation,
    MetricDepthAugmentationConfig,
)
from humanoidverse.perception.depth_camera import (
    DepthCameraConfig,
    depth_frame_from_raycast,
    optical_depth_from_raycast,
    rotation_matrix_to_xyzw,
)
from humanoidverse.perception.depth_noise import (
    DepthNoiseConfig,
    DepthNoisePipeline,
    depth_noise_preset,
)
from humanoidverse.perception.depth_preprocessing import (
    DepthCropConfig,
    crop_and_resize_depth,
    crop_and_resize_depth_with_conservative_invalid_mask,
    crop_and_scale_intrinsics,
)
from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter
from humanoidverse.perception.local_depth_terrain_adapter import LocalDepthTerrainAdapter
from humanoidverse.perception.self_occluding_depth import (
    SelfOcclusionDepthConfig,
    make_self_occlusion_camera_pair,
    self_occluding_depth_from_sensors,
)
from humanoidverse.perception.temporal_terrain import (
    OdometryFreeTerrainHistoryBuffer,
    resolve_terrain_output_mode,
    select_terrain_actor_clearance,
    TemporalTerrainCompletion,
    TerrainHistoryBuffer,
)
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
    "underfoot_mae",
    "stairs_edge_mae",
    "current_visible_fraction",
    "temporal_coverage_fraction",
    "action_deviation_from_clean",
    "sensor_clean_valid_fraction",
    "sensor_noisy_valid_fraction",
    "sensor_latency_frames",
    "sensor_latency_seconds",
    "sensor_extrinsic_translation_norm_m",
    "sensor_extrinsic_rotation_deg",
)


def _per_env_map_metrics(
    terrain_input: torch.Tensor,
    target: torch.Tensor,
    *,
    current_visible: torch.Tensor,
    history_visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map-error and coverage metrics without reducing across environments."""
    finite = torch.isfinite(terrain_input) & torch.isfinite(target)
    error = torch.where(finite, (terrain_input - target).abs(), 0.0)
    valid_count = finite.sum(dim=1).clamp_min(1)
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-0.4, 1.6, 21, device=target.device, dtype=target.dtype),
        torch.linspace(-0.6, 0.6, 13, device=target.device, dtype=target.dtype),
        indexing="ij",
    )
    underfoot = ((grid_x.abs() <= 0.2001) & (grid_y.abs() <= 0.2001)).reshape(1, -1)
    values = target.reshape(-1, 21, 13)
    edge = torch.zeros_like(values, dtype=torch.bool)
    edge[:, 1:, :] |= (values[:, 1:, :] - values[:, :-1, :]).abs() > 0.04
    edge[:, :-1, :] |= (values[:, 1:, :] - values[:, :-1, :]).abs() > 0.04
    edge[:, :, 1:] |= (values[:, :, 1:] - values[:, :, :-1]).abs() > 0.04
    edge[:, :, :-1] |= (values[:, :, 1:] - values[:, :, :-1]).abs() > 0.04
    edge = edge.reshape(target.shape) & finite

    def masked_mean(mask: torch.Tensor) -> torch.Tensor:
        mask = mask & finite
        return (error * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    return {
        "terrain_input_mae": error.sum(dim=1) / valid_count,
        "underfoot_mae": masked_mean(underfoot.expand_as(finite)),
        "stairs_edge_mae": masked_mean(edge),
        "current_visible_fraction": (current_visible & finite).sum(dim=1) / valid_count,
        "temporal_coverage_fraction": (history_visible & finite).sum(dim=1) / valid_count,
    }


def _load_latent(path: Path, device: str) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("z"), torch.Tensor):
        raise ValueError(f"Invalid saved latent: {path}")
    z = payload["z"]
    if z.ndim != 2 or z.shape[0] != 1 or not torch.isfinite(z).all():
        raise ValueError("closed-loop evaluation requires one finite saved latent [1, Z]")
    checksum = tensor_checksum(z)
    if payload.get("z_checksum") != checksum:
        raise ValueError(f"Saved latent checksum mismatch: stored={payload.get('z_checksum')!r}, computed={checksum!r}")
    return z.to(device), {**payload, "z_checksum": checksum}


def _load_perception(path: Path, device: str) -> tuple[TemporalTerrainCompletion, dict[str, Any]]:
    checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = TemporalTerrainCompletion(
        hidden_channels=int(config["hidden_channels"]),
        proprio_dim=int(config["proprio_dim"]),
        proprio_channels=int(config.get("proprio_channels", 8)),
        motion_feature_dim=int(config.get("motion_feature_dim", 6)),
        use_grid_coordinates=bool(config.get("use_grid_coordinates", False)),
        global_context_dim=int(config.get("global_context_dim", 0)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


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
    noise_config: DepthNoiseConfig,
    noise_seed: int,
) -> tuple[list[dict[str, Any]], torch.Tensor, dict[str, Any]]:
    if mode not in OBSERVATION_MODES:
        raise ValueError(f"Unknown observation mode: {mode}")
    history_mode = str(perception_config.get("history_mode", "egomotion_warp"))
    terrain_output_mode = resolve_terrain_output_mode(perception_config)
    dataset_metadata = dict(perception_config.get("dataset_metadata", {}))
    semantic_config = None
    scene_camera = None
    if history_mode == "no_odometry" and dataset_metadata.get("self_occlusion"):
        semantic_contract = dataset_metadata.get("self_occlusion_contract")
        if not isinstance(semantic_contract, dict):
            raise ValueError("self-occluding checkpoint is missing its semantic contract")
        semantic_config = SelfOcclusionDepthConfig.from_metadata(semantic_contract)
        camera, scene_camera = make_self_occlusion_camera_pair(camera, semantic_config)
        wrapped_env, _ = build_depth_evaluation_env(
            env_config,
            num_envs=num_envs,
            camera=camera,
            extra_cameras=(
                (
                    scene_camera,
                    False,
                    semantic_config.camera_housing_geom_names,
                    semantic_config.camera_housing_mesh_names,
                    semantic_config.camera_housing_geom_group,
                ),
            ),
        )
    else:
        wrapped_env, _ = build_depth_evaluation_env(env_config, num_envs=num_envs, camera=camera)
    core = wrapped_env._env
    if history_mode == "no_odometry":
        camera_quat_torso = rotation_matrix_to_xyzw(camera.torso_from_optical().to(device=device, dtype=torch.float32))
        target_image = dict(dataset_metadata.get("target_image", {}))
        target_width = int(target_image.get("width", camera.width))
        target_height = int(target_image.get("height", camera.height))
        depth_crop = DepthCropConfig.from_metadata(dataset_metadata.get("depth_crop"))
        target_intrinsics = crop_and_scale_intrinsics(
            camera.intrinsics(),
            native_width=camera.width,
            native_height=camera.height,
            target_width=target_width,
            target_height=target_height,
            crop=depth_crop,
        )
        adapter = LocalDepthTerrainAdapter(
            target_intrinsics,
            target_height,
            target_width,
            camera_pos_torso=camera.mount_pos_torso,
            camera_optical_quat_torso_xyzw=tuple(float(value) for value in camera_quat_torso),
        ).to(device)
        augmentation_metadata = dict(dataset_metadata.get("depth_augmentation") or {})
        deployment_preprocessing = dict(dataset_metadata.get("deployment_depth_preprocessing") or {})
        clean_augmentation_config = MetricDepthAugmentationConfig(
            max_depth_m=float(deployment_preprocessing.get("max_depth_m", 2.0)),
            blur_probability=1.0,
            sigma_min_px=float(deployment_preprocessing.get("blur_sigma_px", 1.5)),
            sigma_max_px=float(deployment_preprocessing.get("blur_sigma_px", 1.5)),
        )
        if noise_config.is_identity():
            noisy_augmentation_config = clean_augmentation_config
        elif noise_config.condition == "combined" and augmentation_metadata:
            noisy_augmentation_config = MetricDepthAugmentationConfig(**augmentation_metadata)
        else:
            noisy_augmentation_config = replace(
                clean_augmentation_config,
                measurement_base_std_m=noise_config.measurement.base_std_m,
                measurement_quadratic_std_m_per_m2=noise_config.measurement.quadratic_std_m_per_m2,
                edge_depth_threshold_m=noise_config.edge.depth_threshold_m,
                edge_corruption_probability=noise_config.edge.corruption_probability,
                edge_invalid_probability=noise_config.edge.invalid_probability,
                pixel_dropout_probability=noise_config.dropout.probability,
            )
        local_depth_augmentation = MetricDepthAugmentation(
            noisy_augmentation_config,
            seed=noise_seed + 17_003,
        )
        clean_depth_augmentation = MetricDepthAugmentation(
            clean_augmentation_config,
            seed=noise_seed + 17_103,
        )
        calibration_metadata = dict(dataset_metadata.get("depth_calibration") or {})
        if noise_config.is_identity():
            local_calibration_config = DepthCalibrationAugmentationConfig()
        elif noise_config.condition == "combined" and calibration_metadata:
            local_calibration_config = DepthCalibrationAugmentationConfig(**calibration_metadata)
        else:
            local_calibration_config = DepthCalibrationAugmentationConfig(
                translation_bound_m=noise_config.extrinsic.translation_bound_m,
                rotation_bound_deg=noise_config.extrinsic.rotation_bound_deg,
            )
        local_calibration = LocalCalibrationAugmentation(
            local_calibration_config,
            intrinsic_matrix=target_intrinsics,
            camera_pos_torso=camera.mount_pos_torso,
            camera_optical_quat_torso_xyzw=tuple(float(value) for value in camera_quat_torso),
            batch_size=num_envs,
            device=device,
            seed=noise_seed + 19_009,
        )
        timing_metadata = dict(dataset_metadata.get("depth_timing") or {})
        base_camera_frequency = float(timing_metadata.get("camera_frequency_hz", 30.0))
        clean_timing_config = DepthTimingAugmentationConfig(
            camera_frequency_hz=base_camera_frequency,
            control_frequency_hz=1.0 / float(core.dt),
        )
        if not noise_config.is_identity() and noise_config.condition in {"latency", "combined"} and timing_metadata:
            noisy_timing_config = DepthTimingAugmentationConfig(**timing_metadata)
        else:
            noisy_timing_config = clean_timing_config
        local_frame_scheduler = CameraFrameScheduler(
            noisy_timing_config,
            batch_size=num_envs,
            device=device,
            seed=noise_seed + 23_011,
        )
        clean_frame_scheduler = CameraFrameScheduler(
            clean_timing_config,
            batch_size=num_envs,
            device=device,
            seed=noise_seed + 23_111,
        )
        waist_indices = torch.tensor(
            [core.dof_names.index(name) for name in core.config.robot.waist_dof_names],
            device=device,
            dtype=torch.long,
        )
        history = OdometryFreeTerrainHistoryBuffer(
            batch_size=num_envs,
            time_steps=int(perception_config["sequence_steps"]),
            proprio_dim=int(perception_config["proprio_dim"]),
            device=device,
        )
        clean_history = OdometryFreeTerrainHistoryBuffer(
            batch_size=num_envs,
            time_steps=int(perception_config["sequence_steps"]),
            proprio_dim=int(perception_config["proprio_dim"]),
            device=device,
        )
        noise_pipeline = None
    else:
        adapter = DepthTerrainAdapter(camera.intrinsics(), camera.height, camera.width).to(device)
        waist_indices = None
        history = TerrainHistoryBuffer(
            batch_size=num_envs,
            time_steps=int(perception_config["sequence_steps"]),
            proprio_dim=int(perception_config["proprio_dim"]),
            device=device,
        )
        clean_history = TerrainHistoryBuffer(
            batch_size=num_envs,
            time_steps=int(perception_config["sequence_steps"]),
            proprio_dim=int(perception_config["proprio_dim"]),
            device=device,
        )
        noise_pipeline = DepthNoisePipeline(
            noise_config,
            batch_size=num_envs,
            image_height=camera.height,
            image_width=camera.width,
            device=device,
            noise_seed=noise_seed,
        )
        local_depth_augmentation = None
        clean_depth_augmentation = None
        depth_crop = None
        local_calibration = None
        local_frame_scheduler = None
        clean_frame_scheduler = None
        target_height, target_width = camera.height, camera.width
    env_ids = torch.arange(num_envs, device=device, dtype=torch.int64)
    identity_noise = noise_config.is_identity()
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
        underfoot_error_sum = torch.zeros(num_envs, device=device)
        edge_error_sum = torch.zeros(num_envs, device=device)
        visibility_sum = torch.zeros(num_envs, device=device)
        temporal_coverage_sum = torch.zeros(num_envs, device=device)
        action_deviation_sum = torch.zeros(num_envs, device=device)
        sensor_diagnostic_sums = {
            name: torch.zeros(num_envs, device=device)
            for name in (
                "clean_valid_fraction",
                "noisy_valid_fraction",
                "latency_frames",
                "latency_seconds",
                "extrinsic_translation_norm_m",
                "extrinsic_rotation_deg",
            )
        }
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
                sensor_names = (camera.name, scene_camera.name) if scene_camera is not None else camera.name
                synchronize_depth_and_gt(core, sensor_names)
                sensor = core.mjlab_env.scene.sensors[camera.name]
                if history_mode == "no_odometry":
                    assert isinstance(adapter, LocalDepthTerrainAdapter)
                    assert waist_indices is not None
                    assert local_depth_augmentation is not None
                    assert clean_depth_augmentation is not None and depth_crop is not None
                    assert local_calibration is not None
                    assert local_frame_scheduler is not None and clean_frame_scheduler is not None
                    local_calibration.reset(pending_reset)
                    local_frame_scheduler.reset(pending_reset)
                    clean_frame_scheduler.reset(pending_reset)
                    frame_valid, frame_timestamp, _timing = local_frame_scheduler.step(episode_time)
                    clean_frame_valid, clean_frame_timestamp, _clean_timing = clean_frame_scheduler.step(episode_time)
                    if semantic_config is not None:
                        assert scene_camera is not None
                        semantic_frame = self_occluding_depth_from_sensors(
                            sensor,
                            core.mjlab_env.scene.sensors[scene_camera.name],
                            camera,
                            semantic_config,
                            local_depth_augmentation,
                        )
                        if torch.any(semantic_frame.ambiguous_mask):
                            raise RuntimeError("scene/terrain raycasts produced ambiguous first hits")
                        depth_z, resized_self_mask = crop_and_resize_depth_with_conservative_invalid_mask(
                            semantic_frame.final_depth_z,
                            semantic_frame.dilated_self_mask,
                            target_height=target_height,
                            target_width=target_width,
                            crop=depth_crop,
                        )
                        if torch.any(torch.isfinite(depth_z) & resized_self_mask):
                            raise RuntimeError("conservative self mask was lost before terrain projection")
                        noisy_valid = semantic_frame.valid_terrain_mask
                        if identity_noise:
                            clean_depth_z = depth_z
                            clean_valid = noisy_valid
                        else:
                            clean_semantic_frame = self_occluding_depth_from_sensors(
                                sensor,
                                core.mjlab_env.scene.sensors[scene_camera.name],
                                camera,
                                semantic_config,
                                clean_depth_augmentation,
                            )
                            clean_depth_z, clean_self_mask = crop_and_resize_depth_with_conservative_invalid_mask(
                                clean_semantic_frame.final_depth_z,
                                clean_semantic_frame.dilated_self_mask,
                                target_height=target_height,
                                target_width=target_width,
                                crop=depth_crop,
                            )
                            if torch.any(torch.isfinite(clean_depth_z) & clean_self_mask):
                                raise RuntimeError("clean conservative self mask was lost before terrain projection")
                            clean_valid = clean_semantic_frame.valid_terrain_mask
                    else:
                        raw_depth_z, _range_image, _ray_valid = optical_depth_from_raycast(sensor, camera)
                        depth_z, noisy_valid, _sigma_px = local_depth_augmentation(raw_depth_z)
                        depth_z = crop_and_resize_depth(
                            depth_z,
                            target_height=target_height,
                            target_width=target_width,
                            crop=depth_crop,
                        )
                        if identity_noise:
                            clean_depth_z = depth_z
                            clean_valid = noisy_valid
                        else:
                            clean_depth_z, clean_valid, _clean_sigma = clean_depth_augmentation(raw_depth_z)
                            clean_depth_z = crop_and_resize_depth(
                                clean_depth_z,
                                target_height=target_height,
                                target_width=target_width,
                                crop=depth_crop,
                            )
                    partial_map, visible_mask = adapter(
                        depth_z,
                        core.projected_gravity,
                        core.dof_pos[:, waist_indices],
                        intrinsic_matrix=local_calibration.intrinsics,
                        camera_pos_torso=local_calibration.camera_pos_torso,
                        camera_optical_quat_torso_xyzw=local_calibration.camera_quat_torso,
                    )
                    if identity_noise:
                        clean_partial_map, clean_visible_mask = partial_map, visible_mask
                        clean_frame_valid = frame_valid
                        clean_frame_timestamp = frame_timestamp
                    else:
                        clean_partial_map, clean_visible_mask = adapter(
                            clean_depth_z,
                            core.projected_gravity,
                            core.dof_pos[:, waist_indices],
                        )
                    translation_error = local_calibration.camera_pos_torso - adapter.camera_pos_torso
                    base_quaternion = adapter.camera_optical_quat_torso_xyzw.expand(num_envs, -1)
                    quaternion_dot = (local_calibration.camera_quat_torso * base_quaternion).sum(dim=-1).abs()
                    zero = torch.zeros(num_envs, device=device)
                    sensor_diagnostics = {
                        "clean_valid_fraction": clean_valid.float().mean(dim=(1, 2)),
                        "noisy_valid_fraction": noisy_valid.float().mean(dim=(1, 2)),
                        "latency_frames": zero,
                        "latency_seconds": zero,
                        "extrinsic_translation_norm_m": torch.linalg.vector_norm(translation_error, dim=-1),
                        "extrinsic_rotation_deg": torch.rad2deg(2.0 * torch.acos(quaternion_dot.clamp(max=1.0))),
                    }
                else:
                    assert isinstance(adapter, DepthTerrainAdapter)
                    assert noise_pipeline is not None
                    frame = depth_frame_from_raycast(sensor, camera)
                    noisy_frame = noise_pipeline(
                        depth_z=frame.depth_z,
                        camera_pos_w=frame.camera_pos_w,
                        camera_optical_quat_w=frame.camera_optical_quat_w,
                        timestamp_s=episode_time,
                        env_ids=env_ids,
                        reset_mask=pending_reset,
                    )
                    heading_quat = calc_heading_quat(core.base_quat, w_last=True)
                    partial_map, visible_mask = adapter(
                        noisy_frame.depth_z,
                        noisy_frame.camera_pos_w,
                        noisy_frame.camera_optical_quat_w,
                        core.robot_root_states[:, :3],
                        heading_quat,
                    )
                    if identity_noise:
                        clean_partial_map, clean_visible_mask = partial_map, visible_mask
                    else:
                        clean_partial_map, clean_visible_mask = adapter(
                            frame.depth_z,
                            frame.camera_pos_w,
                            frame.camera_optical_quat_w,
                            core.robot_root_states[:, :3],
                            heading_quat,
                        )
                    frame_timestamp = noisy_frame.timestamp_s
                    sensor_diagnostics = noisy_frame.diagnostics
                gt = core._terrain_actor_obs().clone()
                history.reset(pending_reset)
                clean_history.reset(pending_reset)
                if history_mode == "no_odometry":
                    assert isinstance(history, OdometryFreeTerrainHistoryBuffer)
                    assert isinstance(clean_history, OdometryFreeTerrainHistoryBuffer)
                    history.append(
                        partial_map=partial_map,
                        visible_mask=visible_mask,
                        timestamp_s=frame_timestamp,
                        proprio=observation["state"],
                        append_mask=frame_valid,
                    )
                    clean_history.append(
                        partial_map=clean_partial_map,
                        visible_mask=clean_visible_mask,
                        timestamp_s=clean_frame_timestamp,
                        proprio=observation["state"],
                        append_mask=clean_frame_valid,
                    )
                    visible_mask = history.visible_masks[:, -1]
                    clean_visible_mask = clean_history.visible_masks[:, -1]
                else:
                    assert isinstance(history, TerrainHistoryBuffer)
                    assert isinstance(clean_history, TerrainHistoryBuffer)
                    yaw = get_euler_xyz(core.base_quat, w_last=True)[2]
                    history.append(
                        partial_map=partial_map,
                        visible_mask=visible_mask,
                        pelvis_pos_w=core.robot_root_states[:, :3],
                        heading_yaw_w=yaw,
                        timestamp_s=frame_timestamp,
                        proprio=observation["state"],
                    )
                    clean_history.append(
                        partial_map=clean_partial_map,
                        visible_mask=clean_visible_mask,
                        pelvis_pos_w=core.robot_root_states[:, :3],
                        heading_yaw_w=yaw,
                        timestamp_s=episode_time,
                        proprio=observation["state"],
                    )
                if mode == "gt":
                    terrain_input = gt
                    clean_terrain_input = gt
                    history_visible = visible_mask
                else:
                    selected = history.single_frame_view() if mode == "single" else history
                    warped = (
                        selected.history(history_seconds=float(perception_config["history_seconds"]))
                        if isinstance(selected, OdometryFreeTerrainHistoryBuffer)
                        else selected.warp(
                            history_seconds=float(perception_config["history_seconds"]),
                            interpolation="bilinear",
                        )
                    )
                    completion = perception(warped, proprio=selected.proprio)
                    terrain_input = select_terrain_actor_clearance(completion, mode=terrain_output_mode)
                    if not torch.isfinite(terrain_input).all():
                        raise RuntimeError(f"{mode} terrain completion produced non-finite Actor input")
                    if identity_noise:
                        clean_terrain_input = terrain_input
                    else:
                        clean_selected = clean_history.single_frame_view() if mode == "single" else clean_history
                        clean_warped = (
                            clean_selected.history(history_seconds=float(perception_config["history_seconds"]))
                            if isinstance(clean_selected, OdometryFreeTerrainHistoryBuffer)
                            else clean_selected.warp(
                                history_seconds=float(perception_config["history_seconds"]),
                                interpolation="bilinear",
                            )
                        )
                        clean_terrain_input = select_terrain_actor_clearance(
                            perception(clean_warped, proprio=clean_selected.proprio),
                            mode=terrain_output_mode,
                        )
                    history_visible = warped.visible_masks.any(dim=1)
                observation["terrain_actor"] = terrain_input
                map_metrics = _per_env_map_metrics(
                    terrain_input,
                    gt,
                    current_visible=visible_mask,
                    history_visible=history_visible,
                )
                input_error_sum[active] += map_metrics["terrain_input_mae"][active]
                underfoot_error_sum[active] += map_metrics["underfoot_mae"][active]
                edge_error_sum[active] += map_metrics["stairs_edge_mae"][active]
                visibility_sum[active] += map_metrics["current_visible_fraction"][active]
                temporal_coverage_sum[active] += map_metrics["temporal_coverage_fraction"][active]
                action = model.act(observation, z, mean=True)
                if identity_noise or mode == "gt":
                    clean_action = action
                else:
                    clean_observation = dict(observation)
                    clean_observation["terrain_actor"] = clean_terrain_input
                    clean_action = model.act(clean_observation, z, mean=True)
                action_deviation = torch.linalg.vector_norm(action - clean_action, dim=-1)
                action_deviation_sum[active] += action_deviation[active]
                for name, values in sensor_diagnostics.items():
                    if name in sensor_diagnostic_sums:
                        sensor_diagnostic_sums[name][active] += values[active]
                observation, _reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
                reset = torch.as_tensor(terminated, device=device).bool() | torch.as_tensor(truncated, device=device).bool()
                impact = _body_impact(info, num_envs=num_envs, device=device)
                current_active = active.clone()
                impact_sum[current_active] += impact[current_active]
                impact_max[current_active] = torch.maximum(impact_max[current_active], impact[current_active])
                root_velocity_sum[current_active] += torch.linalg.vector_norm(core.robot_root_states[:, 7:9], dim=-1)[current_active]
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
        terrain_levels = core.mjlab_env.scene["terrain"].terrain_levels.detach().cpu() if terrain.startswith("stairs") else None
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
                "planar_displacement": float(torch.linalg.vector_norm(final_root[env_index, :2] - initial_root[env_index, :2])),
                "mean_root_velocity": float(root_velocity_sum[env_index] / steps),
                "mean_body_impact": float(impact_sum[env_index] / steps),
                "max_body_impact": float(impact_max[env_index]),
                "min_ground_clearance": min(clearances),
                "final_ground_clearance": clearances[-1],
                "terrain_input_mae": float(input_error_sum[env_index] / steps),
                "underfoot_mae": float(underfoot_error_sum[env_index] / steps),
                "stairs_edge_mae": float(edge_error_sum[env_index] / steps),
                "current_visible_fraction": float(visibility_sum[env_index] / steps),
                "temporal_coverage_fraction": float(temporal_coverage_sum[env_index] / steps),
                "action_deviation_from_clean": float(action_deviation_sum[env_index] / steps),
                "sensor_clean_valid_fraction": float(sensor_diagnostic_sums["clean_valid_fraction"][env_index] / steps),
                "sensor_noisy_valid_fraction": float(sensor_diagnostic_sums["noisy_valid_fraction"][env_index] / steps),
                "sensor_latency_frames": float(sensor_diagnostic_sums["latency_frames"][env_index] / steps),
                "sensor_latency_seconds": float(sensor_diagnostic_sums["latency_seconds"][env_index] / steps),
                "sensor_extrinsic_translation_norm_m": float(sensor_diagnostic_sums["extrinsic_translation_norm_m"][env_index] / steps),
                "sensor_extrinsic_rotation_deg": float(sensor_diagnostic_sums["extrinsic_rotation_deg"][env_index] / steps),
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
                    row["outer_ground_reached"] and row["impact_safe"] and row["normal_final_clearance"] and not row["fell"]
                )
            rows.append(row)
        diagnostics = {
            "mode": mode,
            "initial_state_checksum": initial_state_checksum,
            "history_valid_after_final_step_min": int(history.frame_valid.sum(dim=1).min().item()),
            "history_valid_after_final_step_max": int(history.frame_valid.sum(dim=1).max().item()),
            "noise_condition": noise_config.condition,
            "noise_severity": noise_config.severity,
            "noise_seed": noise_seed,
            "noise_config_hash": noise_config.hash(),
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
    noise_config: DepthNoiseConfig | None = None,
    noise_seed: int = 0,
) -> dict[str, Any]:
    if terrain not in TERRAIN_NAMES:
        raise ValueError(f"Unsupported terrain: {terrain!r}")
    if min(num_envs, episode_steps) <= 0:
        raise ValueError("num_envs and episode_steps must be positive")
    if not modes or len(set(modes)) != len(modes) or any(mode not in OBSERVATION_MODES for mode in modes):
        raise ValueError(f"modes must be a unique non-empty subset of {OBSERVATION_MODES}")
    noise_config = noise_config or depth_noise_preset("clean", max_depth_m=camera.max_range)
    noise_config.validate()
    model_folder = model_folder.expanduser().resolve()
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=checkpoint_load_device(device))
    model.to(device).eval()
    actor_override = _load_actor_override(model, actor_checkpoint) if actor_checkpoint is not None else None
    actor_checksum = _state_dict_checksum(model._actor.state_dict())
    perception, perception_checkpoint_data = _load_perception(perception_checkpoint, device)
    perception_checksum = _state_dict_checksum(perception.state_dict())
    perception_config = perception_checkpoint_data["config"]
    perception_dataset_metadata = dict(perception_config.get("dataset_metadata", {}))
    if perception_config.get("history_mode") == "no_odometry":
        if perception_config.get("dataset_schema") != "odometry_free_local":
            raise ValueError("no-odometry perception checkpoint has an incompatible dataset schema")
        if not perception_dataset_metadata.get("camera"):
            raise ValueError("no-odometry perception checkpoint does not record its native camera")
        camera = DepthCameraConfig(**dict(perception_dataset_metadata["camera"]))
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
            "hydra_overrides": replace_hydra_override(list(env_config.hydra_overrides), "terrain.terrain_type", terrain),
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
            noise_config=noise_config,
            noise_seed=noise_seed,
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
        "actor_checksum": actor_checksum,
        "perception_checkpoint": str(perception_checkpoint.expanduser().resolve()),
        "perception_epoch": int(perception_checkpoint_data["epoch"]),
        "perception_checksum": perception_checksum,
        "perception_input_contract": {
            "history_mode": perception_config.get("history_mode", "egomotion_warp"),
            "terrain_output_mode": resolve_terrain_output_mode(perception_config),
            "dataset_schema": perception_config.get("dataset_schema"),
            "uses_odometry": perception_dataset_metadata.get("uses_odometry"),
            "deployment_inputs": perception_dataset_metadata.get("deployment_inputs"),
            "native_camera": perception_dataset_metadata.get("camera"),
            "target_image": perception_dataset_metadata.get("target_image"),
            "depth_augmentation": perception_dataset_metadata.get("depth_augmentation"),
            "depth_crop": perception_dataset_metadata.get("depth_crop"),
            "depth_timing": perception_dataset_metadata.get("depth_timing"),
            "depth_calibration": perception_dataset_metadata.get("depth_calibration"),
            "deployment_depth_preprocessing": perception_dataset_metadata.get("deployment_depth_preprocessing"),
            "self_occlusion": perception_dataset_metadata.get("self_occlusion", False),
            "self_occlusion_contract": perception_dataset_metadata.get("self_occlusion_contract"),
        },
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
        "environment_seed": seed,
        "noise_seed": noise_seed,
        "noise_config": asdict(noise_config),
        "noise_config_hash": noise_config.hash(),
        "initial_state_identical_across_modes": True,
        "diagnostics": diagnostics,
        "modes": {mode: _aggregate([row for row in all_rows if row["mode"] == mode]) for mode in modes},
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
    parser.add_argument(
        "--noise-condition",
        choices=("clean", "measurement", "dropout", "edge", "latency", "extrinsic", "combined"),
        default="clean",
    )
    parser.add_argument("--noise-severity", choices=("mild", "nominal", "strong"), default="nominal")
    parser.add_argument("--noise-seed", type=int, default=271828)
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
        noise_config=depth_noise_preset(
            args.noise_condition,
            args.noise_severity,
            max_depth_m=args.max_range,
        ),
        noise_seed=args.noise_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
