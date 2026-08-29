"""Collect projected terrain-map supervision from a frozen GT-map policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.depth_terrain_evaluation import (
    TERRAIN_NAMES,
    build_depth_evaluation_env,
    synchronize_depth_and_gt,
)
from humanoidverse.mjlab_inference_utils import (
    checkpoint_load_device,
    load_mjlab_env_cfg,
    replace_hydra_override,
)
from humanoidverse.perception.depth_augmentation import (
    CameraFrameScheduler,
    DepthCalibrationAugmentationConfig,
    DepthTimingAugmentationConfig,
    LocalCalibrationAugmentation,
    MetricDepthAugmentation,
    MetricDepthAugmentationConfig,
    deployment_clean_depth_augmentation_config,
    deployment_clean_timing_config,
    phase2i_v1_depth_augmentation_config,
    phase2i_v1_timing_augmentation_config,
    phase2i_v2_depth_augmentation_config,
    phase2i_v2_calibration_augmentation_config,
    phase2i_v2_timing_augmentation_config,
)
from humanoidverse.perception.depth_camera import (
    DepthCameraConfig,
    depth_frame_from_raycast,
    optical_depth_from_raycast,
    rotation_matrix_to_xyzw,
)
from humanoidverse.perception.depth_preprocessing import (
    DEPTH_CROP_CANDIDATES,
    DepthCropConfig,
    crop_and_resize_depth,
    crop_and_resize_depth_with_conservative_invalid_mask,
    crop_and_scale_intrinsics,
    depth_crop_candidate,
)
from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter
from humanoidverse.perception.local_depth_terrain_adapter import LocalDepthTerrainAdapter
from humanoidverse.perception.self_occluding_depth import (
    SelfOcclusionDepthConfig,
    expected_max_dilation_radius,
    make_self_occlusion_camera_pair,
    self_occluding_depth_from_sensors,
)
from humanoidverse.perception.terrain_dataset import (
    OdometryFreeTerrainPerceptionFrameBatch,
    TerrainPerceptionChunkWriter,
    TerrainPerceptionFrameBatch,
)
from humanoidverse.utils.torch_utils import calc_heading_quat, get_euler_xyz


def collect_terrain_perception(
    *,
    model_folder: Path,
    output_dir: Path,
    num_envs: int,
    num_steps: int,
    terrain: str,
    terrain_difficulty: float | None = None,
    terrain_difficulty_range: tuple[float, float] | None = None,
    device: str,
    seed: int,
    chunk_steps: int,
    camera: DepthCameraConfig,
    depth_augmentation: MetricDepthAugmentationConfig | None = None,
    depth_timing: DepthTimingAugmentationConfig | None = None,
    depth_calibration: DepthCalibrationAugmentationConfig | None = None,
    depth_crop: DepthCropConfig | None = None,
    target_width: int = 64,
    target_height: int = 36,
    projection_mode: str = "world_pose",
    self_occlusion: bool = False,
    self_hit_tolerance_m: float = 0.002,
    dilation_sigma_multiplier: float = 3.0,
) -> dict[str, object]:
    """Roll out a frozen policy on GT maps while recording camera-map pairs."""
    if min(num_envs, num_steps, chunk_steps) <= 0:
        raise ValueError("num_envs, num_steps, and chunk_steps must be positive")
    if terrain not in {"mixed", *TERRAIN_NAMES}:
        raise ValueError(f"unknown terrain selection: {terrain!r}")
    if terrain_difficulty is not None and not 0.0 <= terrain_difficulty <= 1.0:
        raise ValueError("terrain_difficulty must lie in [0, 1]")
    if terrain_difficulty_range is not None:
        low, high = terrain_difficulty_range
        if terrain_difficulty is not None:
            raise ValueError("fixed terrain difficulty and difficulty range are mutually exclusive")
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("terrain_difficulty_range must satisfy 0 <= low <= high <= 1")
    if projection_mode not in {"world_pose", "local_no_odometry"}:
        raise ValueError("projection_mode must be 'world_pose' or 'local_no_odometry'")
    if self_occlusion and projection_mode != "local_no_odometry":
        raise ValueError("self-occluding collection requires projection_mode='local_no_odometry'")
    if depth_calibration is not None and projection_mode != "local_no_odometry":
        raise ValueError("calibration DR is only defined for local_no_odometry projection")
    if self_occlusion:
        sigma_contract = depth_augmentation is not None and (
            (
                depth_augmentation.sigma_min_px == 0.0
                and depth_augmentation.sigma_max_px == 3.0
            )
            or (
                depth_augmentation.sigma_min_px == 1.5
                and depth_augmentation.sigma_max_px == 1.5
            )
        )
        expected_augmentation = depth_augmentation is not None and (
            depth_augmentation.max_depth_m == 2.0
            and depth_augmentation.blur_probability == 1.0
            and sigma_contract
        )
        if not expected_augmentation:
            raise ValueError(
                "self-occluding collection requires 2m, blur=100%, and either training U(0,3px) or deployment 1.5px"
            )
    depth_crop = depth_crop or DepthCropConfig()
    depth_crop.validate()

    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    model = load_model_from_checkpoint_dir(
        checkpoint_dir,
        device=checkpoint_load_device(device),
    )
    model.to(device)
    model.eval()

    env_config, _ = load_mjlab_env_cfg(
        model_folder,
        data_path=None,
        robot_config=None,
        device=device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=10_000.0,
    )
    updates: dict[str, object] = {"seed": seed}
    if terrain != "mixed":
        updates["hydra_overrides"] = replace_hydra_override(
            list(env_config.hydra_overrides),
            "terrain.terrain_type",
            terrain,
        )
    if terrain_difficulty is not None:
        updates["hydra_overrides"] = replace_hydra_override(
            list(updates.get("hydra_overrides", env_config.hydra_overrides)),
            "terrain.difficulty_range",
            f"[{terrain_difficulty},{terrain_difficulty}]",
        )
    elif terrain_difficulty_range is not None:
        low, high = terrain_difficulty_range
        updates["hydra_overrides"] = replace_hydra_override(
            list(updates.get("hydra_overrides", env_config.hydra_overrides)),
            "terrain.difficulty_range",
            f"[{low},{high}]",
        )
    env_config = env_config.model_copy(update=updates)
    semantic_config = None
    scene_camera = None
    active_camera = camera
    if self_occlusion:
        semantic_config = SelfOcclusionDepthConfig(
            min_ray_range_m=0.10,
            max_ray_range_m=2.0,
            hit_tolerance_m=self_hit_tolerance_m,
            dilation_sigma_multiplier=dilation_sigma_multiplier,
        )
        active_camera, scene_camera = make_self_occlusion_camera_pair(camera, semantic_config)
        wrapped_env, _ = build_depth_evaluation_env(
            env_config,
            num_envs=num_envs,
            camera=active_camera,
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
    if min(target_width, target_height) <= 0:
        raise ValueError("target image dimensions must be positive")
    native_intrinsics = active_camera.intrinsics()
    target_intrinsics = crop_and_scale_intrinsics(
        native_intrinsics,
        native_height=active_camera.height,
        native_width=active_camera.width,
        target_height=target_height,
        target_width=target_width,
        crop=depth_crop,
    )
    if projection_mode == "local_no_odometry":
        camera_quat_torso = rotation_matrix_to_xyzw(active_camera.torso_from_optical().to(device=device, dtype=torch.float32))
        adapter = LocalDepthTerrainAdapter(
            target_intrinsics,
            target_height,
            target_width,
            camera_pos_torso=active_camera.mount_pos_torso,
            camera_optical_quat_torso_xyzw=tuple(float(value) for value in camera_quat_torso),
        ).to(device)
        waist_indices = torch.tensor(
            [core.dof_names.index(name) for name in core.config.robot.waist_dof_names],
            device=device,
            dtype=torch.long,
        )
        calibration_augmentation = LocalCalibrationAugmentation(
            depth_calibration or DepthCalibrationAugmentationConfig(),
            intrinsic_matrix=target_intrinsics,
            camera_pos_torso=active_camera.mount_pos_torso,
            camera_optical_quat_torso_xyzw=tuple(float(value) for value in camera_quat_torso),
            batch_size=num_envs,
            device=device,
            seed=seed + 41_009,
        )
    else:
        adapter = DepthTerrainAdapter(target_intrinsics, target_height, target_width).to(device)
        waist_indices = None
        calibration_augmentation = None
    augmentation = MetricDepthAugmentation(depth_augmentation, seed=seed + 17_003) if depth_augmentation is not None else None
    control_frequency_hz = 1.0 / float(core.dt)
    if depth_timing is None:
        depth_timing = DepthTimingAugmentationConfig(
            camera_frequency_hz=control_frequency_hz,
            control_frequency_hz=control_frequency_hz,
        )
    depth_timing.validate()
    if abs(depth_timing.control_frequency_hz - control_frequency_hz) > 1.0e-3:
        raise ValueError(
            f"timing DR expects {depth_timing.control_frequency_hz:g} Hz control, environment uses {control_frequency_hz:g} Hz"
        )
    frame_scheduler = CameraFrameScheduler(
        depth_timing,
        batch_size=num_envs,
        device=device,
        seed=seed + 31_337,
    )
    episode_id = torch.zeros(num_envs, device=device, dtype=torch.long)
    episode_time = torch.zeros(num_envs, device=device)
    env_id = torch.arange(num_envs, device=device, dtype=torch.long)
    latent = model.sample_z(num_envs, device=device)
    visible_sum = 0
    frame_count = 0
    valid_camera_frames = 0
    timing_counts = {"scheduled": 0, "dropped": 0, "duplicated": 0}
    semantic_counts = {"terrain": 0, "self": 0, "dilated_self": 0, "far_or_no_hit": 0, "ambiguous": 0}

    metadata = {
        "model_folder": str(model_folder),
        "checkpoint_dir": str(checkpoint_dir),
        "terrain": terrain,
        "terrain_difficulty": terrain_difficulty,
        "terrain_difficulty_range": terrain_difficulty_range,
        "seed": seed,
        "control_dt_s": core.dt,
        "camera": asdict(active_camera),
        "target_image": {"width": target_width, "height": target_height},
        "depth_crop": depth_crop.to_metadata(),
        "terrain_component_names": list(core.terrain_component_names),
        "policy_terrain_input": "GT terrain_actor",
        "stored_depth": False,
        "depth_augmentation": None if depth_augmentation is None else asdict(depth_augmentation),
        "depth_timing": asdict(depth_timing),
        "depth_calibration": (
            None if depth_calibration is None else asdict(depth_calibration)
        ),
        "deployment_depth_preprocessing": {
            "max_depth_m": 2.0,
            "blur_sigma_px": 1.5,
            "artificial_random_dr": False,
        },
        "projection_mode": projection_mode,
        "deployment_inputs": (
            ["depth_z_m", "projected_gravity", "waist_joint_pos", "fixed_camera_extrinsic"]
            if projection_mode == "local_no_odometry"
            else ["depth_z_m", "camera_pose_w", "pelvis_pose_w"]
        ),
        "uses_odometry": projection_mode != "local_no_odometry",
        "self_occlusion": self_occlusion,
        "self_occlusion_contract": (
            None
            if semantic_config is None
            else {
                **asdict(semantic_config),
                "scene_camera": asdict(scene_camera),
                "scene_exclude_parent_body": False,
                "max_dilation_radius_px": expected_max_dilation_radius(
                    semantic_config,
                    depth_augmentation.sigma_max_px,
                ),
                "far_depth_is_geometry_valid": False,
                "self_depth_is_geometry_valid": False,
                "range_definition": "euclidean distance along normalized camera ray",
                "adapter_depth_definition": "optical-axis Z in meters",
            }
        ),
    }
    writer = TerrainPerceptionChunkWriter(
        output_dir,
        chunk_steps=chunk_steps,
        metadata=metadata,
        odometry_free=projection_mode == "local_no_odometry",
    )
    try:
        observation, _ = wrapped_env.reset(to_numpy=False)
        for _step in range(num_steps):
            frame_valid, frame_timestamp, timing_diagnostics = frame_scheduler.step(episode_time)
            for name in timing_counts:
                timing_counts[name] += int(timing_diagnostics[name].sum().item())
            sensor_names = (
                (active_camera.name, scene_camera.name)
                if scene_camera is not None
                else active_camera.name
            )
            synchronize_depth_and_gt(core, sensor_names)
            sensor = core.mjlab_env.scene.sensors[active_camera.name]
            resized_self_mask = None
            if self_occlusion:
                assert semantic_config is not None and scene_camera is not None and augmentation is not None
                semantic_frame = self_occluding_depth_from_sensors(
                    sensor,
                    core.mjlab_env.scene.sensors[scene_camera.name],
                    active_camera,
                    semantic_config,
                    augmentation,
                )
                if torch.any(semantic_frame.ambiguous_mask):
                    terrain_ranges = sensor.data.distances.reshape(-1, active_camera.height, active_camera.width)
                    scene_ranges = core.mjlab_env.scene.sensors[scene_camera.name].data.distances.reshape(
                        -1,
                        active_camera.height,
                        active_camera.width,
                    )
                    differences = (scene_ranges - terrain_ranges)[semantic_frame.ambiguous_mask]
                    ambiguous_scene = scene_ranges[semantic_frame.ambiguous_mask]
                    quantiles = torch.quantile(
                        ambiguous_scene,
                        torch.tensor((0.0, 0.5, 0.95, 1.0), device=ambiguous_scene.device),
                    )
                    hit_geom_summary = "unavailable"
                    scene_sensor = core.mjlab_env.scene.sensors[scene_camera.name]
                    if scene_sensor._ray_geomid is not None:  # noqa: SLF001 - failure-only MJLab diagnostic
                        import warp as wp

                        geom_ids = wp.to_torch(scene_sensor._ray_geomid).reshape_as(scene_ranges)[  # noqa: SLF001
                            semantic_frame.ambiguous_mask
                        ]
                        unique_ids, counts = torch.unique(geom_ids, return_counts=True)
                        ranked = torch.argsort(counts, descending=True)[:8]
                        entries = []
                        for index in ranked.tolist():
                            geom_id = int(unique_ids[index].item())
                            geom_name = scene_sensor._mj_model.geom(geom_id).name  # noqa: SLF001
                            entries.append(f"{geom_name}:{int(counts[index].item())}")
                        hit_geom_summary = ",".join(entries)
                    raise RuntimeError(
                        "scene/terrain raycasts produced inconsistent hits outside the configured tolerance: "
                        f"count={differences.numel()}, min_delta={float(differences.min().item()):.6f}, "
                        f"max_delta={float(differences.max().item()):.6f}, "
                        f"terrain_hit_scene_miss={int(torch.count_nonzero(semantic_frame.ambiguous_mask & (scene_ranges < 0)).item())}, "
                        f"scene_range_quantiles={quantiles.tolist()}, hit_geoms={hit_geom_summary}"
                    )
                depth_z, resized_self_mask = crop_and_resize_depth_with_conservative_invalid_mask(
                    semantic_frame.final_depth_z,
                    semantic_frame.dilated_self_mask,
                    target_height=target_height,
                    target_width=target_width,
                    crop=depth_crop,
                )
                semantic_counts["terrain"] += int(semantic_frame.terrain_mask.sum().item())
                semantic_counts["self"] += int(semantic_frame.self_mask.sum().item())
                semantic_counts["dilated_self"] += int(semantic_frame.dilated_self_mask.sum().item())
                semantic_counts["far_or_no_hit"] += int(semantic_frame.far_or_no_hit_mask.sum().item())
                semantic_counts["ambiguous"] += int(semantic_frame.ambiguous_mask.sum().item())
                frame = None
            elif projection_mode == "local_no_odometry":
                depth_z, _range_image, _ray_valid = optical_depth_from_raycast(sensor, active_camera)
                frame = None
            else:
                frame = depth_frame_from_raycast(sensor, active_camera)
                depth_z = frame.depth_z
            if augmentation is not None and not self_occlusion:
                depth_z, _valid_depth, _sigma_px = augmentation(depth_z)
            if not self_occlusion:
                depth_z = crop_and_resize_depth(
                    depth_z,
                    target_height=target_height,
                    target_width=target_width,
                    crop=depth_crop,
                )
            if projection_mode == "local_no_odometry":
                assert isinstance(adapter, LocalDepthTerrainAdapter)
                assert waist_indices is not None and calibration_augmentation is not None
                partial_map, visible_mask = adapter(
                    depth_z,
                    core.projected_gravity,
                    core.dof_pos[:, waist_indices],
                    intrinsic_matrix=calibration_augmentation.intrinsics,
                    camera_pos_torso=calibration_augmentation.camera_pos_torso,
                    camera_optical_quat_torso_xyzw=calibration_augmentation.camera_quat_torso,
                )
                if resized_self_mask is not None and torch.any(torch.isfinite(depth_z) & resized_self_mask):
                    raise RuntimeError("conservative self mask was lost before local terrain projection")
            else:
                assert isinstance(adapter, DepthTerrainAdapter)
                assert frame is not None
                heading_quat = calc_heading_quat(core.base_quat, w_last=True)
                partial_map, visible_mask = adapter(
                    depth_z,
                    frame.camera_pos_w,
                    frame.camera_optical_quat_w,
                    core.robot_root_states[:, :3],
                    heading_quat,
                )
            visible_mask = visible_mask & frame_valid.unsqueeze(1)
            partial_map = torch.where(
                visible_mask,
                partial_map,
                torch.full_like(partial_map, float("nan")),
            )
            gt = core._terrain_actor_obs().clone()
            common_frame = {
                "partial_map": partial_map,
                "visible_mask": visible_mask,
                "timestamp_s": frame_timestamp,
                "proprio": observation["state"],
                "gt_terrain_actor": gt,
                "episode_id": episode_id,
                "env_id": env_id,
                "terrain_type": core._current_terrain_type_ids(),
            }
            if projection_mode == "local_no_odometry":
                writer.append(
                    OdometryFreeTerrainPerceptionFrameBatch(
                        **common_frame,
                        frame_valid=frame_valid,
                    )
                )
            else:
                yaw = get_euler_xyz(core.base_quat, w_last=True)[2]
                writer.append(
                    TerrainPerceptionFrameBatch(
                        **common_frame,
                        pelvis_pos_w=core.robot_root_states[:, :3],
                        heading_yaw_w=yaw,
                    )
                )
            visible_sum += int(visible_mask.sum().item())
            valid_camera_frames += int(frame_valid.sum().item())
            frame_count += int(frame_valid.sum().item()) * visible_mask.shape[1]

            with torch.no_grad():
                action = model.act(observation, latent, mean=True)
            observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
            reset = (
                torch.as_tensor(terminated, device=device).bool()
                | torch.as_tensor(
                    truncated,
                    device=device,
                ).bool()
            )
            episode_time += core.dt
            if torch.any(reset):
                episode_id[reset] += 1
                episode_time[reset] = 0.0
                latent[reset] = model.sample_z(int(reset.sum().item()), device=device)
                frame_scheduler.reset(reset)
                if calibration_augmentation is not None:
                    calibration_augmentation.reset(reset)
    finally:
        writer.close()
        wrapped_env.close()

    summary = {
        **metadata,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "frames": num_envs * num_steps,
        "visible_fraction": visible_sum / frame_count if frame_count else 0.0,
        "valid_camera_fraction": valid_camera_frames / (num_envs * num_steps),
        "timing_event_fractions": {
            name: count / (num_envs * num_steps) for name, count in timing_counts.items()
        },
        "history_target_s": 0.6,
        "semantic_pixel_fractions": (
            None
            if not self_occlusion or not frame_count
            else {name: count / (num_envs * num_steps * active_camera.height * active_camera.width) for name, count in semantic_counts.items()}
        ),
    }
    (output_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--chunk-steps", type=int, default=128)
    parser.add_argument("--terrain", choices=("mixed", *TERRAIN_NAMES), default="mixed")
    parser.add_argument(
        "--terrain-difficulty",
        type=float,
        default=None,
        help="Fix normalized terrain difficulty; stairs 0.75 corresponds to 16 cm.",
    )
    parser.add_argument(
        "--terrain-difficulty-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Sample normalized terrain difficulty from a closed interval.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--target-width", type=int, default=64)
    parser.add_argument("--target-height", type=int, default=36)
    parser.add_argument("--horizontal-fov", type=float, default=89.0)
    parser.add_argument("--vertical-fov", type=float, default=58.0)
    parser.add_argument("--down-pitch", type=float, default=48.0)
    parser.add_argument("--min-range", type=float, default=0.10)
    parser.add_argument("--max-range", type=float, default=2.50)
    parser.add_argument("--depth-gate-max", type=float, default=None, help="Invalidate metric depth above this value")
    parser.add_argument("--blur-probability", type=float, default=0.0)
    parser.add_argument("--blur-sigma-min-px", type=float, default=0.0)
    parser.add_argument("--blur-sigma-max-px", type=float, default=0.0)
    parser.add_argument(
        "--projection-mode",
        choices=("world_pose", "local_no_odometry"),
        default="world_pose",
    )
    parser.add_argument("--self-occlusion", action="store_true")
    parser.add_argument("--self-hit-tolerance-m", type=float, default=0.002)
    parser.add_argument("--dilation-sigma-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--depth-crop",
        choices=tuple(DEPTH_CROP_CANDIDATES),
        default="full",
        help="Crop in 64x36-equivalent pixels before raw-frame downsampling.",
    )
    parser.add_argument(
        "--depth-dr-preset",
        choices=("none", "phase2i_v1", "deployment_clean", "phase2i_v2"),
        default="none",
    )
    parser.add_argument(
        "--timing-dr-preset",
        choices=("none", "phase2i_v1", "deployment_clean", "phase2i_v2"),
        default="none",
    )
    parser.add_argument(
        "--calibration-dr-preset",
        choices=("none", "phase2i_v2"),
        default="none",
    )
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
    depth_augmentation = None
    if args.depth_dr_preset == "phase2i_v2":
        depth_augmentation = phase2i_v2_depth_augmentation_config()
    elif args.depth_dr_preset == "phase2i_v1":
        depth_augmentation = phase2i_v1_depth_augmentation_config()
    elif args.depth_dr_preset == "deployment_clean":
        depth_augmentation = deployment_clean_depth_augmentation_config()
    elif args.depth_gate_max is not None or args.blur_probability > 0.0 or args.blur_sigma_max_px > 0.0:
        depth_augmentation = MetricDepthAugmentationConfig(
            max_depth_m=2.0 if args.depth_gate_max is None else args.depth_gate_max,
            blur_probability=args.blur_probability,
            sigma_min_px=args.blur_sigma_min_px,
            sigma_max_px=args.blur_sigma_max_px,
        )
    depth_timing = (
        phase2i_v2_timing_augmentation_config()
        if args.timing_dr_preset == "phase2i_v2"
        else (
            phase2i_v1_timing_augmentation_config()
            if args.timing_dr_preset == "phase2i_v1"
            else (
                deployment_clean_timing_config()
                if args.timing_dr_preset == "deployment_clean"
                else None
            )
        )
    )
    depth_calibration = (
        phase2i_v2_calibration_augmentation_config()
        if args.calibration_dr_preset == "phase2i_v2"
        else None
    )
    result = collect_terrain_perception(
        model_folder=args.model_folder,
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        chunk_steps=args.chunk_steps,
        terrain=args.terrain,
        terrain_difficulty=args.terrain_difficulty,
        terrain_difficulty_range=(
            tuple(args.terrain_difficulty_range)
            if args.terrain_difficulty_range is not None
            else None
        ),
        device=args.device,
        seed=args.seed,
        camera=camera,
        depth_augmentation=depth_augmentation,
        depth_timing=depth_timing,
        depth_calibration=depth_calibration,
        depth_crop=depth_crop_candidate(args.depth_crop),
        target_width=args.target_width,
        target_height=args.target_height,
        projection_mode=args.projection_mode,
        self_occlusion=args.self_occlusion,
        self_hit_tolerance_m=args.self_hit_tolerance_m,
        dilation_sigma_multiplier=args.dilation_sigma_multiplier,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
