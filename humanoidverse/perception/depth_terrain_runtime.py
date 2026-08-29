"""Optional clean depth-to-terrain runtime used by inference entrypoints."""

from __future__ import annotations

from pathlib import Path

import torch

from humanoidverse.depth_terrain_evaluation import build_depth_evaluation_env, synchronize_depth_and_gt
from humanoidverse.perception.depth_augmentation import (
    MetricDepthAugmentation,
    MetricDepthAugmentationConfig,
)
from humanoidverse.perception.depth_camera import (
    DepthCameraConfig,
    depth_frame_from_raycast,
    optical_depth_from_raycast,
    rotation_matrix_to_xyzw,
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
from humanoidverse.utils.torch_utils import calc_heading_quat, get_euler_xyz


def load_temporal_perception(path: Path, device: str) -> tuple[TemporalTerrainCompletion, dict]:
    checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError(f"Invalid temporal perception checkpoint: {path}")
    config = checkpoint["config"]
    model = TemporalTerrainCompletion(
        hidden_channels=int(config["hidden_channels"]),
        proprio_dim=int(config["proprio_dim"]),
        proprio_channels=int(config.get("proprio_channels", 8)),
        motion_feature_dim=int(config.get("motion_feature_dim", 6)),
        use_grid_coordinates=bool(config.get("use_grid_coordinates", False)),
        global_context_dim=int(config.get("global_context_dim", 0)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


class TemporalDepthTerrainRuntime:
    """Generate terrain_actor from MJLab depth using the checkpoint's pose contract."""

    def __init__(self, env_config, *, perception_checkpoint: Path, device: str, camera: DepthCameraConfig | None = None):
        self.device = device
        self.perception, self.perception_config = load_temporal_perception(perception_checkpoint, device)
        config = self.perception_config["config"]
        self.terrain_output_mode = resolve_terrain_output_mode(config)
        dataset_metadata = dict(config.get("dataset_metadata", {}))
        if camera is None and config.get("history_mode") == "no_odometry" and dataset_metadata.get("camera"):
            camera = DepthCameraConfig(**dict(dataset_metadata["camera"]))
        self.camera = camera or DepthCameraConfig()
        self.semantic_config = None
        self.scene_camera = None
        if config.get("history_mode") == "no_odometry" and dataset_metadata.get("self_occlusion"):
            semantic_contract = dataset_metadata.get("self_occlusion_contract")
            if not isinstance(semantic_contract, dict):
                raise ValueError("self-occluding checkpoint is missing its semantic contract")
            self.semantic_config = SelfOcclusionDepthConfig.from_metadata(semantic_contract)
            self.camera, self.scene_camera = make_self_occlusion_camera_pair(self.camera, self.semantic_config)
            self.wrapped_env, _ = build_depth_evaluation_env(
                env_config,
                num_envs=1,
                camera=self.camera,
                extra_cameras=(
                    (
                        self.scene_camera,
                        False,
                        self.semantic_config.camera_housing_geom_names,
                        self.semantic_config.camera_housing_mesh_names,
                        self.semantic_config.camera_housing_geom_group,
                    ),
                ),
            )
        else:
            self.wrapped_env, _ = build_depth_evaluation_env(env_config, num_envs=1, camera=self.camera)
        self.core = self.wrapped_env._env
        sequence_steps = int(config["sequence_steps"])
        proprio_dim = int(config["proprio_dim"])
        self.history_mode = str(config.get("history_mode", "egomotion_warp"))
        if self.history_mode == "no_odometry":
            if config.get("dataset_schema") not in (None, "odometry_free_local"):
                raise ValueError("no-odometry checkpoint has an incompatible dataset schema")
            camera_quat_torso = rotation_matrix_to_xyzw(self.camera.torso_from_optical().to(device=device, dtype=torch.float32))
            target_image = dict(dataset_metadata.get("target_image", {}))
            self.target_width = int(target_image.get("width", self.camera.width))
            self.target_height = int(target_image.get("height", self.camera.height))
            self.depth_crop = DepthCropConfig.from_metadata(dataset_metadata.get("depth_crop"))
            target_intrinsics = crop_and_scale_intrinsics(
                self.camera.intrinsics(),
                native_height=self.camera.height,
                native_width=self.camera.width,
                target_height=self.target_height,
                target_width=self.target_width,
                crop=self.depth_crop,
            )
            self.adapter = LocalDepthTerrainAdapter(
                target_intrinsics,
                self.target_height,
                self.target_width,
                camera_pos_torso=self.camera.mount_pos_torso,
                camera_optical_quat_torso_xyzw=tuple(float(value) for value in camera_quat_torso),
            ).to(device)
            self.waist_indices = torch.tensor(
                [self.core.dof_names.index(name) for name in self.core.config.robot.waist_dof_names],
                device=device,
                dtype=torch.long,
            )
            augmentation = dict(dataset_metadata.get("depth_augmentation") or {})
            deployment = dict(dataset_metadata.get("deployment_depth_preprocessing") or {})
            self.depth_gate = MetricDepthAugmentation(
                MetricDepthAugmentationConfig(
                    max_depth_m=float(augmentation.get("max_depth_m", 2.0)),
                    blur_probability=1.0,
                    sigma_min_px=float(deployment.get("blur_sigma_px", 1.5)),
                    sigma_max_px=float(deployment.get("blur_sigma_px", 1.5)),
                )
            )
            self.history = OdometryFreeTerrainHistoryBuffer(
                batch_size=1,
                time_steps=sequence_steps,
                proprio_dim=proprio_dim,
                device=device,
            )
        elif self.history_mode == "egomotion_warp":
            self.adapter = DepthTerrainAdapter(self.camera.intrinsics(), self.camera.height, self.camera.width).to(device)
            self.target_width, self.target_height = self.camera.width, self.camera.height
            self.waist_indices = None
            self.depth_gate = None
            self.depth_crop = None
            self.history = TerrainHistoryBuffer(
                batch_size=1,
                time_steps=sequence_steps,
                proprio_dim=proprio_dim,
                device=device,
            )
        else:
            raise ValueError(f"unsupported perception history mode: {self.history_mode}")
        self.episode_time = torch.zeros(1, device=device)
        self._last_reset = torch.ones(1, device=device, dtype=torch.bool)

    def reset(self) -> None:
        self.history.reset(torch.ones(1, device=self.history.partial_maps.device, dtype=torch.bool))
        self.episode_time.zero_()
        self._last_reset.fill_(True)

    @torch.inference_mode()
    def terrain_actor(self, observation: dict[str, torch.Tensor], *, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        if reset_mask is None:
            reset_mask = self._last_reset
        reset_mask = torch.as_tensor(reset_mask, device=self.device, dtype=torch.bool).reshape(1)
        sensor_names = (
            (self.camera.name, self.scene_camera.name)
            if self.scene_camera is not None
            else self.camera.name
        )
        synchronize_depth_and_gt(self.core, sensor_names)
        sensor = self.core.mjlab_env.scene.sensors[self.camera.name]
        self.history.reset(reset_mask)
        if self.history_mode == "no_odometry":
            assert isinstance(self.adapter, LocalDepthTerrainAdapter)
            assert isinstance(self.history, OdometryFreeTerrainHistoryBuffer)
            assert self.waist_indices is not None and self.depth_gate is not None
            if self.semantic_config is not None:
                assert self.scene_camera is not None
                semantic_frame = self_occluding_depth_from_sensors(
                    sensor,
                    self.core.mjlab_env.scene.sensors[self.scene_camera.name],
                    self.camera,
                    self.semantic_config,
                    self.depth_gate,
                )
                if torch.any(semantic_frame.ambiguous_mask):
                    raise RuntimeError("scene/terrain raycasts produced ambiguous first hits")
                depth_z, resized_self_mask = crop_and_resize_depth_with_conservative_invalid_mask(
                    semantic_frame.final_depth_z,
                    semantic_frame.dilated_self_mask,
                    target_height=self.target_height,
                    target_width=self.target_width,
                    crop=self.depth_crop,
                )
                if torch.any(torch.isfinite(depth_z) & resized_self_mask):
                    raise RuntimeError("conservative self mask was lost before terrain projection")
            else:
                depth_z, _range_image, _ray_valid = optical_depth_from_raycast(sensor, self.camera)
                depth_z, _valid_depth, _sigma = self.depth_gate(depth_z)
                depth_z = crop_and_resize_depth(
                    depth_z,
                    target_height=self.target_height,
                    target_width=self.target_width,
                    crop=self.depth_crop,
                )
            partial_map, visible_mask = self.adapter(
                depth_z,
                self.core.projected_gravity,
                self.core.dof_pos[:, self.waist_indices],
            )
            self.history.append(
                partial_map=partial_map,
                visible_mask=visible_mask,
                timestamp_s=self.episode_time,
                proprio=observation["state"],
            )
            terrain_history = self.history.history(history_seconds=float(self.perception_config["config"]["history_seconds"]))
        else:
            assert isinstance(self.adapter, DepthTerrainAdapter)
            assert isinstance(self.history, TerrainHistoryBuffer)
            frame = depth_frame_from_raycast(sensor, self.camera)
            heading_quat = calc_heading_quat(self.core.base_quat, w_last=True)
            partial_map, visible_mask = self.adapter(
                frame.depth_z,
                frame.camera_pos_w,
                frame.camera_optical_quat_w,
                self.core.robot_root_states[:, :3],
                heading_quat,
            )
            yaw = get_euler_xyz(self.core.base_quat, w_last=True)[2]
            self.history.append(
                partial_map=partial_map,
                visible_mask=visible_mask,
                pelvis_pos_w=self.core.robot_root_states[:, :3],
                heading_yaw_w=yaw,
                timestamp_s=self.episode_time,
                proprio=observation["state"],
            )
            terrain_history = self.history.warp(
                history_seconds=float(self.perception_config["config"]["history_seconds"]),
                interpolation="bilinear",
            )
        terrain_actor = select_terrain_actor_clearance(
            self.perception(terrain_history, proprio=self.history.proprio),
            mode=self.terrain_output_mode,
        )
        if terrain_actor.shape != (1, 273) or not torch.isfinite(terrain_actor).all():
            raise RuntimeError("temporal depth perception produced an invalid terrain_actor")
        self._last_reset.zero_()
        return terrain_actor

    def after_step(self, reset_mask: torch.Tensor) -> None:
        reset_mask = torch.as_tensor(reset_mask, device=self.device, dtype=torch.bool).reshape(1)
        self.episode_time += float(self.core.dt)
        self.episode_time[reset_mask] = 0.0
        self._last_reset = reset_mask

    def close(self) -> None:
        self.wrapped_env.close()
