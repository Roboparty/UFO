"""RealSense depth-to-terrain runtime for hardware bring-up.

The hardware boundary is deliberately small: this module consumes a depth
frame in meters and a synchronized torso/pelvis pose. It does not know about
MuJoCo, Actor weights, or robot control. The output contract is the same
273D ``terrain_actor`` representation used by the simulation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from humanoidverse.perception.depth_augmentation import (
    MetricDepthAugmentation,
    MetricDepthAugmentationConfig,
)
from humanoidverse.perception.depth_camera import quaternion_multiply_xyzw, rotate_xyzw
from humanoidverse.perception.depth_preprocessing import resize_full_fov_depth, scale_full_fov_intrinsics
from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter
from humanoidverse.perception.depth_terrain_runtime import load_temporal_perception
from humanoidverse.perception.temporal_terrain import (
    resolve_terrain_output_mode,
    select_terrain_actor_clearance,
    TerrainHistoryBuffer,
)
from humanoidverse.utils.torch_utils import get_euler_xyz


@dataclass(frozen=True)
class RealSenseCalibration:
    """D435i intrinsics and the calibrated optical-camera-to-torso transform.

    ``intrinsic_matrix`` is the native depth-stream matrix in pixel units. The
    target matrix is derived by scaling the complete native image, preserving
    the full field of view. ``optical_quat_torso_xyzw`` maps optical-camera
    vectors (+x right, +y down, +z forward) into the torso frame.
    """

    native_width: int = 1280
    native_height: int = 720
    target_width: int = 64
    target_height: int = 36
    intrinsic_matrix: tuple[float, ...] = (
        640.0,
        0.0,
        640.0,
        0.0,
        640.0,
        360.0,
        0.0,
        0.0,
        1.0,
    )
    depth_scale_m: float = 0.001
    mount_pos_torso: tuple[float, float, float] = (0.0487988662332928, 0.01, 0.4378029937970051)
    optical_quat_torso_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def validate(self) -> None:
        if min(self.native_width, self.native_height, self.target_width, self.target_height) <= 0:
            raise ValueError("RealSense image dimensions must be positive")
        matrix = self.native_intrinsics()
        if not torch.isfinite(matrix).all() or abs(float(torch.linalg.det(matrix))) <= 1.0e-12:
            raise ValueError("intrinsic_matrix must be finite and invertible")
        if self.depth_scale_m <= 0.0:
            raise ValueError("depth_scale_m must be positive")
        quaternion = torch.tensor(self.optical_quat_torso_xyzw, dtype=torch.float64)
        if not torch.isfinite(quaternion).all() or torch.linalg.vector_norm(quaternion) <= 1.0e-8:
            raise ValueError("optical_quat_torso_xyzw must be finite and non-zero")
        if len(self.mount_pos_torso) != 3:
            raise ValueError("mount_pos_torso must have three components")

    def native_intrinsics(self) -> torch.Tensor:
        values = torch.as_tensor(self.intrinsic_matrix, dtype=torch.float64)
        if values.numel() != 9:
            raise ValueError("intrinsic_matrix must contain nine values")
        return values.reshape(3, 3)

    def target_intrinsics(self) -> torch.Tensor:
        self.validate()
        return scale_full_fov_intrinsics(
            self.native_intrinsics(),
            native_height=self.native_height,
            native_width=self.native_width,
            target_height=self.target_height,
            target_width=self.target_width,
        )

    @classmethod
    def from_json(cls, path: Path) -> "RealSenseCalibration":
        import json

        payload = json.loads(path.expanduser().read_text())
        if not isinstance(payload, dict):
            raise ValueError("calibration JSON must contain an object")
        return cls(**payload)


def depth_to_meters(depth: torch.Tensor, *, depth_scale_m: float) -> torch.Tensor:
    """Convert native D435i depth values to meters and preserve invalid as NaN."""
    if depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be positive")
    depth = torch.as_tensor(depth)
    if not depth.is_floating_point():
        depth = depth.to(torch.float32)
    meters = depth * float(depth_scale_m)
    valid = torch.isfinite(meters) & (meters > 0.0)
    return torch.where(valid, meters, torch.full_like(meters, float("nan")))


@dataclass
class RealSenseTerrainOutput:
    partial_map: torch.Tensor
    visible_mask: torch.Tensor
    terrain_actor: torch.Tensor
    camera_pos_w: torch.Tensor
    camera_optical_quat_w: torch.Tensor
    depth_m: torch.Tensor


class RealSenseDepthRuntime:
    """Project synchronized D435i frames and optionally complete them temporally."""

    def __init__(
        self,
        *,
        calibration: RealSenseCalibration,
        perception_checkpoint: Path | None,
        device: str | torch.device = "cpu",
        batch_size: int = 1,
        depth_augmentation: MetricDepthAugmentationConfig | None = None,
    ) -> None:
        calibration.validate()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.calibration = calibration
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.depth_augmentation = MetricDepthAugmentation(depth_augmentation, seed=0) if depth_augmentation is not None else None
        self.adapter = DepthTerrainAdapter(
            calibration.target_intrinsics(),
            calibration.target_height,
            calibration.target_width,
        ).to(self.device)
        self.perception = None
        self.perception_config: dict = {"config": {"sequence_steps": 1, "history_seconds": 0.6, "proprio_dim": 0}}
        if perception_checkpoint is not None:
            self.perception, self.perception_config = load_temporal_perception(Path(perception_checkpoint), str(self.device))
        config = self.perception_config["config"]
        self.sequence_steps = int(config["sequence_steps"])
        self.history_seconds = float(config["history_seconds"])
        self.proprio_dim = int(config["proprio_dim"])
        self.terrain_output_mode = resolve_terrain_output_mode(config)
        self.history = TerrainHistoryBuffer(
            batch_size=batch_size,
            time_steps=self.sequence_steps,
            proprio_dim=self.proprio_dim,
            device=self.device,
        )

    def reset(self, reset_mask: torch.Tensor | None = None) -> None:
        if reset_mask is None:
            reset_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        self.history.reset(torch.as_tensor(reset_mask, device=self.device, dtype=torch.bool))

    def _camera_pose(
        self,
        torso_pos_w: torch.Tensor,
        torso_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mount_pos = torch.tensor(self.calibration.mount_pos_torso, device=self.device, dtype=torso_pos_w.dtype)
        mount_quat = torch.tensor(self.calibration.optical_quat_torso_xyzw, device=self.device, dtype=torso_quat_w.dtype)
        camera_pos_w = torso_pos_w + rotate_xyzw(torso_quat_w, mount_pos.expand_as(torso_pos_w))
        camera_quat_w = quaternion_multiply_xyzw(torso_quat_w, mount_quat.expand_as(torso_quat_w))
        return camera_pos_w, camera_quat_w

    @torch.inference_mode()
    def step(
        self,
        depth_native: torch.Tensor,
        *,
        torso_pos_w: torch.Tensor,
        torso_quat_w: torch.Tensor,
        pelvis_pos_w: torch.Tensor,
        pelvis_heading_quat_w: torch.Tensor,
        timestamp_s: torch.Tensor,
        proprio: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> RealSenseTerrainOutput:
        """Consume one synchronized frame; poses are world-frame xyzw quaternions."""
        depth_native = torch.as_tensor(depth_native, device=self.device)
        if depth_native.ndim == 2:
            depth_native = depth_native.unsqueeze(0)
        expected = (self.batch_size, self.calibration.native_height, self.calibration.native_width)
        if tuple(depth_native.shape) != expected:
            raise ValueError(f"depth_native must have shape {expected}, got {tuple(depth_native.shape)}")
        tensors = (torso_pos_w, torso_quat_w, pelvis_pos_w, pelvis_heading_quat_w, timestamp_s)
        torso_pos_w, torso_quat_w, pelvis_pos_w, pelvis_heading_quat_w, timestamp_s = tuple(
            torch.as_tensor(value, device=self.device, dtype=torch.float32) for value in tensors
        )
        if torso_pos_w.shape != (self.batch_size, 3) or pelvis_pos_w.shape != (self.batch_size, 3):
            raise ValueError("torso_pos_w and pelvis_pos_w must have shape [B, 3]")
        if torso_quat_w.shape != (self.batch_size, 4) or pelvis_heading_quat_w.shape != (self.batch_size, 4):
            raise ValueError("torso_quat_w and pelvis_heading_quat_w must have shape [B, 4]")
        if timestamp_s.shape != (self.batch_size,):
            raise ValueError("timestamp_s must have shape [B]")
        if reset_mask is None:
            reset_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        reset_mask = torch.as_tensor(reset_mask, device=self.device, dtype=torch.bool).reshape(self.batch_size)
        self.reset(reset_mask)
        depth_m = depth_to_meters(depth_native, depth_scale_m=self.calibration.depth_scale_m)
        if self.depth_augmentation is not None:
            depth_m, _valid_depth, _sigma_px = self.depth_augmentation(depth_m)
        depth_m = resize_full_fov_depth(
            depth_m,
            target_height=self.calibration.target_height,
            target_width=self.calibration.target_width,
        )
        camera_pos_w, camera_quat_w = self._camera_pose(torso_pos_w, torso_quat_w)
        partial_map, visible_mask = self.adapter(
            depth_m,
            camera_pos_w,
            camera_quat_w,
            pelvis_pos_w,
            pelvis_heading_quat_w,
        )
        if proprio is None:
            if self.proprio_dim:
                proprio = torch.zeros((self.batch_size, self.proprio_dim), device=self.device)
        else:
            proprio = torch.as_tensor(proprio, device=self.device, dtype=torch.float32)
        if self.proprio_dim and (proprio is None or proprio.shape != (self.batch_size, self.proprio_dim)):
            raise ValueError(f"proprio must have shape [{self.batch_size}, {self.proprio_dim}]")
        self.history.append(
            partial_map=partial_map,
            visible_mask=visible_mask,
            pelvis_pos_w=pelvis_pos_w,
            heading_yaw_w=get_euler_xyz(pelvis_heading_quat_w, w_last=True)[2],
            timestamp_s=timestamp_s,
            proprio=proprio if proprio is not None else torch.zeros((self.batch_size, 0), device=self.device),
        )
        if self.perception is None:
            terrain_actor = partial_map
        else:
            warped = self.history.warp(history_seconds=self.history_seconds, interpolation="bilinear")
            terrain_actor = select_terrain_actor_clearance(
                self.perception(warped, proprio=self.history.proprio),
                mode=self.terrain_output_mode,
            )
        if self.perception is not None and not torch.isfinite(terrain_actor).all():
            raise RuntimeError("RealSense terrain runtime produced non-finite terrain_actor")
        return RealSenseTerrainOutput(
            partial_map=partial_map,
            visible_mask=visible_mask,
            terrain_actor=terrain_actor,
            camera_pos_w=camera_pos_w,
            camera_optical_quat_w=camera_quat_w,
            depth_m=depth_m,
        )


class RealSenseDepthSource:
    """Optional pyrealsense2 depth source; pose synchronization remains external."""

    def __init__(self, *, width: int = 1280, height: int = 720, fps: int = 30, serial: str | None = None) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("Install pyrealsense2 on the hardware host to use the live D435i source") from error
        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        self.depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())

    def read_depth(self) -> tuple[torch.Tensor, float]:
        frames = self.pipeline.wait_for_frames()
        frame = frames.get_depth_frame()
        if frame is None:
            raise RuntimeError("D435i returned no depth frame")
        import numpy as np

        return torch.from_numpy(np.asanyarray(frame.get_data()).copy()), float(frame.get_timestamp()) / 1000.0

    def close(self) -> None:
        self.pipeline.stop()
