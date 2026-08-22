"""MJLab raycast depth camera with an explicit optical-frame contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

CameraConvention = Literal["optical", "opencv", "ros", "opengl", "mujoco", "world", "flu", "camera_link"]


def wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion without changing the represented rotation."""
    return quaternion[..., (1, 2, 3, 0)]


def xyzw_to_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion without changing the represented rotation."""
    return quaternion[..., (3, 0, 1, 2)]


def quaternion_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compose active rotations in ``xyzw`` order."""
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def rotate_xyzw(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by active ``xyzw`` quaternions."""
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if not torch.isfinite(quaternion).all() or torch.any(norm <= 1.0e-8):
        raise ValueError("quaternion must be finite and have non-zero norm")
    quaternion = quaternion / norm
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    while xyz.ndim < vectors.ndim:
        xyz = xyz.unsqueeze(-2)
        w = w.unsqueeze(-2)
    cross = 2.0 * torch.cross(xyz.expand_as(vectors), vectors, dim=-1)
    return vectors + w * cross + torch.cross(xyz.expand_as(vectors), cross, dim=-1)


def rotation_matrix_to_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """Convert proper rotation matrices to normalized ``xyzw`` quaternions."""
    from mjlab.utils.lab_api.math import quat_from_matrix

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must end in [3, 3], got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    return wxyz_to_xyzw(quat_from_matrix(matrix))


def source_from_optical_rotation(convention: CameraConvention) -> torch.Tensor:
    """Return the rotation from optical axes into a named camera frame.

    Optical/OpenCV and MJLab's ``ros`` camera convention use ``+x right,
    +y down, +z forward``. MuJoCo/OpenGL cameras use ``+x right, +y up,
    -z forward``. MJLab's ``world`` convention, ROS ``camera_link``, and
    the common FLU body frame use ``+x forward, +y left, +z up``.
    """
    if convention in {"optical", "opencv", "ros"}:
        return torch.eye(3, dtype=torch.float64)
    if convention in {"opengl", "mujoco"}:
        return torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))
    if convention in {"world", "flu", "camera_link"}:
        return torch.tensor(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=torch.float64,
        )
    raise ValueError(f"Unsupported camera convention: {convention!r}")


def convert_camera_quaternion_to_optical_xyzw(
    camera_frame_quat_w: torch.Tensor,
    *,
    convention: CameraConvention,
    input_order: Literal["xyzw", "wxyz"] = "xyzw",
) -> torch.Tensor:
    """Convert a camera-frame-to-world rotation into optical-to-world."""
    frame_xyzw = wxyz_to_xyzw(camera_frame_quat_w) if input_order == "wxyz" else camera_frame_quat_w
    frame_from_optical = source_from_optical_rotation(convention).to(
        device=frame_xyzw.device, dtype=frame_xyzw.dtype
    )
    mount_xyzw = rotation_matrix_to_xyzw(frame_from_optical)
    while mount_xyzw.ndim < frame_xyzw.ndim:
        mount_xyzw = mount_xyzw.unsqueeze(0)
    return quaternion_multiply_xyzw(frame_xyzw, mount_xyzw.expand_as(frame_xyzw))


def intrinsic_matrix_from_fov(
    *,
    width: int,
    height: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> torch.Tensor:
    """Build centered pinhole intrinsics using pixel-center coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < horizontal_fov_deg < 180.0 or not 0.0 < vertical_fov_deg < 180.0:
        raise ValueError("camera FOV must lie in (0, 180) degrees")
    fx = (width - 1) / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    fy = (height - 1) / (2.0 * math.tan(math.radians(vertical_fov_deg) / 2.0))
    return torch.tensor(
        [[fx, 0.0, (width - 1) / 2.0], [0.0, fy, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def torso_from_optical_rotation(down_pitch_deg: float) -> torch.Tensor:
    """Return optical-to-FLU torso rotation for a forward/down camera."""
    pitch = math.radians(float(down_pitch_deg))
    right = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)
    down = torch.tensor([-math.sin(pitch), 0.0, -math.cos(pitch)], dtype=torch.float64)
    forward = torch.tensor([math.cos(pitch), 0.0, -math.sin(pitch)], dtype=torch.float64)
    return torch.stack((right, down, forward), dim=-1)


@dataclass(frozen=True)
class DepthCameraConfig:
    """Configuration for the Phase-2A torso-mounted terrain camera."""

    name: str = "pbfm_depth_camera"
    width: int = 64
    height: int = 36
    horizontal_fov_deg: float = 89.0
    vertical_fov_deg: float = 58.0
    intrinsic_matrix: tuple[float, ...] | None = None
    mount_body: str = "torso_link"
    mount_pos_torso: tuple[float, float, float] = (0.10, 0.0, 0.40)
    optical_quat_torso_xyzw: tuple[float, float, float, float] | None = None
    down_pitch_deg: float = 48.0
    min_range: float = 0.10
    max_range: float = 2.50
    include_geom_groups: tuple[int, ...] = (5,)

    def intrinsics(self) -> torch.Tensor:
        if self.intrinsic_matrix is None:
            return intrinsic_matrix_from_fov(
                width=self.width,
                height=self.height,
                horizontal_fov_deg=self.horizontal_fov_deg,
                vertical_fov_deg=self.vertical_fov_deg,
            )
        matrix = torch.tensor(self.intrinsic_matrix, dtype=torch.float64)
        if matrix.numel() != 9:
            raise ValueError("intrinsic_matrix must contain exactly 9 values")
        matrix = matrix.reshape(3, 3)
        if not torch.isfinite(matrix).all() or abs(float(torch.linalg.det(matrix))) <= 1.0e-12:
            raise ValueError("intrinsic_matrix must be finite and invertible")
        return matrix

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if not 0.0 <= self.min_range < self.max_range:
            raise ValueError("range must satisfy 0 <= min_range < max_range")
        if not self.include_geom_groups:
            raise ValueError("at least one terrain geom group is required")
        if not self.mount_body:
            raise ValueError("mount_body must be non-empty")
        if self.optical_quat_torso_xyzw is not None:
            quaternion = torch.tensor(self.optical_quat_torso_xyzw, dtype=torch.float64)
            if not torch.isfinite(quaternion).all() or torch.linalg.vector_norm(quaternion) <= 1.0e-8:
                raise ValueError("optical_quat_torso_xyzw must be finite and non-zero")
        self.intrinsics()

    def torso_from_optical(self) -> torch.Tensor:
        if self.optical_quat_torso_xyzw is None:
            return torso_from_optical_rotation(self.down_pitch_deg)
        quaternion = torch.tensor(self.optical_quat_torso_xyzw, dtype=torch.float64)
        return rotate_xyzw(quaternion, torch.eye(3, dtype=torch.float64)).transpose(-1, -2)


@dataclass
class FullIntrinsicsDepthPatternCfg:
    """Full-K pinhole rays embedded in a torso attachment frame."""

    camera: DepthCameraConfig

    def generate_rays(self, mj_model, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        del mj_model
        self.camera.validate()
        intrinsic = self.camera.intrinsics().to(device=device, dtype=torch.float32)
        rows, columns = torch.meshgrid(
            torch.arange(self.camera.height, device=device, dtype=torch.float32),
            torch.arange(self.camera.width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        pixels = torch.stack((columns, rows, torch.ones_like(columns)), dim=-1)
        optical = pixels @ torch.linalg.inv(intrinsic).transpose(0, 1)
        optical = optical / torch.linalg.vector_norm(optical, dim=-1, keepdim=True)
        torso_from_optical = self.camera.torso_from_optical().to(
            device=device, dtype=torch.float32
        )
        directions = optical.reshape(-1, 3) @ torso_from_optical.transpose(0, 1)
        origin = torch.tensor(self.camera.mount_pos_torso, device=device, dtype=torch.float32)
        offsets = origin.unsqueeze(0).expand(self.camera.width * self.camera.height, 3).clone()
        return offsets, directions


def make_depth_camera_sensor_cfg(camera: DepthCameraConfig, *, torso_body_name: str | None = None):
    """Create an MJLab terrain-only raycast sensor configuration."""
    from mjlab.sensor import ObjRef
    from mjlab.sensor.raycast_sensor import RayCastSensorCfg

    camera.validate()
    return RayCastSensorCfg(
        name=camera.name,
        frame=ObjRef(type="body", name=torso_body_name or camera.mount_body, entity="robot"),
        pattern=FullIntrinsicsDepthPatternCfg(camera),
        ray_alignment="base",
        max_distance=camera.max_range,
        exclude_parent_body=True,
        include_geom_groups=camera.include_geom_groups,
        debug_vis=False,
    )


@dataclass
class DepthCameraFrame:
    """One batched camera sample in the adapter's optical convention."""

    depth_z: torch.Tensor
    camera_pos_w: torch.Tensor
    camera_optical_quat_w: torch.Tensor
    range_image: torch.Tensor
    valid: torch.Tensor


def depth_frame_from_raycast(sensor, camera: DepthCameraConfig) -> DepthCameraFrame:
    """Convert MJLab range rays and torso pose to optical-Z and optical pose."""
    data = sensor.data
    distances = data.distances.reshape(-1, camera.height, camera.width)
    device, dtype = distances.device, distances.dtype
    intrinsic = camera.intrinsics().to(device=device, dtype=dtype)
    rows, columns = torch.meshgrid(
        torch.arange(camera.height, device=device, dtype=dtype),
        torch.arange(camera.width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack((columns, rows, torch.ones_like(columns)), dim=-1)
    optical = pixels @ torch.linalg.inv(intrinsic).transpose(0, 1)
    optical_unit = optical / torch.linalg.vector_norm(optical, dim=-1, keepdim=True)
    valid = torch.isfinite(distances) & (distances >= camera.min_range) & (distances <= camera.max_range)
    depth_z = distances * optical_unit[..., 2]
    depth_z = torch.where(valid, depth_z, torch.full_like(depth_z, float("nan")))

    torso_pos_w = data.frame_pos_w[:, 0]
    torso_quat_w = wxyz_to_xyzw(data.frame_quat_w[:, 0])
    mount_pos = torch.tensor(camera.mount_pos_torso, device=device, dtype=dtype)
    camera_pos_w = torso_pos_w + rotate_xyzw(torso_quat_w, mount_pos.expand_as(torso_pos_w))
    mount_quat = rotation_matrix_to_xyzw(
        camera.torso_from_optical().to(device=device, dtype=dtype)
    ).expand_as(torso_quat_w)
    camera_optical_quat_w = quaternion_multiply_xyzw(torso_quat_w, mount_quat)
    return DepthCameraFrame(
        depth_z=depth_z,
        camera_pos_w=camera_pos_w,
        camera_optical_quat_w=camera_optical_quat_w,
        range_image=torch.where(valid, distances, torch.full_like(distances, float("nan"))),
        valid=valid,
    )
