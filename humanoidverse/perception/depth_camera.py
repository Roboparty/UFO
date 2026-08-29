"""MJLab raycast depth camera with an explicit optical-frame contract."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import warp as wp
from mjlab.sensor.raycast_sensor import RayCastSensor, RayCastSensorCfg

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
    frame_from_optical = source_from_optical_rotation(convention).to(device=frame_xyzw.device, dtype=frame_xyzw.dtype)
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
    """Build edge-defined pinhole intrinsics for pixel-center sampling.

    This is the same contract used by InstinctMJ's grouped camera: the
    principal point is at ``(width / 2, height / 2)`` and integer pixel
    indices are shifted by ``0.5`` before applying ``K^-1``.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < horizontal_fov_deg < 180.0 or not 0.0 < vertical_fov_deg < 180.0:
        raise ValueError("camera FOV must lie in (0, 180) degrees")
    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vertical_fov_deg) / 2.0))
    return torch.tensor(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def optical_rays_from_intrinsics(
    camera: "DepthCameraConfig",
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    normalize: bool = True,
) -> torch.Tensor:
    """Return raster-ordered optical rays using the canonical pixel centers."""
    intrinsic = camera.intrinsics().to(device=device, dtype=dtype)
    rows, columns = torch.meshgrid(
        torch.arange(camera.height, device=device, dtype=dtype) + 0.5,
        torch.arange(camera.width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((columns, rows, torch.ones_like(columns)), dim=-1)
    optical = pixels @ torch.linalg.inv(intrinsic).transpose(0, 1)
    if normalize:
        optical = optical / torch.linalg.vector_norm(optical, dim=-1, keepdim=True)
    return optical


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
    mount_pos_torso: tuple[float, float, float] = (
        0.0487988662332928,
        0.01,
        0.4378029937970051,
    )
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
    ray_origin_shift: float = 0.0

    def generate_rays(self, mj_model, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        del mj_model
        self.camera.validate()
        optical = optical_rays_from_intrinsics(
            self.camera,
            device=device,
            dtype=torch.float32,
        )
        torso_from_optical = self.camera.torso_from_optical().to(device=device, dtype=torch.float32)
        directions = optical.reshape(-1, 3) @ torso_from_optical.transpose(0, 1)
        origin = torch.tensor(self.camera.mount_pos_torso, device=device, dtype=torch.float32)
        offsets = origin.unsqueeze(0).expand(self.camera.width * self.camera.height, 3).clone()
        if self.ray_origin_shift:
            offsets += directions * float(self.ray_origin_shift)
        return offsets, directions


def _euler_xyz_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert batched XYZ Euler angles to rotation matrices."""
    roll, pitch, yaw = euler.unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ),
        dim=-1,
    ).reshape(*euler.shape[:-1], 3, 3)


class InstallationRandomizedRayCastSensor(RayCastSensor):
    """Raycast camera with a fixed installation error for each episode."""

    cfg: "InstallationRandomizedRayCastSensorCfg"

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        num_envs = int(data.nworld)
        self._position_delta = torch.zeros((num_envs, 3), device=device)
        self._rotation_delta = torch.eye(3, device=device).expand(num_envs, 3, 3).clone()
        self._mount_rotation = torch.tensor(
            self.cfg.mount_rotation_torso,
            device=device,
            dtype=self._position_delta.dtype,
        ).reshape(3, 3)
        self.randomize_installation(torch.arange(num_envs, device=device))

    def randomize_installation(self, env_ids: torch.Tensor) -> None:
        """Sample one camera mounting error per selected environment."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._position_delta.device).reshape(-1)
        if env_ids.numel() == 0:
            return
        if self.cfg.position_error > 0.0:
            self._position_delta[env_ids] = torch.empty(
                (env_ids.numel(), 3), device=self._position_delta.device
            ).uniform_(-self.cfg.position_error, self.cfg.position_error)
        else:
            self._position_delta[env_ids] = 0.0
        if self.cfg.angle_error_rad > 0.0:
            angle_delta = torch.empty(
                (env_ids.numel(), 3), device=self._position_delta.device
            ).uniform_(-self.cfg.angle_error_rad, self.cfg.angle_error_rad)
            self._rotation_delta[env_ids] = _euler_xyz_matrix(angle_delta)
        else:
            self._rotation_delta[env_ids] = torch.eye(
                3, device=self._position_delta.device, dtype=self._position_delta.dtype
            )

    def prepare_rays(self) -> None:
        super().prepare_rays()
        if self.cfg.position_error == 0.0 and self.cfg.angle_error_rad == 0.0:
            return

        assert self._cached_world_origins is not None
        assert self._cached_world_rays is not None
        assert self._cached_frame_mat is not None
        assert self._ray_pnt is not None and self._ray_vec is not None

        frame_mat = self._cached_frame_mat
        aligned_parent = self._compute_alignment_rotation(frame_mat.reshape(-1, 3, 3)).reshape_as(frame_mat)
        camera_mat = aligned_parent @ self._mount_rotation
        delta_world = camera_mat @ self._rotation_delta[:, None] @ camera_mat.transpose(-1, -2)
        translated = torch.einsum("bfij,bj->bfi", camera_mat, self._position_delta)

        batch, frames = frame_mat.shape[:2]
        rays_per_frame = self.num_rays_per_frame
        origins = self._cached_world_origins.view(batch, frames, rays_per_frame, 3)
        rays = self._cached_world_rays.view(batch, frames, rays_per_frame, 3)
        self._cached_world_origins = (origins + translated[:, :, None, :]).reshape(batch, -1, 3)
        self._cached_world_rays = torch.einsum("bfij,bfnj->bfni", delta_world, rays).reshape(batch, -1, 3)

        wp.to_torch(self._ray_pnt).view_as(self._cached_world_origins).copy_(self._cached_world_origins)
        wp.to_torch(self._ray_vec).view_as(self._cached_world_rays).copy_(self._cached_world_rays)


@dataclass
class InstallationRandomizedRayCastSensorCfg(RayCastSensorCfg):
    """Configuration for per-episode depth-camera extrinsic randomization."""

    position_error: float = 0.0
    angle_error_rad: float = 0.0
    mount_rotation_torso: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def build(self) -> InstallationRandomizedRayCastSensor:
        if self.position_error < 0.0 or self.angle_error_rad < 0.0:
            raise ValueError("camera installation error bounds must be non-negative")
        mount_rotation = torch.tensor(self.mount_rotation_torso, dtype=torch.float64)
        if mount_rotation.numel() != 9 or not torch.isfinite(mount_rotation).all():
            raise ValueError("mount_rotation_torso must contain nine finite values")
        return InstallationRandomizedRayCastSensor(self)


class CameraHousingAwareRayCastSensor(RayCastSensor):
    """Ray sensor that removes only the simulated shell containing its origin."""

    cfg: "CameraHousingAwareRayCastSensorCfg"

    def edit_spec(self, scene_spec, entities) -> None:
        del entities
        matched_geom_names: set[str] = set()
        matched_mesh_names: set[str] = set()
        for geom in scene_spec.geoms:
            geom_name = str(geom.name)
            mesh_name = str(geom.meshname)
            for suffix in self.cfg.excluded_geom_name_suffixes:
                if geom_name == suffix or geom_name.endswith(f"/{suffix}"):
                    geom.group = self.cfg.excluded_geom_group
                    matched_geom_names.add(suffix)
            for suffix in self.cfg.excluded_mesh_name_suffixes:
                if mesh_name == suffix or mesh_name.endswith(f"/{suffix}"):
                    geom.group = self.cfg.excluded_geom_group
                    matched_mesh_names.add(suffix)
        missing_geoms = set(self.cfg.excluded_geom_name_suffixes) - matched_geom_names
        missing_meshes = set(self.cfg.excluded_mesh_name_suffixes) - matched_mesh_names
        if missing_geoms or missing_meshes:
            raise RuntimeError(
                "camera-housing geom exclusions did not resolve in the MJLab scene: "
                f"missing_geom_names={sorted(missing_geoms)}, missing_mesh_names={sorted(missing_meshes)}"
            )


@dataclass
class CameraHousingAwareRayCastSensorCfg(RayCastSensorCfg):
    """Raycast configuration with explicit camera-housing-only exclusions."""

    excluded_geom_name_suffixes: tuple[str, ...] = field(default_factory=tuple)
    excluded_mesh_name_suffixes: tuple[str, ...] = field(default_factory=tuple)
    excluded_geom_group: int = 4

    def build(self) -> CameraHousingAwareRayCastSensor:
        if self.include_geom_groups is None or self.excluded_geom_group in self.include_geom_groups:
            raise ValueError("excluded camera-housing group must not be visible to this raycast")
        return CameraHousingAwareRayCastSensor(self)


def make_depth_camera_sensor_cfg(
    camera: DepthCameraConfig,
    *,
    torso_body_name: str | None = None,
    exclude_parent_body: bool = True,
    ray_alignment: Literal["base", "yaw", "world"] = "base",
    ray_origin_shift: float = 0.0,
    position_error: float = 0.0,
    angle_error_rad: float = 0.0,
    excluded_geom_name_suffixes: tuple[str, ...] = (),
    excluded_mesh_name_suffixes: tuple[str, ...] = (),
    excluded_geom_group: int = 4,
):
    """Create an MJLab terrain-only raycast sensor configuration."""
    from mjlab.sensor import ObjRef
    camera.validate()
    installation_randomization = position_error > 0.0 or angle_error_rad > 0.0
    housing_exclusions = bool(excluded_geom_name_suffixes or excluded_mesh_name_suffixes)
    if installation_randomization and housing_exclusions:
        raise ValueError("camera installation randomization cannot currently be combined with housing exclusions")
    if installation_randomization:
        cfg_type = InstallationRandomizedRayCastSensorCfg
    elif housing_exclusions:
        cfg_type = CameraHousingAwareRayCastSensorCfg
    else:
        cfg_type = RayCastSensorCfg
    extra = {}
    if cfg_type is CameraHousingAwareRayCastSensorCfg:
        extra = {
            "excluded_geom_name_suffixes": excluded_geom_name_suffixes,
            "excluded_mesh_name_suffixes": excluded_mesh_name_suffixes,
            "excluded_geom_group": excluded_geom_group,
        }
    elif cfg_type is InstallationRandomizedRayCastSensorCfg:
        mount_rotation = camera.torso_from_optical().reshape(-1)
        extra = {
            "position_error": float(position_error),
            "angle_error_rad": float(angle_error_rad),
            "mount_rotation_torso": tuple(float(value) for value in mount_rotation),
        }
    return cfg_type(
        name=camera.name,
        frame=ObjRef(type="body", name=torso_body_name or camera.mount_body, entity="robot"),
        pattern=FullIntrinsicsDepthPatternCfg(camera, ray_origin_shift=float(ray_origin_shift)),
        ray_alignment=ray_alignment,
        max_distance=camera.max_range - float(ray_origin_shift),
        exclude_parent_body=exclude_parent_body,
        include_geom_groups=camera.include_geom_groups,
        debug_vis=False,
        **extra,
    )


@dataclass
class DepthCameraFrame:
    """One batched camera sample in the adapter's optical convention."""

    depth_z: torch.Tensor
    camera_pos_w: torch.Tensor
    camera_optical_quat_w: torch.Tensor
    range_image: torch.Tensor
    valid: torch.Tensor


def optical_depth_from_raycast(sensor, camera: DepthCameraConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return optical-Z, range, and validity without reading any world pose."""
    data = sensor.data
    distances = data.distances.reshape(-1, camera.height, camera.width)
    device, dtype = distances.device, distances.dtype
    optical_unit = optical_rays_from_intrinsics(camera, device=device, dtype=dtype)
    valid = torch.isfinite(distances) & (distances >= camera.min_range) & (distances <= camera.max_range)
    depth_z = distances * optical_unit[..., 2]
    depth_z = torch.where(valid, depth_z, torch.full_like(depth_z, float("nan")))
    range_image = torch.where(valid, distances, torch.full_like(distances, float("nan")))
    return depth_z, range_image, valid


def depth_frame_from_raycast(sensor, camera: DepthCameraConfig) -> DepthCameraFrame:
    """Convert MJLab range rays and torso pose to optical-Z and optical pose."""
    depth_z, range_image, valid = optical_depth_from_raycast(sensor, camera)
    data = sensor.data
    device, dtype = depth_z.device, depth_z.dtype

    torso_pos_w = data.frame_pos_w[:, 0]
    torso_quat_w = wxyz_to_xyzw(data.frame_quat_w[:, 0])
    mount_pos = torch.tensor(camera.mount_pos_torso, device=device, dtype=dtype)
    camera_pos_w = torso_pos_w + rotate_xyzw(torso_quat_w, mount_pos.expand_as(torso_pos_w))
    mount_quat = rotation_matrix_to_xyzw(camera.torso_from_optical().to(device=device, dtype=dtype)).expand_as(torso_quat_w)
    camera_optical_quat_w = quaternion_multiply_xyzw(torso_quat_w, mount_quat)
    return DepthCameraFrame(
        depth_z=depth_z,
        camera_pos_w=camera_pos_w,
        camera_optical_quat_w=camera_optical_quat_w,
        range_image=range_image,
        valid=valid,
    )
