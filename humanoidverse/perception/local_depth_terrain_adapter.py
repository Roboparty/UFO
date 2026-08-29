"""Odometry-free depth projection into the PBFM terrain grid."""

from __future__ import annotations

import torch

from humanoidverse.perception.depth_camera import quaternion_multiply_xyzw, rotate_xyzw
from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter

TORSO_ORIGIN_IN_WAIST_YAW = (-0.0039635, 0.0, 0.044)


def _axis_quaternion_xyzw(angles: torch.Tensor, axis: int) -> torch.Tensor:
    half = 0.5 * angles
    quaternion = torch.zeros((*angles.shape, 4), device=angles.device, dtype=angles.dtype)
    quaternion[..., axis] = torch.sin(half)
    quaternion[..., 3] = torch.cos(half)
    return quaternion


def g1_torso_pose_in_pelvis(waist_joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return torso position and xyzw orientation in the G1 pelvis frame."""
    if waist_joint_pos.ndim != 2 or waist_joint_pos.shape[1] != 3:
        raise ValueError("waist_joint_pos must have shape [B, 3] in yaw, roll, pitch order")
    if not torch.isfinite(waist_joint_pos).all():
        raise ValueError("waist_joint_pos must contain only finite values")
    yaw = _axis_quaternion_xyzw(waist_joint_pos[:, 0], 2)
    roll = _axis_quaternion_xyzw(waist_joint_pos[:, 1], 0)
    pitch = _axis_quaternion_xyzw(waist_joint_pos[:, 2], 1)
    torso_quat_p = quaternion_multiply_xyzw(quaternion_multiply_xyzw(yaw, roll), pitch)
    torso_offset = torch.tensor(
        TORSO_ORIGIN_IN_WAIST_YAW,
        device=waist_joint_pos.device,
        dtype=waist_joint_pos.dtype,
    ).expand(waist_joint_pos.shape[0], -1)
    torso_pos_p = rotate_xyzw(yaw, torso_offset)
    return torso_pos_p, torso_quat_p


def gravity_aligned_basis_in_pelvis(projected_gravity: torch.Tensor) -> torch.Tensor:
    """Return ``R_PH`` whose columns are heading-frame axes in pelvis coordinates."""
    if projected_gravity.ndim != 2 or projected_gravity.shape[1] != 3:
        raise ValueError("projected_gravity must have shape [B, 3]")
    gravity_norm = torch.linalg.vector_norm(projected_gravity, dim=-1, keepdim=True)
    if not torch.isfinite(projected_gravity).all() or torch.any(gravity_norm <= 1.0e-8):
        raise ValueError("projected_gravity must be finite and non-zero")
    up_p = -projected_gravity / gravity_norm
    forward_p = torch.zeros_like(up_p)
    forward_p[:, 0] = 1.0
    forward_horizontal = forward_p - (forward_p * up_p).sum(dim=-1, keepdim=True) * up_p
    forward_norm = torch.linalg.vector_norm(forward_horizontal, dim=-1, keepdim=True)
    if torch.any(forward_norm <= 1.0e-6):
        raise ValueError("pelvis forward axis is degenerate with gravity")
    x_p = forward_horizontal / forward_norm
    y_p = torch.cross(up_p, x_p, dim=-1)
    return torch.stack((x_p, y_p, up_p), dim=-1)


class LocalDepthTerrainAdapter(DepthTerrainAdapter):
    """Project depth using only IMU gravity, waist FK, and camera extrinsics.

    The adapter never consumes a world pose, global yaw, base linear velocity,
    or relative odometry. Camera coordinates use optical ``+x right, +y down,
    +z forward`` and output clearance remains in meters.
    """

    def __init__(
        self,
        intrinsic_matrix: torch.Tensor,
        image_height: int,
        image_width: int,
        *,
        camera_pos_torso: tuple[float, float, float],
        camera_optical_quat_torso_xyzw: tuple[float, float, float, float],
    ) -> None:
        super().__init__(intrinsic_matrix, image_height, image_width)
        camera_pos = torch.as_tensor(camera_pos_torso, dtype=self.intrinsic_matrix.dtype)
        camera_quat = torch.as_tensor(camera_optical_quat_torso_xyzw, dtype=self.intrinsic_matrix.dtype)
        if camera_pos.shape != (3,) or not torch.isfinite(camera_pos).all():
            raise ValueError("camera_pos_torso must contain three finite values")
        if camera_quat.shape != (4,) or not torch.isfinite(camera_quat).all():
            raise ValueError("camera_optical_quat_torso_xyzw must contain four finite values")
        if torch.linalg.vector_norm(camera_quat) <= 1.0e-8:
            raise ValueError("camera optical quaternion must have non-zero norm")
        self.register_buffer("camera_pos_torso", camera_pos)
        self.register_buffer(
            "camera_optical_quat_torso_xyzw",
            camera_quat / torch.linalg.vector_norm(camera_quat),
        )
        rows, columns = torch.meshgrid(
            torch.arange(image_height, dtype=self.intrinsic_matrix.dtype),
            torch.arange(image_width, dtype=self.intrinsic_matrix.dtype),
            indexing="ij",
        )
        self.register_buffer(
            "homogeneous_pixels",
            torch.stack((columns, rows, torch.ones_like(columns)), dim=-1),
        )

    def camera_pose_in_pelvis(
        self,
        waist_joint_pos: torch.Tensor,
        *,
        camera_pos_torso: torch.Tensor | None = None,
        camera_optical_quat_torso_xyzw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torso_pos_p, torso_quat_p = g1_torso_pose_in_pelvis(waist_joint_pos)
        batch_size = waist_joint_pos.shape[0]
        if camera_pos_torso is None:
            camera_pos_t = self.camera_pos_torso.expand(batch_size, -1)
        else:
            camera_pos_t = torch.as_tensor(camera_pos_torso)
            if camera_pos_t.shape != (batch_size, 3) or not torch.isfinite(camera_pos_t).all():
                raise ValueError("camera_pos_torso override must be finite with shape [B, 3]")
        if camera_optical_quat_torso_xyzw is None:
            camera_quat_t = self.camera_optical_quat_torso_xyzw.expand(batch_size, -1)
        else:
            camera_quat_t = torch.as_tensor(camera_optical_quat_torso_xyzw)
            if camera_quat_t.shape != (batch_size, 4) or not torch.isfinite(camera_quat_t).all():
                raise ValueError("camera quaternion override must be finite with shape [B, 4]")
            if torch.any(torch.linalg.vector_norm(camera_quat_t, dim=-1) <= 1.0e-8):
                raise ValueError("camera quaternion override must have non-zero norm")
        camera_pos_t = camera_pos_t.to(device=waist_joint_pos.device, dtype=waist_joint_pos.dtype)
        camera_quat_t = camera_quat_t.to(device=waist_joint_pos.device, dtype=waist_joint_pos.dtype)
        camera_pos_p = torso_pos_p + rotate_xyzw(torso_quat_p, camera_pos_t)
        camera_quat_p = quaternion_multiply_xyzw(torso_quat_p, camera_quat_t)
        return camera_pos_p, camera_quat_p

    def forward(
        self,
        depth_z: torch.Tensor,
        projected_gravity: torch.Tensor,
        waist_joint_pos: torch.Tensor,
        *,
        intrinsic_matrix: torch.Tensor | None = None,
        camera_pos_torso: torch.Tensor | None = None,
        camera_optical_quat_torso_xyzw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if depth_z.ndim != 3 or tuple(depth_z.shape[1:]) != (self.image_height, self.image_width):
            raise ValueError(f"depth_z must have shape [B, {self.image_height}, {self.image_width}], got {tuple(depth_z.shape)}")
        batch_size = depth_z.shape[0]
        if projected_gravity.shape != (batch_size, 3):
            raise ValueError("projected_gravity must have shape [B, 3]")
        if waist_joint_pos.shape != (batch_size, 3):
            raise ValueError("waist_joint_pos must have shape [B, 3]")

        if intrinsic_matrix is None:
            rays = self.pixel_ray_lut.to(device=depth_z.device, dtype=depth_z.dtype).unsqueeze(0)
        else:
            intrinsic_matrix = torch.as_tensor(intrinsic_matrix, device=depth_z.device, dtype=depth_z.dtype)
            if intrinsic_matrix.shape != (batch_size, 3, 3) or not torch.isfinite(intrinsic_matrix).all():
                raise ValueError("intrinsic_matrix override must be finite with shape [B, 3, 3]")
            inverse = torch.linalg.inv(intrinsic_matrix)
            pixels = self.homogeneous_pixels.to(device=depth_z.device, dtype=depth_z.dtype)
            rays = torch.einsum("hwj,bkj->bhwk", pixels, inverse)
            if torch.any(rays[..., 2].abs() <= 1.0e-12):
                raise ValueError("intrinsic_matrix override produces invalid optical rays")
            rays = rays / rays[..., 2:3]
        points_camera = (rays * depth_z.unsqueeze(-1)).reshape(batch_size, -1, 3)
        camera_pos_p, camera_quat_p = self.camera_pose_in_pelvis(
            waist_joint_pos,
            camera_pos_torso=camera_pos_torso,
            camera_optical_quat_torso_xyzw=camera_optical_quat_torso_xyzw,
        )
        points_p = rotate_xyzw(camera_quat_p, points_camera) + camera_pos_p[:, None, :]
        pelvis_from_heading = gravity_aligned_basis_in_pelvis(projected_gravity)
        points_heading = torch.einsum("bni,bij->bnj", points_p, pelvis_from_heading)
        return self._rasterize_heading_points(depth_z, points_heading)
