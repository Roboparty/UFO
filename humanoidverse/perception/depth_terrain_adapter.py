"""Project optical-axis depth into the PBFM robot-centric terrain grid."""

from __future__ import annotations

import torch
from torch import nn


class DepthTerrainAdapter(nn.Module):
    """Convert calibrated depth images to pelvis-to-terrain clearances.

    Camera and pelvis quaternions use the repository's ``xyzw`` convention and
    map vectors from their local frame to the world frame. Camera coordinates
    follow the optical convention: +x right, +y down, +z forward. ``depth_z``
    is distance along that optical +z axis, in meters.
    """

    X_MIN = -0.4
    X_MAX = 1.6
    Y_MIN = -0.6
    Y_MAX = 0.6
    RESOLUTION = 0.1
    GRID_SHAPE = (21, 13)
    GRID_DIMENSION = 273
    CENTER_INDEX = 58

    def __init__(
        self,
        intrinsic_matrix: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> None:
        super().__init__()
        intrinsic_matrix = torch.as_tensor(intrinsic_matrix)
        if not intrinsic_matrix.is_floating_point():
            intrinsic_matrix = intrinsic_matrix.to(torch.float32)
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError(f"intrinsic_matrix must have shape [3, 3], got {tuple(intrinsic_matrix.shape)}")
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if not torch.isfinite(intrinsic_matrix).all():
            raise ValueError("intrinsic_matrix must contain only finite values")

        try:
            inverse_intrinsics = torch.linalg.inv(intrinsic_matrix)
        except RuntimeError as error:
            raise ValueError("intrinsic_matrix must be invertible") from error

        rows, columns = torch.meshgrid(
            torch.arange(image_height, dtype=intrinsic_matrix.dtype, device=intrinsic_matrix.device),
            torch.arange(image_width, dtype=intrinsic_matrix.dtype, device=intrinsic_matrix.device),
            indexing="ij",
        )
        homogeneous_pixels = torch.stack((columns, rows, torch.ones_like(columns)), dim=-1)
        rays = homogeneous_pixels @ inverse_intrinsics.transpose(0, 1)
        ray_z = rays[..., 2:3]
        if not torch.isfinite(rays).all() or torch.any(ray_z.abs() <= 1e-12):
            raise ValueError("intrinsic_matrix produces invalid optical rays")
        rays = rays / ray_z

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(self.X_MIN, self.X_MAX, self.GRID_SHAPE[0], dtype=intrinsic_matrix.dtype, device=intrinsic_matrix.device),
            torch.linspace(self.Y_MIN, self.Y_MAX, self.GRID_SHAPE[1], dtype=intrinsic_matrix.dtype, device=intrinsic_matrix.device),
            indexing="ij",
        )
        grid_offsets = torch.stack(
            (
                grid_x.reshape(-1),
                grid_y.reshape(-1),
                torch.zeros(self.GRID_DIMENSION, dtype=intrinsic_matrix.dtype, device=intrinsic_matrix.device),
            ),
            dim=-1,
        )
        self.register_buffer("intrinsic_matrix", intrinsic_matrix.clone())
        self.register_buffer("pixel_ray_lut", rays)
        self.register_buffer("grid_offsets", grid_offsets)
        self.image_height = image_height
        self.image_width = image_width

    @staticmethod
    def _rotate_xyzw(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
        xyz = quaternion[..., :3]
        w = quaternion[..., 3:4]
        while xyz.ndim < vectors.ndim:
            xyz = xyz.unsqueeze(-2)
            w = w.unsqueeze(-2)
        cross = 2.0 * torch.cross(xyz.expand_as(vectors), vectors, dim=-1)
        return vectors + w * cross + torch.cross(xyz.expand_as(vectors), cross, dim=-1)

    @classmethod
    def _rotate_inverse_xyzw(cls, quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        conjugate = torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)
        return cls._rotate_xyzw(conjugate, vectors)

    def forward(
        self,
        depth_z: torch.Tensor,
        camera_pos_w: torch.Tensor,
        camera_optical_quat_w: torch.Tensor,
        pelvis_pos_w: torch.Tensor,
        pelvis_heading_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(clearance, visible)`` tensors with shape ``[B, 273]``.

        Invisible cells contain ``NaN`` clearance and ``False`` in the mask.
        If several depth pixels enter one cell, the highest surface wins.
        """
        if depth_z.ndim != 3 or tuple(depth_z.shape[1:]) != (self.image_height, self.image_width):
            raise ValueError(f"depth_z must have shape [B, {self.image_height}, {self.image_width}], got {tuple(depth_z.shape)}")
        batch_size = depth_z.shape[0]
        expected_shapes = {
            "camera_pos_w": (batch_size, 3),
            "camera_optical_quat_w": (batch_size, 4),
            "pelvis_pos_w": (batch_size, 3),
            "pelvis_heading_quat_w": (batch_size, 4),
        }
        inputs = {
            "camera_pos_w": camera_pos_w,
            "camera_optical_quat_w": camera_optical_quat_w,
            "pelvis_pos_w": pelvis_pos_w,
            "pelvis_heading_quat_w": pelvis_heading_quat_w,
        }
        for name, expected in expected_shapes.items():
            if tuple(inputs[name].shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(inputs[name].shape)}")
        for name in ("camera_pos_w", "pelvis_pos_w"):
            if not torch.isfinite(inputs[name]).all():
                raise ValueError(f"{name} must contain only finite values")
        for name in ("camera_optical_quat_w", "pelvis_heading_quat_w"):
            quaternion = inputs[name]
            if not torch.isfinite(quaternion).all():
                raise ValueError(f"{name} must contain only finite values")
            if torch.any(torch.linalg.vector_norm(quaternion, dim=-1) <= 1e-8):
                raise ValueError(f"{name} must have non-zero norm")

        rays = self.pixel_ray_lut.to(device=depth_z.device, dtype=depth_z.dtype)
        points_camera = rays.unsqueeze(0) * depth_z.unsqueeze(-1)
        flat_camera = points_camera.reshape(batch_size, -1, 3)
        points_world = self._rotate_xyzw(camera_optical_quat_w, flat_camera) + camera_pos_w[:, None, :]
        points_heading = self._rotate_inverse_xyzw(pelvis_heading_quat_w, points_world - pelvis_pos_w[:, None, :])

        ix = torch.floor((points_heading[..., 0] - self.X_MIN) / self.RESOLUTION + 0.5).long()
        iy = torch.floor((points_heading[..., 1] - self.Y_MIN) / self.RESOLUTION + 0.5).long()
        valid = (
            torch.isfinite(depth_z.reshape(batch_size, -1))
            & (depth_z.reshape(batch_size, -1) > 0.0)
            & torch.isfinite(points_heading).all(dim=-1)
            & (ix >= 0)
            & (ix < self.GRID_SHAPE[0])
            & (iy >= 0)
            & (iy < self.GRID_SHAPE[1])
        )
        indices = (ix * self.GRID_SHAPE[1] + iy).clamp(0, self.GRID_DIMENSION - 1)
        clearances = (pelvis_pos_w[:, None, 2] - points_world[..., 2]).clamp_min(0.0)
        clearances = torch.where(valid, clearances, torch.full_like(clearances, float("inf")))

        projected = torch.full(
            (batch_size, self.GRID_DIMENSION),
            float("inf"),
            device=depth_z.device,
            dtype=depth_z.dtype,
        )
        projected.scatter_reduce_(1, indices, clearances, reduce="amin", include_self=True)
        visible = torch.isfinite(projected)
        projected = torch.where(visible, projected, torch.full_like(projected, float("nan")))
        return projected, visible
