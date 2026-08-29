"""Realistic self-occluding depth preprocessing for terrain completion."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from humanoidverse.perception.depth_augmentation import MetricDepthAugmentation
from humanoidverse.perception.depth_camera import DepthCameraConfig


@dataclass(frozen=True)
class SelfOcclusionDepthConfig:
    """Semantic contract shared by simulation collection and deployment."""

    min_ray_range_m: float = 0.10
    max_ray_range_m: float = 2.0
    hit_tolerance_m: float = 0.002
    dilation_sigma_multiplier: float = 3.0
    terrain_geom_groups: tuple[int, ...] = (5,)
    scene_geom_groups: tuple[int, ...] = (2, 3, 5)
    camera_housing_geom_names: tuple[str, ...] = ("head_collision",)
    camera_housing_mesh_names: tuple[str, ...] = ("head_link",)
    camera_housing_geom_group: int = 4

    def validate(self) -> None:
        if not 0.0 <= self.min_ray_range_m < self.max_ray_range_m:
            raise ValueError("ray range must satisfy 0 <= min < max")
        if self.hit_tolerance_m <= 0.0:
            raise ValueError("hit_tolerance_m must be positive")
        if self.dilation_sigma_multiplier < 0.0:
            raise ValueError("dilation_sigma_multiplier must be non-negative")
        if 5 not in self.terrain_geom_groups or 5 not in self.scene_geom_groups:
            raise ValueError("terrain group 5 must be visible to both synchronized raycasts")
        if not set(self.terrain_geom_groups).issubset(self.scene_geom_groups):
            raise ValueError("scene raycast must include every terrain-only geom group")
        if self.camera_housing_geom_group in self.scene_geom_groups:
            raise ValueError("camera-housing exclusion group must not be included in the scene raycast")

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> SelfOcclusionDepthConfig:
        """Restore semantic fields while ignoring derived collection metadata."""
        names = {field.name for field in fields(cls)}
        values = {name: metadata[name] for name in names if name in metadata}
        for name in (
            "terrain_geom_groups",
            "scene_geom_groups",
            "camera_housing_geom_names",
            "camera_housing_mesh_names",
        ):
            if name in values:
                values[name] = tuple(values[name])
        config = cls(**values)
        config.validate()
        return config


@dataclass
class SelfOccludingDepthFrame:
    """One batched semantic first-hit frame in optical-depth convention."""

    terrain_depth_z: torch.Tensor
    scene_depth_z: torch.Tensor
    final_depth_z: torch.Tensor
    terrain_mask: torch.Tensor
    self_mask: torch.Tensor
    dilated_self_mask: torch.Tensor
    far_or_no_hit_mask: torch.Tensor
    ambiguous_mask: torch.Tensor
    valid_terrain_mask: torch.Tensor
    sigma_px: torch.Tensor
    dilation_radius_px: torch.Tensor


def make_self_occlusion_camera_pair(
    camera: DepthCameraConfig,
    config: SelfOcclusionDepthConfig,
) -> tuple[DepthCameraConfig, DepthCameraConfig]:
    """Return strictly co-registered terrain-only and scene-first-hit cameras."""
    config.validate()
    shared = {
        "min_range": config.min_ray_range_m,
        "max_range": config.max_ray_range_m,
    }
    terrain = replace(
        camera,
        name=f"{camera.name}_terrain_only",
        include_geom_groups=config.terrain_geom_groups,
        **shared,
    )
    scene = replace(
        camera,
        name=f"{camera.name}_scene_first_hit",
        include_geom_groups=config.scene_geom_groups,
        **shared,
    )
    terrain.validate()
    scene.validate()
    return terrain, scene


def _optical_unit_z(camera: DepthCameraConfig, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    intrinsic = camera.intrinsics().to(device=device, dtype=dtype)
    rows, columns = torch.meshgrid(
        torch.arange(camera.height, device=device, dtype=dtype),
        torch.arange(camera.width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack((columns, rows, torch.ones_like(columns)), dim=-1)
    optical = pixels @ torch.linalg.inv(intrinsic).transpose(0, 1)
    return optical[..., 2] / torch.linalg.vector_norm(optical, dim=-1)


def classify_synchronized_first_hits(
    terrain_range_m: torch.Tensor,
    scene_range_m: torch.Tensor,
    config: SelfOcclusionDepthConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify terrain, self, far/no-hit, and inconsistent first hits."""
    config.validate()
    terrain_range_m = torch.as_tensor(terrain_range_m)
    scene_range_m = torch.as_tensor(scene_range_m, device=terrain_range_m.device, dtype=terrain_range_m.dtype)
    if terrain_range_m.shape != scene_range_m.shape or terrain_range_m.ndim != 3:
        raise ValueError("terrain and scene ranges must have matching [B, H, W] shapes")
    terrain_raw_hit = (
        torch.isfinite(terrain_range_m)
        & (terrain_range_m >= 0.0)
        & (terrain_range_m <= config.max_ray_range_m)
    )
    scene_raw_hit = (
        torch.isfinite(scene_range_m)
        & (scene_range_m >= 0.0)
        & (scene_range_m <= config.max_ray_range_m)
    )
    terrain_hit = terrain_raw_hit & (terrain_range_m >= config.min_ray_range_m)
    scene_hit = scene_raw_hit & (scene_range_m >= config.min_ray_range_m)
    delta = scene_range_m - terrain_range_m
    same_terrain = terrain_hit & scene_hit & (delta.abs() <= config.hit_tolerance_m)
    self_hit = scene_raw_hit & (~terrain_raw_hit | (delta < -config.hit_tolerance_m))
    ambiguous = (terrain_raw_hit & ~scene_raw_hit) | (
        terrain_raw_hit & scene_raw_hit & (delta > config.hit_tolerance_m)
    )
    far_or_no_hit = ~(same_terrain | self_hit | ambiguous)
    return same_terrain, self_hit, far_or_no_hit, ambiguous


def dilate_self_mask(
    self_mask: torch.Tensor,
    sigma_px: torch.Tensor,
    *,
    sigma_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dilate each image by ``ceil(sigma_multiplier * sigma)`` pixels."""
    self_mask = torch.as_tensor(self_mask, dtype=torch.bool)
    sigma_px = torch.as_tensor(sigma_px, device=self_mask.device, dtype=torch.float32)
    if self_mask.ndim != 3 or sigma_px.shape != (self_mask.shape[0],):
        raise ValueError("self_mask must be [B, H, W] and sigma_px must be [B]")
    if sigma_multiplier < 0.0 or not torch.isfinite(sigma_px).all() or torch.any(sigma_px < 0.0):
        raise ValueError("dilation parameters must be finite and non-negative")
    radii = torch.ceil(float(sigma_multiplier) * sigma_px).to(torch.long)
    output = self_mask.clone()
    for radius in torch.unique(radii).tolist():
        radius = int(radius)
        if radius <= 0:
            continue
        selected = radii == radius
        pooled = F.max_pool2d(
            self_mask[selected].to(torch.float32).unsqueeze(1),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        ).squeeze(1)
        output[selected] = pooled > 0.0
    return output, radii


def self_occluding_depth_from_sensors(
    terrain_sensor,
    scene_sensor,
    camera: DepthCameraConfig,
    config: SelfOcclusionDepthConfig,
    augmentation: MetricDepthAugmentation,
) -> SelfOccludingDepthFrame:
    """Build terrain-only optical depth from synchronized semantic first hits."""
    terrain_range = terrain_sensor.data.distances.reshape(-1, camera.height, camera.width)
    scene_range = scene_sensor.data.distances.reshape(-1, camera.height, camera.width)
    if terrain_range.shape != scene_range.shape:
        raise RuntimeError("synchronized depth sensors returned different shapes")
    terrain_mask, self_mask, far_mask, ambiguous_mask = classify_synchronized_first_hits(
        terrain_range,
        scene_range,
        config,
    )
    unit_z = _optical_unit_z(camera, device=terrain_range.device, dtype=terrain_range.dtype)
    terrain_depth_z = terrain_range * unit_z
    scene_depth_z = scene_range * unit_z
    terrain_depth_z = torch.where(terrain_mask, terrain_depth_z, torch.full_like(terrain_depth_z, float("nan")))
    scene_depth_z = torch.where(
        torch.isfinite(scene_range) & (scene_range >= config.min_ray_range_m),
        scene_depth_z,
        torch.full_like(scene_depth_z, float("nan")),
    )
    sigma = augmentation.sample_sigma(
        terrain_range.shape[0],
        device=terrain_range.device,
        dtype=terrain_range.dtype,
    )
    dilated_self, radii = dilate_self_mask(
        self_mask,
        sigma,
        sigma_multiplier=config.dilation_sigma_multiplier,
    )
    valid_terrain = terrain_mask & ~dilated_self
    final_depth_z, valid_terrain, sigma = augmentation.apply_to_valid_depth(
        terrain_depth_z,
        valid_terrain,
        sigma=sigma,
    )
    if not torch.equal(torch.isfinite(final_depth_z), valid_terrain):
        raise RuntimeError("self-occluding depth preprocessing expanded or lost semantic validity")
    return SelfOccludingDepthFrame(
        terrain_depth_z=terrain_depth_z,
        scene_depth_z=scene_depth_z,
        final_depth_z=final_depth_z,
        terrain_mask=terrain_mask,
        self_mask=self_mask,
        dilated_self_mask=dilated_self,
        far_or_no_hit_mask=far_mask,
        ambiguous_mask=ambiguous_mask,
        valid_terrain_mask=valid_terrain,
        sigma_px=sigma,
        dilation_radius_px=radii,
    )


def expected_max_dilation_radius(config: SelfOcclusionDepthConfig, sigma_max_px: float) -> int:
    """Return the largest raw-image dilation radius for metadata/preflight."""
    return int(math.ceil(config.dilation_sigma_multiplier * float(sigma_max_px)))
