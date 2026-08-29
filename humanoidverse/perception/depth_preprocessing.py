"""Dependency-free depth preprocessing shared by simulation and hardware runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DepthCropConfig:
    """Angular ROI expressed in pixels of a fixed reference image.

    The Phase-2I camera is rendered at 480x270 but the historical crop
    candidates were specified on the 64x36 network image.  Keeping the
    reference resolution explicit makes ``top=6`` mean the same angular crop
    at either resolution while still cropping the raw frame before resize.
    """

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    reference_width: int = 64
    reference_height: int = 36

    def validate(self) -> None:
        values = (self.top, self.bottom, self.left, self.right)
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("crop margins must be non-negative integers")
        if min(self.reference_width, self.reference_height) <= 0:
            raise ValueError("crop reference dimensions must be positive")
        if self.top + self.bottom >= self.reference_height:
            raise ValueError("vertical crop removes the complete reference image")
        if self.left + self.right >= self.reference_width:
            raise ValueError("horizontal crop removes the complete reference image")

    @classmethod
    def from_metadata(cls, value: Mapping[str, Any] | None) -> "DepthCropConfig":
        config = cls() if value is None else cls(**dict(value))
        config.validate()
        return config

    def to_metadata(self) -> dict[str, int]:
        self.validate()
        return asdict(self)

    def native_bounds(self, *, native_height: int, native_width: int) -> tuple[int, int, int, int]:
        """Return ``(top, bottom, left, right)`` slice bounds at native size."""
        self.validate()
        if min(native_height, native_width) <= 0:
            raise ValueError("native image dimensions must be positive")
        top = round(self.top * native_height / self.reference_height)
        bottom_margin = round(self.bottom * native_height / self.reference_height)
        left = round(self.left * native_width / self.reference_width)
        right_margin = round(self.right * native_width / self.reference_width)
        bottom = native_height - bottom_margin
        right = native_width - right_margin
        if top >= bottom or left >= right:
            raise ValueError("scaled crop removes the complete native image")
        return top, bottom, left, right


DEPTH_CROP_CANDIDATES: dict[str, DepthCropConfig] = {
    "full": DepthCropConfig(),
    "top6_side4": DepthCropConfig(top=6, left=4, right=4),
    "top10_side8": DepthCropConfig(top=10, left=8, right=8),
    "top14_side12": DepthCropConfig(top=14, left=12, right=12),
    "top18_side16": DepthCropConfig(top=18, left=16, right=16),
}


def depth_crop_candidate(name: str) -> DepthCropConfig:
    try:
        return DEPTH_CROP_CANDIDATES[name]
    except KeyError as error:
        raise ValueError(f"unknown depth crop candidate: {name!r}") from error


def scale_full_fov_intrinsics(
    intrinsic_matrix: torch.Tensor,
    *,
    native_height: int,
    native_width: int,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Scale calibrated intrinsics for a full-frame resize using pixel centers."""
    if min(native_height, native_width, target_height, target_width) <= 0:
        raise ValueError("image dimensions must be positive")
    intrinsic_matrix = torch.as_tensor(intrinsic_matrix)
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError("intrinsic_matrix must have shape [3, 3]")
    scale_x = target_width / native_width
    scale_y = target_height / native_height
    target = intrinsic_matrix.clone()
    target[0, 0] *= scale_x
    target[0, 1] *= scale_x
    target[0, 2] = (intrinsic_matrix[0, 2] + 0.5) * scale_x - 0.5
    target[1, 1] *= scale_y
    target[1, 2] = (intrinsic_matrix[1, 2] + 0.5) * scale_y - 0.5
    return target


def crop_and_scale_intrinsics(
    intrinsic_matrix: torch.Tensor,
    *,
    native_height: int,
    native_width: int,
    target_height: int,
    target_width: int,
    crop: DepthCropConfig | None = None,
) -> torch.Tensor:
    """Adjust principal point for a raw-frame crop, then scale pixel centers."""
    crop = crop or DepthCropConfig()
    top, bottom, left, right = crop.native_bounds(
        native_height=native_height,
        native_width=native_width,
    )
    intrinsic_matrix = torch.as_tensor(intrinsic_matrix)
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError("intrinsic_matrix must have shape [3, 3]")
    cropped = intrinsic_matrix.clone()
    cropped[0, 2] -= left
    cropped[1, 2] -= top
    return scale_full_fov_intrinsics(
        cropped,
        native_height=bottom - top,
        native_width=right - left,
        target_height=target_height,
        target_width=target_width,
    )


def crop_depth_roi(depth_m: torch.Tensor, crop: DepthCropConfig | None = None) -> torch.Tensor:
    """Crop the final two image dimensions using a resolution-invariant ROI."""
    depth_m = torch.as_tensor(depth_m)
    if depth_m.ndim not in (2, 3):
        raise ValueError("depth_m must have shape [H, W] or [B, H, W]")
    crop = crop or DepthCropConfig()
    top, bottom, left, right = crop.native_bounds(
        native_height=depth_m.shape[-2],
        native_width=depth_m.shape[-1],
    )
    return depth_m[..., top:bottom, left:right]


def resize_full_fov_depth(depth_m: torch.Tensor, *, target_height: int, target_width: int) -> torch.Tensor:
    """Area-downsample depth without cropping or turning invalid pixels into zero."""
    depth_m = torch.as_tensor(depth_m)
    squeeze = depth_m.ndim == 2
    if squeeze:
        depth_m = depth_m.unsqueeze(0)
    if depth_m.ndim != 3 or min(target_height, target_width) <= 0:
        raise ValueError("depth_m must have shape [H, W] or [B, H, W]")
    if tuple(depth_m.shape[-2:]) == (target_height, target_width):
        return depth_m.squeeze(0) if squeeze else depth_m
    valid = torch.isfinite(depth_m) & (depth_m > 0.0)
    values = torch.where(valid, depth_m, torch.zeros_like(depth_m))
    values = F.interpolate(values.unsqueeze(1), (target_height, target_width), mode="area").squeeze(1)
    weights = F.interpolate(valid.to(values.dtype).unsqueeze(1), (target_height, target_width), mode="area").squeeze(1)
    result = values / weights.clamp_min(1.0e-8)
    result = torch.where(weights > 0.0, result, torch.full_like(result, float("nan")))
    return result.squeeze(0) if squeeze else result


def crop_and_resize_depth(
    depth_m: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    crop: DepthCropConfig | None = None,
) -> torch.Tensor:
    """Crop native depth first, then validity-aware area resize."""
    return resize_full_fov_depth(
        crop_depth_roi(depth_m, crop),
        target_height=target_height,
        target_width=target_width,
    )


def resize_depth_with_conservative_invalid_mask(
    depth_m: torch.Tensor,
    invalid_mask: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Area-resize valid depth while preserving any masked self occlusion."""
    depth_m = torch.as_tensor(depth_m)
    invalid_mask = torch.as_tensor(invalid_mask, device=depth_m.device, dtype=torch.bool)
    if depth_m.ndim != 3 or invalid_mask.shape != depth_m.shape:
        raise ValueError("depth_m and invalid_mask must have matching [B, H, W] shapes")
    resized = resize_full_fov_depth(depth_m, target_height=target_height, target_width=target_width)
    invalid_fraction = F.interpolate(
        invalid_mask.to(depth_m.dtype).unsqueeze(1),
        (target_height, target_width),
        mode="area",
    ).squeeze(1)
    resized_invalid = invalid_fraction > 0.0
    resized = torch.where(resized_invalid, torch.full_like(resized, float("nan")), resized)
    return resized, resized_invalid


def crop_and_resize_depth_with_conservative_invalid_mask(
    depth_m: torch.Tensor,
    invalid_mask: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    crop: DepthCropConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop both depth and semantic-invalid mask before conservative resize."""
    depth_m = torch.as_tensor(depth_m)
    invalid_mask = torch.as_tensor(invalid_mask, device=depth_m.device, dtype=torch.bool)
    if depth_m.ndim != 3 or invalid_mask.shape != depth_m.shape:
        raise ValueError("depth_m and invalid_mask must have matching [B, H, W] shapes")
    return resize_depth_with_conservative_invalid_mask(
        crop_depth_roi(depth_m, crop),
        crop_depth_roi(invalid_mask, crop),
        target_height=target_height,
        target_width=target_width,
    )
