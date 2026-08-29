"""RP1-compatible direct-depth preprocessing and temporal stacking.

The camera mount remains the G1 mount used by this repository. Image
processing, invalid-pixel semantics, quantization, delay, and history layout
follow ``/home/xue/UFO-rp1`` at the pinned source revision below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from humanoidverse.perception.depth_camera import DepthCameraConfig, optical_rays_from_intrinsics

REFERENCE_PROJECT = "UFO-rp1"
REFERENCE_COMMIT = "8c364e1001734097aac58e5033a1b5076925d3c5"
BASELINE_TYPE = "rp1_direct_depth"


@dataclass(frozen=True)
class RP1DirectDepthConfig:
    width: int = 64
    height: int = 36
    horizontal_fov_deg: float = 89.51
    vertical_fov_deg: float = 58.29
    min_distance: float = 0.0
    max_distance: float = 2.5
    crop: tuple[int, int, int, int] = (0, 0, 16, 16)
    history_length: int = 37
    history_skip_frames: int = 5
    num_output_frames: int = 8
    delayed_frame_ranges: tuple[int, int] = (0, 1)
    position_error: float = 0.05
    angle_error_rad: float = math.radians(5.0)
    include_geom_groups: tuple[int, ...] = (0, 2, 5)
    exclude_parent_body: bool = False
    mount_body: str = "torso_link"
    mount_pos: tuple[float, float, float] = (0.0487988662332928, 0.01, 0.4378029937970051)
    optical_quat_torso_xyzw: tuple[float, float, float, float] = (
        -0.6579550475696607,
        0.6623183568544073,
        -0.25121840449386046,
        0.25558171377860706,
    )

    @property
    def output_height(self) -> int:
        return self.height - self.crop[0] - self.crop[1]

    @property
    def output_width(self) -> int:
        return self.width - self.crop[2] - self.crop[3]

    @property
    def sampled_ages(self) -> tuple[int, ...]:
        return tuple(
            reversed(tuple(range(0, self.num_output_frames * self.history_skip_frames, self.history_skip_frames)))
        )

    @property
    def max_delay_frames(self) -> int:
        return self.delayed_frame_ranges[1]

    def validate(self) -> None:
        if (self.width, self.height) != (64, 36):
            raise ValueError("RP1 direct depth requires a 64x36 source image")
        if (self.output_height, self.output_width) != (36, 32):
            raise ValueError("RP1 direct-depth crop must produce a 36x32 image")
        if self.history_length != 37 or self.sampled_ages != (35, 30, 25, 20, 15, 10, 5, 0):
            raise ValueError("RP1 direct depth requires the 37-frame, skip-5 history contract")
        if self.delayed_frame_ranges != (0, 1):
            raise ValueError("RP1 direct depth requires an inclusive per-episode delay of 0 or 1 frame")
        if self.position_error < 0.0 or self.angle_error_rad < 0.0:
            raise ValueError("camera installation error bounds must be non-negative")
        if not 0.0 <= self.min_distance < self.max_distance:
            raise ValueError("camera distance bounds must satisfy 0 <= min < max")
        if not self.include_geom_groups:
            raise ValueError("direct-depth camera requires at least one visible geometry group")

    def camera_config(self) -> DepthCameraConfig:
        self.validate()
        return DepthCameraConfig(
            name="g1_direct_depth",
            width=self.width,
            height=self.height,
            horizontal_fov_deg=self.horizontal_fov_deg,
            vertical_fov_deg=self.vertical_fov_deg,
            mount_body=self.mount_body,
            mount_pos_torso=self.mount_pos,
            optical_quat_torso_xyzw=self.optical_quat_torso_xyzw,
            min_range=self.min_distance,
            max_range=self.max_distance,
            include_geom_groups=self.include_geom_groups,
        )


def raycast_ranges_to_image_plane(
    ray_distances: torch.Tensor,
    camera: DepthCameraConfig,
) -> torch.Tensor:
    """Convert Euclidean ray range to optical-Z meters with invalid pixels at zero."""
    distances = ray_distances.reshape(-1, camera.height, camera.width)
    optical_rays = optical_rays_from_intrinsics(camera, device=distances.device, dtype=distances.dtype)
    valid = (distances >= 0.0) & torch.isfinite(distances)
    depth = distances * optical_rays[..., 2]
    valid &= torch.isfinite(depth)
    return torch.where(valid, depth, torch.zeros_like(depth))


def _apply_mask(original: torch.Tensor, changed: torch.Tensor, probability: float) -> torch.Tensor:
    mask = torch.rand((original.shape[0], 1, 1), device=original.device) < probability
    return torch.where(mask, changed, original)


def _apply_domain_randomization(depth: torch.Tensor) -> torch.Tensor:
    """Apply the metric-space augmentation sequence used by UFO-rp1."""
    scales = torch.empty((depth.shape[0], 1, 1), device=depth.device).uniform_(0.95, 1.05)
    depth = _apply_mask(depth, depth * scales, 0.7)

    x = depth[:, None]
    gradient = torch.zeros_like(x)
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    gradient[:, :, :, 1:] += dx
    gradient[:, :, :, :-1] += dx
    gradient[:, :, 1:, :] += dy
    gradient[:, :, :-1, :] += dy
    local_mean = F.avg_pool2d(x, 3, stride=1, padding=1)
    local_var = F.avg_pool2d((x - local_mean).square(), 3, stride=1, padding=1)
    holes = ((gradient > 0.09) | (local_var < 4.0e-4)) & (torch.rand_like(x) < 0.02)
    depth = _apply_mask(depth, depth.masked_fill(holes[:, 0], 0.0), 0.5)

    kernel = torch.randn((1, 1, 3, 3), device=depth.device) * 0.01
    kernel[:, :, 1, 1] += 1.0
    kernel = kernel / kernel.abs().sum().clamp_min(1.0e-6)
    depth = _apply_mask(depth, F.conv2d(depth[:, None], kernel, padding=1)[:, 0], 0.4)

    correlated = torch.zeros_like(depth[:, None])
    frequency, amplitude = 8.0, 1.0
    for _ in range(4):
        grid_height = max(2, int(round(depth.shape[-2] / frequency)))
        grid_width = max(2, int(round(depth.shape[-1] / frequency)))
        octave = torch.randn((depth.shape[0], 1, grid_height, grid_width), device=depth.device)
        correlated += amplitude * F.interpolate(
            octave,
            depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        frequency *= 2.0
        amplitude *= 0.5
    correlated /= correlated.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
    depth = _apply_mask(depth, depth + 0.025 * correlated[:, 0], 0.6)

    failures = depth.clone()
    failures[torch.rand_like(failures) < 4.0e-3] = 0.0
    return _apply_mask(depth, failures, 0.7)


def _gaussian_blur(depth: torch.Tensor) -> torch.Tensor:
    gaussian = torch.tensor([1.0, 2.0, 1.0], device=depth.device, dtype=depth.dtype)
    kernel = (gaussian[:, None] * gaussian[None, :]).reshape(1, 1, 3, 3) / 16.0
    return F.conv2d(F.pad(depth[:, None], (1, 1, 1, 1), mode="reflect"), kernel)[:, 0]


def _sanitize_depth(depth: torch.Tensor, max_depth: float) -> torch.Tensor:
    """Replace non-finite values before crop and augmentations, as in UFO-rp1."""
    invalid_depth = 0.0
    return torch.nan_to_num(
        depth,
        nan=invalid_depth,
        posinf=max_depth,
        neginf=invalid_depth,
    )


def preprocess_rp1_depth(
    raw_depth: torch.Tensor,
    cfg: RP1DirectDepthConfig,
    *,
    enable_noise: bool,
) -> torch.Tensor:
    """Flip, crop, augment, blur, clamp, and quantize one depth frame."""
    cfg.validate()
    if raw_depth.ndim != 3 or raw_depth.shape[-2:] != (cfg.height, cfg.width):
        raise ValueError(f"raw_depth must have shape [B, {cfg.height}, {cfg.width}], got {tuple(raw_depth.shape)}")
    depth = _sanitize_depth(raw_depth, cfg.max_distance)
    up, down, left, right = cfg.crop
    depth = depth.flip(1)[:, up : cfg.height - down, left : cfg.width - right]
    if enable_noise:
        depth = _apply_domain_randomization(depth)
    depth = _gaussian_blur(depth).clamp(cfg.min_distance, cfg.max_distance)
    normalized = (depth - cfg.min_distance) / (cfg.max_distance - cfg.min_distance)
    return normalized.mul(255.0).round().to(torch.uint8)


class RP1DirectDepthRuntime:
    """Maintain RP1's delayed 8-frame input in a 37-frame uint8 ring."""

    output_dtype = torch.uint8
    bytes_per_value = torch.empty((), dtype=output_dtype).element_size()

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        cfg: RP1DirectDepthConfig,
        *,
        enable_noise: bool,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.camera = cfg.camera_config()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.enable_noise = bool(enable_noise)
        self._history = torch.zeros(
            (self.num_envs, cfg.history_length, cfg.output_height, cfg.output_width),
            dtype=self.output_dtype,
            device=self.device,
        )
        self._frame_offsets = torch.tensor(cfg.sampled_ages, dtype=torch.long, device=self.device)
        self._delay_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._write_index = 0
        self._latest = torch.zeros(
            (self.num_envs, cfg.num_output_frames, cfg.output_height, cfg.output_width),
            dtype=self.output_dtype,
            device=self.device,
        )

    @property
    def delay_frames(self) -> torch.Tensor:
        return self._delay_frames

    def current_frame(self, sensor) -> torch.Tensor:
        metric_depth = raycast_ranges_to_image_plane(sensor.data.distances, self.camera)
        return preprocess_rp1_depth(metric_depth, self.cfg, enable_noise=self.enable_noise)

    def _sample_history(self, current_index: int) -> torch.Tensor:
        frame_indices = torch.remainder(
            current_index - self._frame_offsets[None, :] - self._delay_frames[:, None],
            self.cfg.history_length,
        )
        batch_indices = torch.arange(self.num_envs, device=self.device)[:, None]
        return self._history[batch_indices, frame_indices]

    def append_from_sensor(self, sensor) -> None:
        frame = self.current_frame(sensor)
        uninitialized = ~self._initialized
        if torch.any(uninitialized):
            self._history[uninitialized] = frame[uninitialized, None]
            self._initialized[uninitialized] = True
        current_index = self._write_index
        self._history[:, current_index].copy_(frame)
        self._latest = self._sample_history(current_index)
        self._write_index = (current_index + 1) % self.cfg.history_length

    def reset_from_sensor(self, sensor, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return
        frame = self.current_frame(sensor)
        self._delay_frames[env_ids] = torch.randint(
            self.cfg.delayed_frame_ranges[0],
            self.cfg.delayed_frame_ranges[1] + 1,
            (env_ids.numel(),),
            device=self.device,
        )
        self._history[env_ids] = frame[env_ids, None]
        self._latest[env_ids] = frame[env_ids, None]
        self._initialized[env_ids] = True

    def observation(self) -> torch.Tensor:
        if not torch.all(self._initialized):
            missing = torch.nonzero(~self._initialized, as_tuple=False).flatten().tolist()
            raise RuntimeError(f"direct-depth history is not initialized for envs {missing[:16]}")
        return self._latest


__all__ = [
    "BASELINE_TYPE",
    "REFERENCE_COMMIT",
    "REFERENCE_PROJECT",
    "RP1DirectDepthConfig",
    "RP1DirectDepthRuntime",
    "preprocess_rp1_depth",
    "raycast_ranges_to_image_plane",
]
