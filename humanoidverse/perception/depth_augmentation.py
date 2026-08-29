"""Metric-depth augmentation used for the sim-to-real TemporalCompletion run."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from humanoidverse.perception.depth_camera import quaternion_multiply_xyzw


@dataclass(frozen=True)
class MetricDepthAugmentationConfig:
    """Raw metric-depth DR followed by a per-frame range gate.

    Zero-valued DR fields preserve the Phase-2I v1 behavior.  All image noise
    is applied before ``max_depth_m`` gating so a noisy sample can move into or
    out of range exactly as it can on hardware.
    """

    max_depth_m: float = 2.0
    max_depth_jitter_m: float = 0.0
    blur_probability: float = 0.5
    sigma_min_px: float = 0.0
    sigma_max_px: float = 3.0
    measurement_base_std_m: float = 0.0
    measurement_quadratic_std_m_per_m2: float = 0.0
    edge_depth_threshold_m: float = 0.04
    edge_corruption_probability: float = 0.0
    edge_invalid_probability: float = 0.0
    pixel_dropout_probability: float = 0.0
    region_dropout_probability: float = 0.0
    region_dropout_min_height_fraction: float = 0.03
    region_dropout_max_height_fraction: float = 0.15
    region_dropout_min_width_fraction: float = 0.03
    region_dropout_max_width_fraction: float = 0.15

    def validate(self) -> None:
        if self.max_depth_m <= 0.0:
            raise ValueError("max_depth_m must be positive")
        if not 0.0 <= self.max_depth_jitter_m < self.max_depth_m:
            raise ValueError("max_depth_jitter_m must lie in [0, max_depth_m)")
        if not 0.0 <= self.blur_probability <= 1.0:
            raise ValueError("blur_probability must lie in [0, 1]")
        if self.sigma_min_px < 0.0 or self.sigma_max_px < self.sigma_min_px:
            raise ValueError("sigma range must be non-negative and ordered")
        if self.measurement_base_std_m < 0.0 or self.measurement_quadratic_std_m_per_m2 < 0.0:
            raise ValueError("measurement-noise scales must be non-negative")
        if self.edge_depth_threshold_m <= 0.0:
            raise ValueError("edge_depth_threshold_m must be positive")
        probabilities = (
            self.edge_corruption_probability,
            self.edge_invalid_probability,
            self.pixel_dropout_probability,
            self.region_dropout_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("depth-noise probabilities must lie in [0, 1]")
        for minimum, maximum, name in (
            (
                self.region_dropout_min_height_fraction,
                self.region_dropout_max_height_fraction,
                "region height",
            ),
            (
                self.region_dropout_min_width_fraction,
                self.region_dropout_max_width_fraction,
                "region width",
            ),
        ):
            if not 0.0 < minimum <= maximum <= 1.0:
                raise ValueError(f"{name} fractions must satisfy 0 < min <= max <= 1")


@dataclass(frozen=True)
class DepthTimingAugmentationConfig:
    """Camera/control asynchrony used by Phase-2I v2 simulation only."""

    camera_frequency_hz: float = 50.0
    control_frequency_hz: float = 50.0
    frame_drop_probability: float = 0.0
    duplicate_frame_probability: float = 0.0
    timestamp_jitter_s: float = 0.0

    def validate(self) -> None:
        if self.camera_frequency_hz <= 0.0 or self.control_frequency_hz <= 0.0:
            raise ValueError("camera and control frequencies must be positive")
        if self.camera_frequency_hz > self.control_frequency_hz:
            raise ValueError("camera_frequency_hz cannot exceed the control frequency")
        if not 0.0 <= self.frame_drop_probability <= 1.0:
            raise ValueError("frame_drop_probability must lie in [0, 1]")
        if not 0.0 <= self.duplicate_frame_probability <= 1.0:
            raise ValueError("duplicate_frame_probability must lie in [0, 1]")
        if self.frame_drop_probability + self.duplicate_frame_probability > 1.0:
            raise ValueError("drop and duplicate probabilities cannot sum above one")
        if self.timestamp_jitter_s < 0.0:
            raise ValueError("timestamp_jitter_s must be non-negative")


@dataclass(frozen=True)
class DepthCalibrationAugmentationConfig:
    """Episode-static assumed camera-calibration error in local projection."""

    focal_scale_bound: float = 0.0
    principal_point_bound_px: tuple[float, float] = (0.0, 0.0)
    translation_bound_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_bound_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        if self.focal_scale_bound < 0.0 or self.focal_scale_bound >= 1.0:
            raise ValueError("focal_scale_bound must lie in [0, 1)")
        if len(self.principal_point_bound_px) != 2:
            raise ValueError("principal_point_bound_px must contain two values")
        if len(self.translation_bound_m) != 3 or len(self.rotation_bound_deg) != 3:
            raise ValueError("extrinsic calibration bounds must contain three values")
        if min(
            *self.principal_point_bound_px,
            *self.translation_bound_m,
            *self.rotation_bound_deg,
        ) < 0.0:
            raise ValueError("calibration perturbation bounds must be non-negative")


class LocalCalibrationAugmentation:
    """Maintain independently perturbed assumed calibration for each episode."""

    def __init__(
        self,
        config: DepthCalibrationAugmentationConfig,
        *,
        intrinsic_matrix: torch.Tensor,
        camera_pos_torso: tuple[float, float, float],
        camera_optical_quat_torso_xyzw: tuple[float, float, float, float],
        batch_size: int,
        device: torch.device | str,
        seed: int,
    ) -> None:
        config.validate()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.config = config
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        intrinsic = torch.as_tensor(intrinsic_matrix, device=self.device, dtype=torch.float32)
        position = torch.as_tensor(camera_pos_torso, device=self.device, dtype=torch.float32)
        quaternion = torch.as_tensor(camera_optical_quat_torso_xyzw, device=self.device, dtype=torch.float32)
        if intrinsic.shape != (3, 3) or position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("invalid base calibration shapes")
        self.base_intrinsics = intrinsic
        self.base_position = position
        self.base_quaternion = quaternion / torch.linalg.vector_norm(quaternion)
        self.intrinsics = intrinsic.expand(batch_size, -1, -1).clone()
        self.camera_pos_torso = position.expand(batch_size, -1).clone()
        self.camera_quat_torso = self.base_quaternion.expand(batch_size, -1).clone()
        self.reset(torch.ones(batch_size, device=self.device, dtype=torch.bool))

    def reset(self, reset_mask: torch.Tensor) -> None:
        reset_mask = torch.as_tensor(reset_mask, device=self.device, dtype=torch.bool)
        if reset_mask.shape != (self.batch_size,):
            raise ValueError("reset_mask must be bool with shape [B]")
        count = int(reset_mask.sum())
        if count == 0:
            return
        uniform = lambda shape: 2.0 * torch.rand(  # noqa: E731 - compact independent draws
            shape,
            device=self.device,
            generator=self.generator,
        ) - 1.0
        intrinsics = self.base_intrinsics.expand(count, -1, -1).clone()
        focal_error = uniform((count, 2)) * self.config.focal_scale_bound
        intrinsics[:, 0, 0] *= 1.0 + focal_error[:, 0]
        intrinsics[:, 1, 1] *= 1.0 + focal_error[:, 1]
        principal_bound = torch.tensor(self.config.principal_point_bound_px, device=self.device)
        principal_error = uniform((count, 2)) * principal_bound
        intrinsics[:, 0, 2] += principal_error[:, 0]
        intrinsics[:, 1, 2] += principal_error[:, 1]

        translation_bound = torch.tensor(self.config.translation_bound_m, device=self.device)
        position = self.base_position + uniform((count, 3)) * translation_bound
        rotation_bound = torch.deg2rad(torch.tensor(self.config.rotation_bound_deg, device=self.device))
        roll, pitch, yaw = (uniform((count, 3)) * rotation_bound).unbind(dim=-1)
        half_roll, half_pitch, half_yaw = roll / 2.0, pitch / 2.0, yaw / 2.0
        cr, cp, cy = torch.cos(half_roll), torch.cos(half_pitch), torch.cos(half_yaw)
        sr, sp, sy = torch.sin(half_roll), torch.sin(half_pitch), torch.sin(half_yaw)
        error_quaternion = torch.stack(
            (
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ),
            dim=-1,
        )
        quaternion = quaternion_multiply_xyzw(
            self.base_quaternion.expand(count, -1),
            error_quaternion,
        )
        quaternion /= torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        self.intrinsics[reset_mask] = intrinsics
        self.camera_pos_torso[reset_mask] = position
        self.camera_quat_torso[reset_mask] = quaternion


class CameraFrameScheduler:
    """Emit per-environment fresh-frame masks on a faster control clock.

    Dropped and duplicated camera packets both become ``frame_valid=False``;
    this matches deployment, where repeated frame numbers are not appended to
    the temporal history.  A reset always forces one fresh first frame.
    """

    def __init__(
        self,
        config: DepthTimingAugmentationConfig,
        *,
        batch_size: int,
        device: torch.device | str,
        seed: int,
    ) -> None:
        config.validate()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.config = config
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.phase = torch.zeros(batch_size, device=self.device)
        self.force_capture = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        self.last_timestamp = torch.full((batch_size,), -1.0e9, device=self.device)

    def reset(self, reset_mask: torch.Tensor) -> None:
        reset_mask = torch.as_tensor(reset_mask, device=self.device, dtype=torch.bool)
        if reset_mask.shape != (self.batch_size,):
            raise ValueError("reset_mask must be bool with shape [B]")
        self.phase[reset_mask] = 0.0
        self.force_capture[reset_mask] = True
        self.last_timestamp[reset_mask] = -1.0e9

    def step(self, control_timestamp_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        control_timestamp_s = torch.as_tensor(control_timestamp_s, device=self.device, dtype=torch.float32)
        if control_timestamp_s.shape != (self.batch_size,) or not torch.isfinite(control_timestamp_s).all():
            raise ValueError("control_timestamp_s must be finite with shape [B]")
        ratio = self.config.camera_frequency_hz / self.config.control_frequency_hz
        self.phase += ratio
        scheduled = self.phase >= 1.0 - 1.0e-7
        self.phase[scheduled] -= 1.0
        forced = self.force_capture.clone()
        scheduled |= forced
        self.phase[forced] = 0.0
        self.force_capture.zero_()

        random = torch.rand(self.batch_size, device=self.device, generator=self.generator)
        dropped = scheduled & (random < self.config.frame_drop_probability)
        duplicated = scheduled & (
            random >= self.config.frame_drop_probability
        ) & (
            random < self.config.frame_drop_probability + self.config.duplicate_frame_probability
        )
        frame_valid = scheduled & ~dropped & ~duplicated

        if self.config.timestamp_jitter_s > 0.0:
            jitter = 2.0 * torch.rand(
                self.batch_size,
                device=self.device,
                generator=self.generator,
            ) - 1.0
            timestamp = control_timestamp_s + jitter * self.config.timestamp_jitter_s
        else:
            timestamp = control_timestamp_s.clone()
        # Keep the serialized control-slot timestamps chronological.  Invalid
        # slots are masked before the estimator sees their ages.
        timestamp = torch.maximum(timestamp, self.last_timestamp + 1.0e-6)
        self.last_timestamp = timestamp
        return frame_valid, timestamp, {
            "scheduled": scheduled,
            "dropped": dropped,
            "duplicated": duplicated,
        }


def phase2i_v2_depth_augmentation_config() -> MetricDepthAugmentationConfig:
    """Nominal D435i-like training DR; deployment must not apply this preset."""
    return MetricDepthAugmentationConfig(
        max_depth_m=2.0,
        max_depth_jitter_m=0.05,
        blur_probability=1.0,
        sigma_min_px=0.0,
        sigma_max_px=3.0,
        measurement_base_std_m=0.002,
        measurement_quadratic_std_m_per_m2=0.001,
        edge_depth_threshold_m=0.04,
        edge_corruption_probability=0.20,
        edge_invalid_probability=0.35,
        pixel_dropout_probability=0.01,
        region_dropout_probability=0.15,
        region_dropout_min_height_fraction=0.03,
        region_dropout_max_height_fraction=0.12,
        region_dropout_min_width_fraction=0.03,
        region_dropout_max_width_fraction=0.12,
    )


def phase2i_v1_depth_augmentation_config() -> MetricDepthAugmentationConfig:
    """Original Phase-2I image model for controlled v2 ablations."""
    return MetricDepthAugmentationConfig(
        max_depth_m=2.0,
        blur_probability=1.0,
        sigma_min_px=0.0,
        sigma_max_px=3.0,
    )


def deployment_clean_depth_augmentation_config() -> MetricDepthAugmentationConfig:
    """Deterministic preprocessing used by real deployment and clean sim."""
    return MetricDepthAugmentationConfig(
        max_depth_m=2.0,
        blur_probability=1.0,
        sigma_min_px=1.5,
        sigma_max_px=1.5,
    )


def phase2i_v2_timing_augmentation_config() -> DepthTimingAugmentationConfig:
    """30 Hz camera timing model for a 50 Hz policy/control loop."""
    return DepthTimingAugmentationConfig(
        camera_frequency_hz=30.0,
        control_frequency_hz=50.0,
        frame_drop_probability=0.03,
        duplicate_frame_probability=0.03,
        timestamp_jitter_s=0.003,
    )


def phase2i_v1_timing_augmentation_config() -> DepthTimingAugmentationConfig:
    """Original synchronous camera/control timing for controlled ablations."""
    return DepthTimingAugmentationConfig(
        camera_frequency_hz=50.0,
        control_frequency_hz=50.0,
    )


def deployment_clean_timing_config() -> DepthTimingAugmentationConfig:
    """Nominal 30 Hz camera on the 50 Hz control loop without failures."""
    return DepthTimingAugmentationConfig(
        camera_frequency_hz=30.0,
        control_frequency_hz=50.0,
    )


def phase2i_v2_calibration_augmentation_config() -> DepthCalibrationAugmentationConfig:
    """Conservative episode-static D435i mounting/calibration DR."""
    return DepthCalibrationAugmentationConfig(
        focal_scale_bound=0.01,
        principal_point_bound_px=(0.5, 0.5),
        translation_bound_m=(0.005, 0.005, 0.005),
        rotation_bound_deg=(0.5, 0.5, 0.5),
    )


class MetricDepthAugmentation:
    """Apply depth-domain augmentation without expanding the valid mask.

    The random generator is CPU-backed so the same seed produces the same
    augmentation stream independently of CUDA kernel scheduling.
    """

    def __init__(self, config: MetricDepthAugmentationConfig, *, seed: int = 0) -> None:
        config.validate()
        self.config = config
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self._device_generators: dict[str, torch.Generator] = {}

    def _generator_for(self, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self._device_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(self.seed + 1_000_003)
            self._device_generators[key] = generator
        return generator

    def sample_max_depth(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        jitter = self.config.max_depth_jitter_m
        if jitter <= 0.0:
            return torch.full((batch_size,), self.config.max_depth_m, device=device, dtype=dtype)
        values = torch.rand(batch_size, generator=self.generator, dtype=torch.float32)
        values = self.config.max_depth_m + (2.0 * values - 1.0) * jitter
        return values.to(device=device, dtype=dtype)

    def sample_sigma(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        config = self.config
        select = torch.rand(batch_size, generator=self.generator, dtype=torch.float32)
        magnitude = torch.rand(batch_size, generator=self.generator, dtype=torch.float32)
        sigma = config.sigma_min_px + (config.sigma_max_px - config.sigma_min_px) * magnitude
        sigma = torch.where(select < config.blur_probability, sigma, torch.zeros_like(sigma))
        return sigma.to(device=device, dtype=dtype)

    def apply_to_valid_depth(
        self,
        depth_m: torch.Tensor,
        valid: torch.Tensor,
        *,
        sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply DR only to caller-approved depth without expanding validity."""
        depth_m = torch.as_tensor(depth_m)
        valid = torch.as_tensor(valid, device=depth_m.device, dtype=torch.bool)
        if depth_m.ndim != 3 or valid.shape != depth_m.shape:
            raise ValueError("depth_m and valid must have matching [B, H, W] shapes")
        if not depth_m.is_floating_point():
            depth_m = depth_m.to(torch.float32)
        valid = valid & torch.isfinite(depth_m) & (depth_m > 0.0)
        if sigma is None:
            sigma = self.sample_sigma(depth_m.shape[0], device=depth_m.device, dtype=depth_m.dtype)
        else:
            sigma = torch.as_tensor(sigma, device=depth_m.device, dtype=depth_m.dtype)
            if sigma.shape != (depth_m.shape[0],):
                raise ValueError(f"sigma must have shape [{depth_m.shape[0]}]")
            if not torch.isfinite(sigma).all() or torch.any(sigma < 0.0):
                raise ValueError("sigma must be finite and non-negative")
        output = torch.where(valid, depth_m, torch.full_like(depth_m, float("nan")))

        measurement_std = (
            self.config.measurement_base_std_m
            + self.config.measurement_quadratic_std_m_per_m2 * output.nan_to_num().square()
        )
        if torch.any(measurement_std > 0.0):
            noise = torch.randn(
                output.shape,
                device=output.device,
                dtype=output.dtype,
                generator=self._generator_for(output.device),
            )
            output = torch.where(valid, output + measurement_std * noise, output)

        if self.config.sigma_max_px > 0.0 and torch.any(sigma > 1.0e-6):
            radius = max(1, int(torch.ceil(torch.tensor(3.0 * self.config.sigma_max_px)).item()))
            coordinates = torch.arange(-radius, radius + 1, device=depth_m.device, dtype=depth_m.dtype)
            yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
            squared_distance = xx.square() + yy.square()
            sigma_safe = sigma.clamp_min(1.0e-6).reshape(-1, 1, 1, 1)
            kernel = torch.exp(-squared_distance.reshape(1, 1, *squared_distance.shape) / (2.0 * sigma_safe.square()))
            kernel = kernel / kernel.sum(dim=(-1, -2), keepdim=True).clamp_min(1.0e-12)
            # Grouped convolution applies one sigma-specific kernel to each batch item.
            values = torch.where(valid, output, torch.zeros_like(output)).unsqueeze(0)
            weights = valid.to(depth_m.dtype).unsqueeze(0)
            blurred_values = F.conv2d(values, kernel, padding=radius, groups=depth_m.shape[0]).squeeze(0)
            blurred_weights = F.conv2d(weights, kernel, padding=radius, groups=depth_m.shape[0]).squeeze(0)
            blurred = blurred_values / blurred_weights.clamp_min(1.0e-8)
            output = torch.where(valid, blurred, torch.full_like(blurred, float("nan")))

        edge_probability = self.config.edge_corruption_probability
        if edge_probability > 0.0:
            finite_values = torch.where(valid, output, torch.zeros_like(output)).unsqueeze(1)
            local_max = F.max_pool2d(finite_values, 3, stride=1, padding=1).squeeze(1)
            large = torch.where(valid, output, torch.full_like(output, 1.0e6)).neg().unsqueeze(1)
            local_min = -F.max_pool2d(large, 3, stride=1, padding=1).squeeze(1)
            neighborhood_valid = F.avg_pool2d(valid.float().unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.5
            edge = valid & neighborhood_valid & ((local_max - local_min) > self.config.edge_depth_threshold_m)
            random = torch.rand(
                output.shape,
                device=output.device,
                dtype=output.dtype,
                generator=self._generator_for(output.device),
            )
            corrupt = edge & (random < edge_probability)
            choose_far = torch.rand(
                output.shape,
                device=output.device,
                dtype=output.dtype,
                generator=self._generator_for(output.device),
            ) < 0.5
            flying = torch.where(choose_far, local_max, local_min)
            output = torch.where(corrupt, flying, output)
            invalidate = corrupt & (
                torch.rand(
                    output.shape,
                    device=output.device,
                    dtype=output.dtype,
                    generator=self._generator_for(output.device),
                )
                < self.config.edge_invalid_probability
            )
            valid &= ~invalidate

        if self.config.pixel_dropout_probability > 0.0:
            dropout = torch.rand(
                output.shape,
                device=output.device,
                dtype=output.dtype,
                generator=self._generator_for(output.device),
            ) < self.config.pixel_dropout_probability
            valid &= ~dropout

        if self.config.region_dropout_probability > 0.0:
            batch_size, height, width = output.shape
            selected = torch.rand(batch_size, generator=self.generator) < self.config.region_dropout_probability
            height_fraction = torch.rand(batch_size, generator=self.generator)
            width_fraction = torch.rand(batch_size, generator=self.generator)
            center_y = torch.rand(batch_size, generator=self.generator)
            center_x = torch.rand(batch_size, generator=self.generator)
            for index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
                fraction_h = self.config.region_dropout_min_height_fraction + float(height_fraction[index]) * (
                    self.config.region_dropout_max_height_fraction - self.config.region_dropout_min_height_fraction
                )
                fraction_w = self.config.region_dropout_min_width_fraction + float(width_fraction[index]) * (
                    self.config.region_dropout_max_width_fraction - self.config.region_dropout_min_width_fraction
                )
                region_h = max(1, round(height * fraction_h))
                region_w = max(1, round(width * fraction_w))
                start_y = min(height - region_h, max(0, round(float(center_y[index]) * height - region_h / 2)))
                start_x = min(width - region_w, max(0, round(float(center_x[index]) * width - region_w / 2)))
                valid[index, start_y : start_y + region_h, start_x : start_x + region_w] = False

        max_depth = self.sample_max_depth(
            output.shape[0],
            device=output.device,
            dtype=output.dtype,
        ).reshape(-1, 1, 1)
        valid &= torch.isfinite(output) & (output > 0.0) & (output <= max_depth)
        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output, valid, sigma

    def __call__(self, depth_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(augmented_depth_m, original_valid_mask, sigma_px)``.

        Invalid pixels remain invalid after blur. In particular, this is not
        an inpainting operation and cannot increase visibility.
        """
        depth_m = torch.as_tensor(depth_m)
        if depth_m.ndim != 3:
            raise ValueError("depth_m must have shape [B, H, W]")
        if not depth_m.is_floating_point():
            depth_m = depth_m.to(torch.float32)
        valid = torch.isfinite(depth_m) & (depth_m > 0.0)
        sigma = self.sample_sigma(depth_m.shape[0], device=depth_m.device, dtype=depth_m.dtype)
        return self.apply_to_valid_depth(depth_m, valid, sigma=sigma)
