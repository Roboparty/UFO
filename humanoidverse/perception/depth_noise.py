"""Deterministic sensor-front-end perturbations for depth robustness evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F

from humanoidverse.perception.depth_camera import quaternion_multiply_xyzw, rotate_xyzw

NoiseCondition = Literal[
    "clean",
    "measurement",
    "dropout",
    "edge",
    "latency",
    "extrinsic",
    "combined",
]
NoiseSeverity = Literal["mild", "nominal", "strong"]


@dataclass(frozen=True)
class MeasurementNoiseConfig:
    base_std_m: float = 0.0
    quadratic_std_m_per_m2: float = 0.0


@dataclass(frozen=True)
class EdgeNoiseConfig:
    depth_threshold_m: float = 0.05
    corruption_probability: float = 0.0
    invalid_probability: float = 0.5


@dataclass(frozen=True)
class DropoutNoiseConfig:
    probability: float = 0.0


@dataclass(frozen=True)
class LatencyNoiseConfig:
    min_frames: int = 0
    max_frames: int = 0
    resample_each_frame: bool = True


@dataclass(frozen=True)
class ExtrinsicNoiseConfig:
    """Episode-static assumed-calibration error in the optical camera frame."""

    translation_bound_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_bound_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DepthNoiseConfig:
    condition: NoiseCondition = "clean"
    severity: NoiseSeverity = "nominal"
    max_depth_m: float = 2.5
    measurement: MeasurementNoiseConfig = field(default_factory=MeasurementNoiseConfig)
    edge: EdgeNoiseConfig = field(default_factory=EdgeNoiseConfig)
    dropout: DropoutNoiseConfig = field(default_factory=DropoutNoiseConfig)
    latency: LatencyNoiseConfig = field(default_factory=LatencyNoiseConfig)
    extrinsic: ExtrinsicNoiseConfig = field(default_factory=ExtrinsicNoiseConfig)

    def validate(self) -> None:
        if self.condition not in {"clean", "measurement", "dropout", "edge", "latency", "extrinsic", "combined"}:
            raise ValueError(f"unsupported depth-noise condition: {self.condition!r}")
        if self.severity not in {"mild", "nominal", "strong"}:
            raise ValueError(f"unsupported depth-noise severity: {self.severity!r}")
        if self.max_depth_m <= 0.0:
            raise ValueError("max_depth_m must be positive")
        if min(self.measurement.base_std_m, self.measurement.quadratic_std_m_per_m2) < 0.0:
            raise ValueError("measurement standard deviations must be non-negative")
        if self.edge.depth_threshold_m <= 0.0:
            raise ValueError("edge depth threshold must be positive")
        for name, probability in (
            ("edge corruption", self.edge.corruption_probability),
            ("edge invalid", self.edge.invalid_probability),
            ("dropout", self.dropout.probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} probability must lie in [0, 1]")
        if not 0 <= self.latency.min_frames <= self.latency.max_frames:
            raise ValueError("latency frames must satisfy 0 <= min <= max")
        if len(self.extrinsic.translation_bound_m) != 3 or len(self.extrinsic.rotation_bound_deg) != 3:
            raise ValueError("extrinsic translation and rotation bounds must have three components")
        if min(*self.extrinsic.translation_bound_m, *self.extrinsic.rotation_bound_deg) < 0.0:
            raise ValueError("extrinsic bounds must be non-negative")

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_identity(self) -> bool:
        return (
            self.measurement.base_std_m == 0.0
            and self.measurement.quadratic_std_m_per_m2 == 0.0
            and self.edge.corruption_probability == 0.0
            and self.dropout.probability == 0.0
            and self.latency.max_frames == 0
            and not any(self.extrinsic.translation_bound_m)
            and not any(self.extrinsic.rotation_bound_deg)
        )


def depth_noise_preset(
    condition: NoiseCondition,
    severity: NoiseSeverity = "nominal",
    *,
    max_depth_m: float = 2.5,
) -> DepthNoiseConfig:
    """Return monotonic evaluation presets, not claimed sensor calibration values."""
    level = {"mild": 0, "nominal": 1, "strong": 2}[severity]
    measurement = (
        MeasurementNoiseConfig(base_std_m=(0.0005, 0.0010, 0.0020)[level], quadratic_std_m_per_m2=(0.0005, 0.0010, 0.0020)[level])
        if condition in {"measurement", "combined"}
        else MeasurementNoiseConfig()
    )
    edge = (
        EdgeNoiseConfig(
            depth_threshold_m=0.05,
            corruption_probability=(0.10, 0.25, 0.50)[level],
            invalid_probability=(0.35, 0.50, 0.65)[level],
        )
        if condition in {"edge", "combined"}
        else EdgeNoiseConfig()
    )
    dropout = DropoutNoiseConfig(probability=(0.01, 0.03, 0.08)[level]) if condition in {"dropout", "combined"} else DropoutNoiseConfig()
    latency_levels = ((1, 1), (1, 2), (2, 3))
    latency = LatencyNoiseConfig(*latency_levels[level]) if condition in {"latency", "combined"} else LatencyNoiseConfig()
    translation = (0.003, 0.007, 0.010)[level]
    rotation = (0.2, 0.6, 1.0)[level]
    extrinsic = (
        ExtrinsicNoiseConfig(
            translation_bound_m=(translation, translation, translation),
            rotation_bound_deg=(rotation, rotation, rotation),
        )
        if condition in {"extrinsic", "combined"}
        else ExtrinsicNoiseConfig()
    )
    config = DepthNoiseConfig(
        condition=condition,
        severity=severity,
        max_depth_m=max_depth_m,
        measurement=measurement,
        edge=edge,
        dropout=dropout,
        latency=latency,
        extrinsic=extrinsic,
    )
    config.validate()
    return config


@dataclass
class NoisyDepthFrame:
    depth_z: torch.Tensor
    camera_pos_w: torch.Tensor
    camera_optical_quat_w: torch.Tensor
    timestamp_s: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class DepthNoisePipeline:
    """Apply image noise, synchronized latency, then assumed extrinsic error.

    Random samples are stateless functions of noise seed, environment id,
    episode counter, frame counter, component stream, and tensor element. This
    keeps one environment's sequence unchanged when another environment resets.
    """

    _MODULUS = 2_147_483_647

    def __init__(
        self,
        config: DepthNoiseConfig,
        *,
        batch_size: int,
        image_height: int,
        image_width: int,
        device: torch.device | str,
        noise_seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        config.validate()
        if min(batch_size, image_height, image_width) <= 0:
            raise ValueError("depth-noise dimensions must be positive")
        self.config = config
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.device = torch.device(device)
        self.dtype = dtype
        self.noise_seed = int(noise_seed)
        self.episode_count = torch.full((batch_size,), -1, device=self.device, dtype=torch.int64)
        self.frame_count = torch.zeros(batch_size, device=self.device, dtype=torch.int64)
        self.initialized = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        self._latency_frames = torch.zeros(batch_size, device=self.device, dtype=torch.int64)
        self._translation_error = torch.zeros((batch_size, 3), device=self.device, dtype=dtype)
        self._rotation_error = torch.zeros((batch_size, 4), device=self.device, dtype=dtype)
        self._rotation_error[:, 3] = 1.0
        queue_length = config.latency.max_frames + 1
        self._depth_queue = torch.full(
            (batch_size, queue_length, image_height, image_width),
            float("nan"),
            device=self.device,
            dtype=dtype,
        )
        self._position_queue = torch.zeros((batch_size, queue_length, 3), device=self.device, dtype=dtype)
        self._quaternion_queue = torch.zeros((batch_size, queue_length, 4), device=self.device, dtype=dtype)
        self._quaternion_queue[..., 3] = 1.0
        self._timestamp_queue = torch.zeros((batch_size, queue_length), device=self.device, dtype=dtype)

    def _validate_inputs(
        self,
        depth_z: torch.Tensor,
        camera_pos_w: torch.Tensor,
        camera_optical_quat_w: torch.Tensor,
        timestamp_s: torch.Tensor,
        env_ids: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if depth_z.shape != (self.batch_size, self.image_height, self.image_width):
            raise ValueError("depth_z has the wrong shape")
        if camera_pos_w.shape != (self.batch_size, 3) or camera_optical_quat_w.shape != (self.batch_size, 4):
            raise ValueError("camera pose has the wrong shape")
        if timestamp_s.shape != (self.batch_size,):
            raise ValueError("timestamp_s has the wrong shape")
        if env_ids.shape != (self.batch_size,) or reset_mask.shape != (self.batch_size,):
            raise ValueError("env_ids and reset_mask must have shape [B]")
        if reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must be bool")
        if not torch.isfinite(camera_pos_w).all() or not torch.isfinite(camera_optical_quat_w).all():
            raise ValueError("camera pose must be finite")
        if not torch.isfinite(timestamp_s).all():
            raise ValueError("timestamp must be finite")
        quaternion_norm = torch.linalg.vector_norm(camera_optical_quat_w, dim=-1)
        if torch.any(quaternion_norm <= 1.0e-8):
            raise ValueError("camera quaternion must be non-zero")
        env_ids = env_ids.to(device=self.device, dtype=torch.int64)
        reset_mask = reset_mask.to(device=self.device)
        if torch.any(~self.initialized & ~reset_mask):
            raise RuntimeError("every environment must be reset on its first depth-noise frame")
        return env_ids, reset_mask

    def _uniform(
        self,
        env_ids: torch.Tensor,
        shape: tuple[int, ...],
        *,
        stream: int,
        frame_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not shape or shape[0] != self.batch_size:
            raise ValueError("random tensor shape must start with batch_size")
        frames = self.frame_count if frame_count is None else frame_count
        base = (
            (self.noise_seed % self._MODULUS) + env_ids * 1_000_003 + self.episode_count * 97_409 + frames * 65_537 + int(stream) * 32_771
        ) % self._MODULUS
        elements_per_env = math.prod(shape[1:])
        element = torch.arange(elements_per_env, device=self.device, dtype=torch.int64)
        value = (base[:, None] + element[None, :] * 374_761_393) % self._MODULUS
        value = torch.bitwise_xor(value, value >> 13)
        value = (value * 1_274_126_177) % self._MODULUS
        return ((value.to(torch.float64) + 0.5) / self._MODULUS).to(self.dtype).reshape(shape)

    def _normal(self, env_ids: torch.Tensor, shape: tuple[int, ...], *, stream: int) -> torch.Tensor:
        first = self._uniform(env_ids, shape, stream=stream).clamp_min(1.0e-7)
        second = self._uniform(env_ids, shape, stream=stream + 1)
        return torch.sqrt(-2.0 * torch.log(first)) * torch.cos(2.0 * math.pi * second)

    def _sample_latency(self, env_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.config.latency
        if cfg.min_frames == cfg.max_frames:
            return torch.full_like(self._latency_frames, cfg.min_frames)
        uniform = self._uniform(env_ids, (self.batch_size, 1), stream=70).squeeze(1)
        return cfg.min_frames + torch.floor(uniform * (cfg.max_frames - cfg.min_frames + 1)).to(torch.int64)

    def _sample_extrinsic(self, env_ids: torch.Tensor, reset_mask: torch.Tensor) -> None:
        translation_bound = torch.tensor(self.config.extrinsic.translation_bound_m, device=self.device, dtype=self.dtype)
        rotation_bound = torch.tensor(self.config.extrinsic.rotation_bound_deg, device=self.device, dtype=self.dtype)
        translation = (2.0 * self._uniform(env_ids, (self.batch_size, 3), stream=80) - 1.0) * translation_bound
        euler = torch.deg2rad((2.0 * self._uniform(env_ids, (self.batch_size, 3), stream=90) - 1.0) * rotation_bound)
        half = euler * 0.5
        roll, pitch, yaw = half.unbind(dim=-1)
        cr, cp, cy = torch.cos(roll), torch.cos(pitch), torch.cos(yaw)
        sr, sp, sy = torch.sin(roll), torch.sin(pitch), torch.sin(yaw)
        quaternion = torch.stack(
            (
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ),
            dim=-1,
        )
        self._translation_error[reset_mask] = translation[reset_mask]
        self._rotation_error[reset_mask] = quaternion[reset_mask]

    def _apply_image_noise(self, clean: torch.Tensor, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        finite_clean = torch.isfinite(clean) & (clean > 0.0)
        noisy = clean.clone()
        measurement = self.config.measurement
        if measurement.base_std_m > 0.0 or measurement.quadratic_std_m_per_m2 > 0.0:
            sigma = measurement.base_std_m + measurement.quadratic_std_m_per_m2 * clean.nan_to_num().square()
            noise = self._normal(env_ids, clean.shape, stream=10) * sigma
            noisy = torch.where(finite_clean, noisy + noise, noisy)

        edge_cfg = self.config.edge
        edge_mask = torch.zeros_like(finite_clean)
        if edge_cfg.corruption_probability > 0.0:
            image = clean.unsqueeze(1)
            valid = finite_clean.unsqueeze(1)
            local_max = F.max_pool2d(torch.where(valid, image, -torch.inf), 3, stride=1, padding=1)
            local_min = -F.max_pool2d(torch.where(valid, -image, -torch.inf), 3, stride=1, padding=1)
            neighborhood_valid = torch.isfinite(local_min) & torch.isfinite(local_max)
            edge_mask = finite_clean & neighborhood_valid.squeeze(1) & ((local_max - local_min).squeeze(1) > edge_cfg.depth_threshold_m)
            corrupt = edge_mask & (self._uniform(env_ids, clean.shape, stream=20) < edge_cfg.corruption_probability)
            blend = self._uniform(env_ids, clean.shape, stream=21)
            flying = local_min.squeeze(1) + blend * (local_max - local_min).squeeze(1)
            make_invalid = self._uniform(env_ids, clean.shape, stream=22) < edge_cfg.invalid_probability
            noisy = torch.where(corrupt & ~make_invalid, flying, noisy)
            noisy = torch.where(corrupt & make_invalid, torch.full_like(noisy, float("nan")), noisy)

        if self.config.dropout.probability > 0.0:
            drop = finite_clean & (self._uniform(env_ids, clean.shape, stream=30) < self.config.dropout.probability)
            noisy = torch.where(drop, torch.full_like(noisy, float("nan")), noisy)
        valid_noisy = torch.isfinite(noisy) & (noisy > 0.0) & (noisy <= self.config.max_depth_m)
        noisy = torch.where(valid_noisy, noisy, torch.full_like(noisy, float("nan")))
        return noisy, edge_mask

    def __call__(
        self,
        *,
        depth_z: torch.Tensor,
        camera_pos_w: torch.Tensor,
        camera_optical_quat_w: torch.Tensor,
        timestamp_s: torch.Tensor,
        env_ids: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> NoisyDepthFrame:
        env_ids, reset_mask = self._validate_inputs(depth_z, camera_pos_w, camera_optical_quat_w, timestamp_s, env_ids, reset_mask)
        self.episode_count[reset_mask] += 1
        self.frame_count[reset_mask] = 0
        self.initialized[reset_mask] = True
        if self.config.is_identity():
            zero = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
            self.frame_count += 1
            return NoisyDepthFrame(
                depth_z=depth_z,
                camera_pos_w=camera_pos_w,
                camera_optical_quat_w=camera_optical_quat_w,
                timestamp_s=timestamp_s,
                diagnostics={
                    "clean_valid_fraction": torch.isfinite(depth_z).float().mean(dim=(1, 2)),
                    "noisy_valid_fraction": torch.isfinite(depth_z).float().mean(dim=(1, 2)),
                    "clean_edge_fraction": zero,
                    "latency_frames": zero,
                    "latency_seconds": zero,
                    "extrinsic_translation_norm_m": zero,
                    "extrinsic_rotation_deg": zero,
                },
            )
        self._sample_extrinsic(env_ids, reset_mask)
        if torch.any(reset_mask) or self.config.latency.resample_each_frame:
            sampled_latency = self._sample_latency(env_ids)
            update_latency = reset_mask if not self.config.latency.resample_each_frame else torch.ones_like(reset_mask)
            self._latency_frames[update_latency] = sampled_latency[update_latency]

        noisy_depth, clean_edge_mask = self._apply_image_noise(depth_z, env_ids)
        queues = (self._depth_queue, self._position_queue, self._quaternion_queue, self._timestamp_queue)
        for queue in queues:
            queue[:, :-1] = queue[:, 1:].clone()
        self._depth_queue[:, -1] = noisy_depth
        self._position_queue[:, -1] = camera_pos_w
        normalized_quaternion = camera_optical_quat_w / torch.linalg.vector_norm(camera_optical_quat_w, dim=-1, keepdim=True)
        self._quaternion_queue[:, -1] = normalized_quaternion
        self._timestamp_queue[:, -1] = timestamp_s
        for queue, current in (
            (self._depth_queue, noisy_depth),
            (self._position_queue, camera_pos_w),
            (self._quaternion_queue, normalized_quaternion),
            (self._timestamp_queue, timestamp_s),
        ):
            queue[reset_mask] = current[reset_mask, None]

        queue_length = self._depth_queue.shape[1]
        selected_index = queue_length - 1 - self._latency_frames
        batch_index = torch.arange(self.batch_size, device=self.device)
        delayed_depth = self._depth_queue[batch_index, selected_index]
        true_position = self._position_queue[batch_index, selected_index]
        true_quaternion = self._quaternion_queue[batch_index, selected_index]
        delayed_timestamp = self._timestamp_queue[batch_index, selected_index]

        assumed_position = true_position + rotate_xyzw(true_quaternion, self._translation_error)
        assumed_quaternion = quaternion_multiply_xyzw(true_quaternion, self._rotation_error)
        assumed_quaternion /= torch.linalg.vector_norm(assumed_quaternion, dim=-1, keepdim=True)
        diagnostics = {
            "clean_valid_fraction": torch.isfinite(depth_z).float().mean(dim=(1, 2)),
            "noisy_valid_fraction": torch.isfinite(delayed_depth).float().mean(dim=(1, 2)),
            "clean_edge_fraction": clean_edge_mask.float().mean(dim=(1, 2)),
            "latency_frames": self._latency_frames.to(self.dtype),
            "latency_seconds": timestamp_s - delayed_timestamp,
            "extrinsic_translation_norm_m": torch.linalg.vector_norm(self._translation_error, dim=-1),
            "extrinsic_rotation_deg": torch.rad2deg(2.0 * torch.acos(self._rotation_error[:, 3].abs().clamp(max=1.0))),
        }
        self.frame_count += 1
        return NoisyDepthFrame(
            depth_z=delayed_depth,
            camera_pos_w=assumed_position,
            camera_optical_quat_w=assumed_quaternion,
            timestamp_s=delayed_timestamp,
            diagnostics=diagnostics,
        )
