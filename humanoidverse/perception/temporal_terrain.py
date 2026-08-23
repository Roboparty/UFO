"""Temporal completion of partial PBFM terrain maps in robot-heading space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter


def _rotation_2d(yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(*yaw.shape, 2, 2)


def _validate_history_inputs(
    partial_maps: torch.Tensor,
    visible_masks: torch.Tensor,
    pelvis_pos_w: torch.Tensor,
    heading_yaw_w: torch.Tensor,
) -> tuple[int, int]:
    if partial_maps.ndim != 3 or partial_maps.shape[-1] != DepthTerrainAdapter.GRID_DIMENSION:
        raise ValueError("partial_maps must have shape [B, T, 273]")
    if visible_masks.shape != partial_maps.shape or visible_masks.dtype != torch.bool:
        raise ValueError("visible_masks must be bool and match partial_maps")
    batch_size, time_steps = partial_maps.shape[:2]
    if pelvis_pos_w.shape != (batch_size, time_steps, 3):
        raise ValueError("pelvis_pos_w must have shape [B, T, 3]")
    if heading_yaw_w.shape != (batch_size, time_steps):
        raise ValueError("heading_yaw_w must have shape [B, T]")
    if not torch.isfinite(pelvis_pos_w).all() or not torch.isfinite(heading_yaw_w).all():
        raise ValueError("pelvis poses and heading yaw must be finite")
    if torch.any(visible_masks & ~torch.isfinite(partial_maps)):
        raise ValueError("visible partial-map cells must contain finite clearance")
    return batch_size, time_steps


@dataclass
class WarpedTerrainHistory:
    """Past partial maps expressed in the final frame's heading coordinates."""

    clearances: torch.Tensor
    visible_masks: torch.Tensor
    motion_features: torch.Tensor


class TerrainHistoryBuffer:
    """Fixed-length per-environment history with explicit reset invalidation."""

    def __init__(
        self,
        *,
        batch_size: int,
        time_steps: int,
        proprio_dim: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if min(batch_size, time_steps) <= 0 or proprio_dim < 0:
            raise ValueError("invalid terrain history dimensions")
        shape = (batch_size, time_steps)
        self.partial_maps = torch.full(
            (*shape, DepthTerrainAdapter.GRID_DIMENSION),
            float("nan"),
            device=device,
            dtype=dtype,
        )
        self.visible_masks = torch.zeros_like(self.partial_maps, dtype=torch.bool)
        self.pelvis_pos_w = torch.zeros((*shape, 3), device=device, dtype=dtype)
        self.heading_yaw_w = torch.zeros(shape, device=device, dtype=dtype)
        self.timestamps_s = torch.zeros(shape, device=device, dtype=dtype)
        self.proprio = torch.zeros((*shape, proprio_dim), device=device, dtype=dtype)
        self.frame_valid = torch.zeros(shape, device=device, dtype=torch.bool)

    @property
    def batch_size(self) -> int:
        return self.partial_maps.shape[0]

    @property
    def time_steps(self) -> int:
        return self.partial_maps.shape[1]

    def reset(self, reset_mask: torch.Tensor) -> None:
        if reset_mask.shape != (self.batch_size,) or reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must be bool with shape [B]")
        reset_mask = reset_mask.to(device=self.partial_maps.device)
        self.partial_maps[reset_mask] = float("nan")
        self.visible_masks[reset_mask] = False
        self.pelvis_pos_w[reset_mask] = 0.0
        self.heading_yaw_w[reset_mask] = 0.0
        self.timestamps_s[reset_mask] = 0.0
        self.proprio[reset_mask] = 0.0
        self.frame_valid[reset_mask] = False

    def append(
        self,
        *,
        partial_map: torch.Tensor,
        visible_mask: torch.Tensor,
        pelvis_pos_w: torch.Tensor,
        heading_yaw_w: torch.Tensor,
        timestamp_s: torch.Tensor,
        proprio: torch.Tensor,
    ) -> None:
        expected_map = (self.batch_size, DepthTerrainAdapter.GRID_DIMENSION)
        if partial_map.shape != expected_map or visible_mask.shape != expected_map:
            raise ValueError("partial_map and visible_mask must have shape [B, 273]")
        if visible_mask.dtype != torch.bool:
            raise ValueError("visible_mask must be bool")
        if pelvis_pos_w.shape != (self.batch_size, 3):
            raise ValueError("pelvis_pos_w must have shape [B, 3]")
        if heading_yaw_w.shape != (self.batch_size,) or timestamp_s.shape != (self.batch_size,):
            raise ValueError("heading_yaw_w and timestamp_s must have shape [B]")
        if proprio.shape != (self.batch_size, self.proprio.shape[-1]):
            raise ValueError("proprio has the wrong shape")
        for value in (
            self.partial_maps,
            self.visible_masks,
            self.pelvis_pos_w,
            self.heading_yaw_w,
            self.timestamps_s,
            self.proprio,
            self.frame_valid,
        ):
            value[:, :-1] = value[:, 1:].clone()
        self.partial_maps[:, -1] = partial_map
        self.visible_masks[:, -1] = visible_mask
        self.pelvis_pos_w[:, -1] = pelvis_pos_w
        self.heading_yaw_w[:, -1] = heading_yaw_w
        self.timestamps_s[:, -1] = timestamp_s
        self.proprio[:, -1] = proprio
        self.frame_valid[:, -1] = True

    def single_frame_view(self) -> "TerrainHistoryBuffer":
        """Return an identical-shape history whose only valid frame is current."""
        result = TerrainHistoryBuffer(
            batch_size=self.batch_size,
            time_steps=self.time_steps,
            proprio_dim=self.proprio.shape[-1],
            device=self.partial_maps.device,
            dtype=self.partial_maps.dtype,
        )
        result.partial_maps[:, -1] = self.partial_maps[:, -1]
        result.visible_masks[:, -1] = self.visible_masks[:, -1]
        result.pelvis_pos_w[:, -1] = self.pelvis_pos_w[:, -1]
        result.heading_yaw_w[:, -1] = self.heading_yaw_w[:, -1]
        result.timestamps_s[:, -1] = self.timestamps_s[:, -1]
        result.proprio[:, -1] = self.proprio[:, -1]
        result.frame_valid[:, -1] = self.frame_valid[:, -1]
        return result

    def warp(self, *, history_seconds: float, interpolation: str = "bilinear") -> WarpedTerrainHistory:
        return warp_terrain_history_to_current(
            self.partial_maps,
            self.visible_masks,
            self.pelvis_pos_w,
            self.heading_yaw_w,
            timestamps_s=self.timestamps_s,
            frame_valid=self.frame_valid,
            history_seconds=history_seconds,
            interpolation=interpolation,
        )


def warp_terrain_history_to_current(
    partial_maps: torch.Tensor,
    visible_masks: torch.Tensor,
    pelvis_pos_w: torch.Tensor,
    heading_yaw_w: torch.Tensor,
    *,
    timestamps_s: torch.Tensor | None = None,
    frame_valid: torch.Tensor | None = None,
    history_seconds: float = 0.6,
    interpolation: str = "nearest",
) -> WarpedTerrainHistory:
    """Warp a partial-map sequence into its last frame using pelvis egomotion.

    ``partial_maps`` stores pelvis-to-terrain clearance. Warping therefore
    transforms XY by the source/target heading poses and adds the target-source
    pelvis world-Z difference to each valid clearance.
    """
    batch_size, time_steps = _validate_history_inputs(
        partial_maps,
        visible_masks,
        pelvis_pos_w,
        heading_yaw_w,
    )
    if history_seconds <= 0.0:
        raise ValueError("history_seconds must be positive")
    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")

    device, dtype = partial_maps.device, partial_maps.dtype
    if timestamps_s is None:
        timestamps_s = torch.arange(time_steps, device=device, dtype=dtype).expand(batch_size, -1)
        timestamps_s = timestamps_s * (history_seconds / max(time_steps - 1, 1))
    elif timestamps_s.shape != (batch_size, time_steps) or not torch.isfinite(timestamps_s).all():
        raise ValueError("timestamps_s must be finite with shape [B, T]")
    else:
        timestamps_s = timestamps_s.to(device=device, dtype=dtype)
    ages = timestamps_s[:, -1:] - timestamps_s
    chronological = ages >= -1.0e-6
    within_window = ages <= history_seconds + 1.0e-6
    if frame_valid is None:
        frame_valid = torch.ones((batch_size, time_steps), device=device, dtype=torch.bool)
    elif frame_valid.shape != (batch_size, time_steps) or frame_valid.dtype != torch.bool:
        raise ValueError("frame_valid must be bool with shape [B, T]")
    else:
        frame_valid = frame_valid.to(device=device)
    frame_valid = frame_valid & chronological & within_window

    grid_x, grid_y = torch.meshgrid(
        torch.linspace(
            DepthTerrainAdapter.X_MIN,
            DepthTerrainAdapter.X_MAX,
            DepthTerrainAdapter.GRID_SHAPE[0],
            device=device,
            dtype=dtype,
        ),
        torch.linspace(
            DepthTerrainAdapter.Y_MIN,
            DepthTerrainAdapter.Y_MAX,
            DepthTerrainAdapter.GRID_SHAPE[1],
            device=device,
            dtype=dtype,
        ),
        indexing="ij",
    )
    current_grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    source_rotation = _rotation_2d(heading_yaw_w)
    target_rotation = source_rotation[:, -1]
    target_xy = pelvis_pos_w[:, -1, :2]

    current_in_world = torch.einsum("bij,nj->bni", target_rotation, current_grid)
    source_to_target = target_xy[:, None, :] - pelvis_pos_w[:, :, :2]
    relative_world = current_in_world[:, None, :, :] + source_to_target[:, :, None, :]
    source_coordinates = torch.einsum(
        "btji,btnj->btni",
        source_rotation,
        relative_world,
    )
    normalized_x = (
        2.0 * ((source_coordinates[..., 0] - DepthTerrainAdapter.X_MIN) / (DepthTerrainAdapter.X_MAX - DepthTerrainAdapter.X_MIN)) - 1.0
    )
    normalized_y = (
        2.0 * ((source_coordinates[..., 1] - DepthTerrainAdapter.Y_MIN) / (DepthTerrainAdapter.Y_MAX - DepthTerrainAdapter.Y_MIN)) - 1.0
    )
    sampling_grid = torch.stack((normalized_x, normalized_y), dim=-1)
    sampling_grid = sampling_grid.reshape(
        batch_size * time_steps,
        *DepthTerrainAdapter.GRID_SHAPE,
        2,
    ).transpose(1, 2)

    valid_cells = visible_masks & torch.isfinite(partial_maps) & frame_valid.unsqueeze(-1)
    values = torch.where(valid_cells, partial_maps, torch.zeros_like(partial_maps))
    values = values.reshape(batch_size * time_steps, 1, *DepthTerrainAdapter.GRID_SHAPE).transpose(2, 3)
    weights = (
        valid_cells.to(dtype)
        .reshape(
            batch_size * time_steps,
            1,
            *DepthTerrainAdapter.GRID_SHAPE,
        )
        .transpose(2, 3)
    )
    sampled_values = F.grid_sample(
        values,
        sampling_grid,
        mode=interpolation,
        padding_mode="zeros",
        align_corners=True,
    )
    sampled_weights = F.grid_sample(
        weights,
        sampling_grid,
        mode=interpolation,
        padding_mode="zeros",
        align_corners=True,
    )
    weight_threshold = 0.5 if interpolation == "nearest" else 1.0e-6
    warped_visible = sampled_weights > weight_threshold
    warped = sampled_values / sampled_weights.clamp_min(1.0e-6)
    height_delta = pelvis_pos_w[:, -1:, 2] - pelvis_pos_w[:, :, 2]
    warped = warped + height_delta.reshape(batch_size * time_steps, 1, 1, 1)
    warped = warped.clamp_min(0.0)
    warped = torch.where(warped_visible, warped, torch.full_like(warped, float("nan")))
    warped = warped.transpose(2, 3).reshape(batch_size, time_steps, -1)
    warped_visible = warped_visible.transpose(2, 3).reshape(batch_size, time_steps, -1)

    delta_world = pelvis_pos_w[:, :, :2] - target_xy[:, None, :]
    delta_current = torch.einsum("bji,btj->bti", target_rotation, delta_world)
    delta_yaw = heading_yaw_w - heading_yaw_w[:, -1:]
    motion_features = torch.stack(
        (
            delta_current[..., 0],
            delta_current[..., 1],
            pelvis_pos_w[:, :, 2] - pelvis_pos_w[:, -1:, 2],
            torch.sin(delta_yaw),
            torch.cos(delta_yaw),
            ages,
        ),
        dim=-1,
    )
    motion_features = torch.where(
        frame_valid.unsqueeze(-1),
        motion_features,
        torch.zeros_like(motion_features),
    )
    return WarpedTerrainHistory(
        clearances=warped,
        visible_masks=warped_visible,
        motion_features=motion_features,
    )


class ConvGRUCell(nn.Module):
    """Small spatial GRU cell for the fixed 21x13 terrain grid."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        if min(input_channels, hidden_channels) <= 0:
            raise ValueError("ConvGRU channel counts must be positive")
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                inputs.shape[0],
                self.hidden_channels,
                inputs.shape[2],
                inputs.shape[3],
                device=inputs.device,
                dtype=inputs.dtype,
            )
        reset, update = torch.sigmoid(self.gates(torch.cat((inputs, hidden), dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((inputs, reset * hidden), dim=1)))
        return (1.0 - update) * hidden + update * candidate


@dataclass
class TemporalTerrainOutput:
    predicted_clearance: torch.Tensor
    completed_clearance: torch.Tensor
    current_visible: torch.Tensor
    hidden: torch.Tensor


class TemporalTerrainCompletion(nn.Module):
    """Mask-aware ConvGRU completion over already-warped 273D terrain maps."""

    def __init__(
        self,
        *,
        hidden_channels: int = 16,
        proprio_dim: int = 0,
        proprio_channels: int = 8,
    ) -> None:
        super().__init__()
        if proprio_dim < 0 or proprio_channels <= 0:
            raise ValueError("invalid proprio dimensions")
        self.proprio_dim = proprio_dim
        self.proprio_encoder = (
            nn.Sequential(
                nn.Linear(proprio_dim, proprio_channels),
                nn.SiLU(),
            )
            if proprio_dim
            else None
        )
        input_channels = 2 + 6 + (proprio_channels if proprio_dim else 0)
        self.recurrent = ConvGRUCell(input_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(
        self,
        history: WarpedTerrainHistory,
        *,
        proprio: torch.Tensor | None = None,
    ) -> TemporalTerrainOutput:
        clearances = history.clearances
        masks = history.visible_masks
        motion = history.motion_features
        if clearances.ndim != 3 or clearances.shape[-1] != DepthTerrainAdapter.GRID_DIMENSION:
            raise ValueError("history clearances must have shape [B, T, 273]")
        if masks.shape != clearances.shape or masks.dtype != torch.bool:
            raise ValueError("history masks must be bool and match clearances")
        if motion.shape != (*clearances.shape[:2], 6):
            raise ValueError("motion_features must have shape [B, T, 6]")
        if self.proprio_dim:
            if proprio is None or proprio.shape != (*clearances.shape[:2], self.proprio_dim):
                raise ValueError(f"proprio must have shape [B, T, {self.proprio_dim}]")
        elif proprio is not None:
            raise ValueError("proprio was provided to a model configured with proprio_dim=0")

        batch_size, time_steps = clearances.shape[:2]
        hidden = None
        for step in range(time_steps):
            values = torch.where(masks[:, step], clearances[:, step], torch.zeros_like(clearances[:, step]))
            values = values.reshape(batch_size, 1, *DepthTerrainAdapter.GRID_SHAPE).transpose(2, 3)
            visible = masks[:, step].to(clearances.dtype)
            visible = visible.reshape(batch_size, 1, *DepthTerrainAdapter.GRID_SHAPE).transpose(2, 3)
            features = [values, visible]
            motion_map = motion[:, step, :, None, None].expand(
                -1,
                -1,
                DepthTerrainAdapter.GRID_SHAPE[1],
                DepthTerrainAdapter.GRID_SHAPE[0],
            )
            features.append(motion_map)
            if self.proprio_encoder is not None:
                encoded = self.proprio_encoder(proprio[:, step])
                features.append(
                    encoded[:, :, None, None].expand(
                        -1,
                        -1,
                        DepthTerrainAdapter.GRID_SHAPE[1],
                        DepthTerrainAdapter.GRID_SHAPE[0],
                    )
                )
            hidden = self.recurrent(torch.cat(features, dim=1), hidden)

        assert hidden is not None
        prediction = F.softplus(self.head(hidden))
        prediction = prediction.transpose(2, 3).reshape(batch_size, -1)
        current_visible = masks[:, -1]
        current = clearances[:, -1]
        completed = torch.where(current_visible, current, prediction)
        return TemporalTerrainOutput(
            predicted_clearance=prediction,
            completed_clearance=completed,
            current_visible=current_visible,
            hidden=hidden,
        )


def terrain_completion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    target_valid: torch.Tensor | None = None,
    underfoot_weight: float = 2.0,
    beta: float = 0.05,
) -> torch.Tensor:
    """Smooth-L1 teacher loss with moderate center/underfoot emphasis."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 273:
        raise ValueError("prediction and target must both have shape [B, 273]")
    if underfoot_weight < 1.0 or beta <= 0.0:
        raise ValueError("underfoot_weight must be >= 1 and beta must be positive")
    finite = torch.isfinite(target)
    if target_valid is not None:
        if target_valid.shape != target.shape or target_valid.dtype != torch.bool:
            raise ValueError("target_valid must be bool and match target")
        finite &= target_valid
    if not torch.any(finite):
        return prediction.sum() * 0.0

    grid_x, grid_y = torch.meshgrid(
        torch.linspace(
            DepthTerrainAdapter.X_MIN,
            DepthTerrainAdapter.X_MAX,
            DepthTerrainAdapter.GRID_SHAPE[0],
            device=prediction.device,
            dtype=prediction.dtype,
        ),
        torch.linspace(
            DepthTerrainAdapter.Y_MIN,
            DepthTerrainAdapter.Y_MAX,
            DepthTerrainAdapter.GRID_SHAPE[1],
            device=prediction.device,
            dtype=prediction.dtype,
        ),
        indexing="ij",
    )
    underfoot = (grid_x.abs() <= 0.2001) & (grid_y.abs() <= 0.2001)
    weights = torch.where(
        underfoot.reshape(1, -1),
        torch.full_like(prediction, underfoot_weight),
        torch.ones_like(prediction),
    )
    errors = F.smooth_l1_loss(
        prediction,
        torch.where(finite, target, torch.zeros_like(target)),
        beta=beta,
        reduction="none",
    )
    return (errors * weights)[finite].sum() / weights[finite].sum()


def terrain_completion_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    current_visible: torch.Tensor | None = None,
    history_visible: torch.Tensor | None = None,
    edge_threshold: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Report map errors separately for missing, underfoot, and edge cells."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 273:
        raise ValueError("prediction and target must both have shape [B, 273]")
    if current_visible is not None and (current_visible.shape != target.shape or current_visible.dtype != torch.bool):
        raise ValueError("current_visible must be bool and match target")
    if history_visible is not None and (history_visible.shape != target.shape or history_visible.dtype != torch.bool):
        raise ValueError("history_visible must be bool and match target")
    if history_visible is not None and current_visible is None:
        raise ValueError("history_visible requires current_visible")
    if edge_threshold <= 0.0:
        raise ValueError("edge_threshold must be positive")
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    absolute_error = (prediction - target).abs()
    target_grid = target.reshape(-1, *DepthTerrainAdapter.GRID_SHAPE)
    edge = torch.zeros_like(target_grid, dtype=torch.bool)
    edge[:, 1:, :] |= (target_grid[:, 1:, :] - target_grid[:, :-1, :]).abs() > edge_threshold
    edge[:, :-1, :] |= (target_grid[:, 1:, :] - target_grid[:, :-1, :]).abs() > edge_threshold
    edge[:, :, 1:] |= (target_grid[:, :, 1:] - target_grid[:, :, :-1]).abs() > edge_threshold
    edge[:, :, :-1] |= (target_grid[:, :, 1:] - target_grid[:, :, :-1]).abs() > edge_threshold
    edge = edge.reshape_as(target)
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(
            DepthTerrainAdapter.X_MIN,
            DepthTerrainAdapter.X_MAX,
            DepthTerrainAdapter.GRID_SHAPE[0],
            device=target.device,
            dtype=target.dtype,
        ),
        torch.linspace(
            DepthTerrainAdapter.Y_MIN,
            DepthTerrainAdapter.Y_MAX,
            DepthTerrainAdapter.GRID_SHAPE[1],
            device=target.device,
            dtype=target.dtype,
        ),
        indexing="ij",
    )
    underfoot = ((grid_x.abs() <= 0.2001) & (grid_y.abs() <= 0.2001)).reshape(1, -1)

    def masked_mean(mask: torch.Tensor) -> torch.Tensor:
        selected = finite & mask
        if not torch.any(selected):
            return absolute_error.sum() * 0.0
        return absolute_error[selected].mean()

    metrics = {
        "mae": masked_mean(torch.ones_like(finite)),
        "underfoot_mae": masked_mean(underfoot.expand_as(finite)),
        "edge_mae": masked_mean(edge),
        "nonedge_mae": masked_mean(~edge),
    }
    if current_visible is not None:
        metrics["visible_mae"] = masked_mean(current_visible)
        metrics["missing_mae"] = masked_mean(~current_visible)
        valid_count = finite.sum().clamp_min(1)
        current_observed = finite & current_visible
        metrics["current_visible_fraction"] = current_observed.sum() / valid_count
        if history_visible is not None:
            historical_observed = finite & history_visible
            historical_only = finite & history_visible & ~current_visible
            metrics["history_visible_fraction"] = historical_observed.sum() / valid_count
            metrics["history_coverage_gain"] = historical_only.sum() / valid_count
            metrics["history_observed_missing_mae"] = masked_mean(history_visible & ~current_visible)
            metrics["never_observed_mae"] = masked_mean(~history_visible)
    return metrics
