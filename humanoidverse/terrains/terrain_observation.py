"""Robot-centric terrain sampling utilities for the privileged V0 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def reference_ray_index(offsets: torch.Tensor, *, atol: float = 1e-6) -> int:
    """Return the unique ray at robot-local ``(x, y) == (0, 0)``."""
    if offsets.ndim != 2 or offsets.shape[1] < 2:
        raise ValueError(f"ray offsets must have shape [N, >=2], got {tuple(offsets.shape)}")
    at_origin = torch.all(torch.isclose(offsets[:, :2], torch.zeros_like(offsets[:, :2]), atol=atol, rtol=0.0), dim=-1)
    matches = torch.nonzero(at_origin, as_tuple=False).flatten()
    if matches.numel() != 1:
        raise RuntimeError(f"expected exactly one terrain ray at local (0, 0), found {matches.numel()}")
    return int(matches.item())


def observations_from_clearances(
    clearances: torch.Tensor, reference_index: int, *, clip: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build root clearance, actor ranges, and relative terrain from one scan."""
    if clearances.ndim != 2:
        raise ValueError(f"terrain clearances must have shape [B, N], got {tuple(clearances.shape)}")
    if not 0 <= reference_index < clearances.shape[1]:
        raise IndexError(f"reference ray index {reference_index} is invalid for {clearances.shape[1]} rays")
    root_clearance = clearances[:, reference_index : reference_index + 1]
    terrain_actor = clearances
    terrain_priv = torch.clamp(root_clearance - clearances, -clip, clip)
    if not torch.isfinite(clearances).all() or not torch.isfinite(terrain_priv).all():
        raise RuntimeError("terrain sensor observations contain non-finite values")
    return root_clearance, terrain_actor, terrain_priv


def flat_zero_observations(
    pelvis_clearance: torch.Tensor, terrain_dimension: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return analytic observations for a known flat plane without raycasts."""
    if pelvis_clearance.ndim == 1:
        pelvis_clearance = pelvis_clearance.unsqueeze(-1)
    if pelvis_clearance.ndim != 2 or pelvis_clearance.shape[-1] != 1:
        raise ValueError(f"pelvis_clearance must have shape [N, 1], got {tuple(pelvis_clearance.shape)}")
    if terrain_dimension <= 0:
        raise ValueError(f"terrain_dimension must be positive, got {terrain_dimension}")
    terrain_actor = pelvis_clearance.expand(-1, terrain_dimension)
    terrain_priv = pelvis_clearance.new_zeros((pelvis_clearance.shape[0], terrain_dimension))
    return pelvis_clearance, terrain_actor, terrain_priv


@dataclass
class RobotCentricGridPatternCfg:
    """Asymmetric XY grid of vertical rays in the robot heading frame."""

    x_min: float = -0.4
    x_max: float = 1.6
    y_min: float = -0.6
    y_max: float = 0.6
    resolution: float = 0.1

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(round((self.x_max - self.x_min) / self.resolution)) + 1,
            int(round((self.y_max - self.y_min) / self.resolution)) + 1,
        )

    @property
    def dimension(self) -> int:
        nx, ny = self.shape
        return nx * ny

    def generate_rays(self, mj_model, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        del mj_model
        nx, ny = self.shape
        x = torch.linspace(self.x_min, self.x_max, nx, device=device, dtype=torch.float32)
        y = torch.linspace(self.y_min, self.y_max, ny, device=device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        offsets = torch.stack(
            (grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros(nx * ny, device=device)), dim=-1
        )
        directions = torch.zeros_like(offsets)
        directions[:, 2] = -1.0
        return offsets, directions
