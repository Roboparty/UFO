"""Behavior-latent companion context utilities.

Heading is a deployable command, not part of the behavior latent.  The
helpers in this module deliberately operate on horizontal unit vectors so
callers never have to unwrap Euler yaw angles.
"""

from __future__ import annotations

import torch

from humanoidverse.utils.torch_utils import my_quat_rotate

HEADING_SOURCE_INVALID = 0
HEADING_SOURCE_FORWARD_COMMAND = 1
HEADING_SOURCE_EXACT_TRACKING = 2


def normalize_heading_xy(value: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    """Normalize ``[..., 2]`` horizontal vectors and reject degenerate input."""

    if value.shape[-1] != 2:
        raise ValueError(f"Expected horizontal vectors ending in 2, got {tuple(value.shape)}")
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if bool((~torch.isfinite(norm) | (norm <= eps)).any()):
        raise ValueError("Heading vectors must be finite and non-degenerate")
    return value / norm


def root_heading_xy(root_quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Return the projected world-frame body-forward direction."""

    if root_quat_xyzw.ndim != 2 or root_quat_xyzw.shape[-1] != 4:
        raise ValueError(f"Expected root quaternion [N,4], got {tuple(root_quat_xyzw.shape)}")
    local_forward = torch.zeros(
        (root_quat_xyzw.shape[0], 3),
        device=root_quat_xyzw.device,
        dtype=root_quat_xyzw.dtype,
    )
    local_forward[:, 0] = 1.0
    return normalize_heading_xy(my_quat_rotate(root_quat_xyzw, local_forward)[:, :2])


def rotate_heading_xy(value: torch.Tensor, rotation_xy: torch.Tensor) -> torch.Tensor:
    """Apply a planar rotation represented by ``[cos(theta), sin(theta)]``."""

    value = normalize_heading_xy(value)
    rotation_xy = normalize_heading_xy(rotation_xy)
    x = rotation_xy[..., 0] * value[..., 0] - rotation_xy[..., 1] * value[..., 1]
    y = rotation_xy[..., 1] * value[..., 0] + rotation_xy[..., 0] * value[..., 1]
    return normalize_heading_xy(torch.stack((x, y), dim=-1))


def rotation_between_heading_xy(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the planar rotation that maps ``source`` onto ``target``."""

    source = normalize_heading_xy(source)
    target = normalize_heading_xy(target)
    cos = torch.sum(source * target, dim=-1)
    sin = source[..., 0] * target[..., 1] - source[..., 1] * target[..., 0]
    return normalize_heading_xy(torch.stack((cos, sin), dim=-1))


def align_heading_sequence(
    reference_xy: torch.Tensor,
    current_heading_xy: torch.Tensor,
    reference_indices: torch.Tensor,
) -> torch.Tensor:
    """World-align each reference sequence at its active reference index."""

    if reference_xy.ndim != 3 or reference_xy.shape[-1] != 2:
        raise ValueError(f"Expected reference headings [N,T,2], got {tuple(reference_xy.shape)}")
    batch = reference_xy.shape[0]
    if current_heading_xy.shape != (batch, 2):
        raise ValueError(
            f"Expected current headings {(batch, 2)}, got {tuple(current_heading_xy.shape)}"
        )
    indices = reference_indices.to(device=reference_xy.device, dtype=torch.long).reshape(-1)
    if indices.shape[0] != batch or bool(((indices < 0) | (indices >= reference_xy.shape[1])).any()):
        raise ValueError("reference_indices must select one valid frame per sequence")
    active_reference = reference_xy[torch.arange(batch, device=reference_xy.device), indices]
    rotation = rotation_between_heading_xy(active_reference, current_heading_xy)
    return rotate_heading_xy(reference_xy, rotation[:, None, :])


def heading_observation(
    current_heading_xy: torch.Tensor,
    target_heading_xy: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return the zero-centered error ``[valid*(1-cos(error)), valid*sin(error)]``.

    Invalid contexts and valid zero-error contexts are deliberately identical:
    both are exactly ``[0, 0]`` before and after the identity normalizer.  This
    prevents heading validity from becoming a behavior-source identifier.
    ``sin(error)`` is positive when the target lies to the left of the robot's
    current forward direction.
    """

    current_heading_xy = normalize_heading_xy(current_heading_xy)
    if target_heading_xy.shape != current_heading_xy.shape:
        raise ValueError(
            f"Current/target heading shapes differ: {tuple(current_heading_xy.shape)} vs "
            f"{tuple(target_heading_xy.shape)}"
        )
    valid = valid.to(device=current_heading_xy.device, dtype=current_heading_xy.dtype).reshape(-1, 1)
    if valid.shape[0] != current_heading_xy.shape[0]:
        raise ValueError("Heading validity must contain one value per environment")

    # Invalid rows may deliberately carry a zero target.  Substitute current
    # heading only for the geometric calculation, then mask all outputs.
    safe_target = torch.where(valid.bool(), target_heading_xy, current_heading_xy)
    safe_target = normalize_heading_xy(safe_target)
    cos = torch.sum(current_heading_xy * safe_target, dim=-1, keepdim=True)
    # Re-normalizing two equal float32 headings can leave cos just below one.
    # Snap that numerical residue to one so valid zero-error and invalid
    # contexts are bitwise-identical zeros at the network boundary.
    cos = torch.where(cos > 1.0 - 1.0e-6, torch.ones_like(cos), cos)
    sin = (
        current_heading_xy[:, 0] * safe_target[:, 1]
        - current_heading_xy[:, 1] * safe_target[:, 0]
    ).unsqueeze(-1)
    return torch.cat((valid * (1.0 - cos), valid * sin), dim=-1)


def relative_heading_target(
    anchor_heading_xy: torch.Tensor,
    reference_heading_xy: torch.Tensor,
    reference_next_heading_xy: torch.Tensor,
) -> torch.Tensor:
    """Apply an expert one-step relative heading change at a policy state."""

    delta = rotation_between_heading_xy(reference_heading_xy, reference_next_heading_xy)
    return rotate_heading_xy(anchor_heading_xy, delta)


def repeat_heading_sequence(reference_xy: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat a turning reference without jumping back to its original yaw."""

    reference_xy = normalize_heading_xy(reference_xy)
    if reference_xy.ndim != 2:
        raise ValueError(f"Expected one heading sequence [T,2], got {tuple(reference_xy.shape)}")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    chunks = [reference_xy]
    previous_end = reference_xy[-1]
    for _ in range(1, repeats):
        rotation = rotation_between_heading_xy(reference_xy[0:1], previous_end.unsqueeze(0))[0]
        chunk = rotate_heading_xy(reference_xy, rotation)
        chunks.append(chunk)
        previous_end = chunk[-1]
    return torch.cat(chunks, dim=0)


__all__ = [
    "HEADING_SOURCE_EXACT_TRACKING",
    "HEADING_SOURCE_FORWARD_COMMAND",
    "HEADING_SOURCE_INVALID",
    "align_heading_sequence",
    "heading_observation",
    "normalize_heading_xy",
    "relative_heading_target",
    "repeat_heading_sequence",
    "root_heading_xy",
]
