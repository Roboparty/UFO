"""Small, dependency-light helpers for exact same-z terrain transfer."""

from __future__ import annotations

import hashlib

import torch


def tensor_checksum(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def clone_same_z_for_terrains(z: torch.Tensor, terrains: list[str]) -> dict[str, torch.Tensor]:
    """Clone one computed latent and prove every terrain receives identical bytes."""
    expected = tensor_checksum(z)
    result = {terrain: z.detach().clone() for terrain in terrains}
    checksums = {terrain: tensor_checksum(value) for terrain, value in result.items()}
    if any(checksum != expected for checksum in checksums.values()):
        raise AssertionError(f"same-z invariant violated: expected={expected}, actual={checksums}")
    return result
