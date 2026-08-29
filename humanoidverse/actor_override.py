"""Read-only Actor milestone overrides for inference and evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


def state_dict_checksum(state_dict: dict[str, torch.Tensor]) -> str:
    """Return a stable checksum for a tensor state dict."""
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state dict value {name!r} is not a tensor")
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_actor_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load and validate an Actor-only milestone without mutating a model."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing Actor override checkpoint: {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    actor_state = payload.get("actor") if isinstance(payload, dict) else None
    if not isinstance(actor_state, dict) or not actor_state:
        raise ValueError(f"invalid Actor milestone checkpoint: {resolved}")
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in actor_state.values()):
        raise ValueError(f"Actor milestone contains non-finite or non-tensor state: {resolved}")
    return actor_state, {
        "path": str(resolved),
        "step": int(payload["step"]),
        "checksum": state_dict_checksum(actor_state),
    }


def load_actor_module_override(actor: torch.nn.Module, path: Path) -> dict[str, Any]:
    """Load an Actor-only milestone into an Actor module."""
    actor_state, identity = load_actor_state(path)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval().requires_grad_(False)
    return identity


def load_actor_override(model: Any, path: Path) -> dict[str, Any]:
    """Load an Actor-only milestone into memory without modifying its source file."""
    actor = getattr(model, "_actor", None)
    if actor is None:
        raise AttributeError("loaded model does not expose an _actor module")
    identity = load_actor_module_override(actor, path)
    model.eval().requires_grad_(False)
    return identity
