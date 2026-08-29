"""Shared durable storage locations for generated motion data and expert caches."""

from __future__ import annotations

import os
from pathlib import Path


def data_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("UFO_DATA_DIR") or Path.cwd() / "data").expanduser().resolve()


def expert_buffer_cache_parent(cache_root: str | Path | None = None) -> Path:
    if cache_root is not None:
        return Path(cache_root).expanduser().resolve()
    return data_root()
