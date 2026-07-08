"""Validation helpers for the UFO motion data schema."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np


REQUIRED_MOTION_FIELDS = ("root_trans_offset", "pose_aa", "fps")


def _fail(source_name: str, motion_key: str | None, message: str) -> None:
    key_text = f", motion_key={motion_key}" if motion_key is not None else ""
    raise ValueError(f"Invalid UFO motion data source={source_name}{key_text}: {message}")


def _coerce_fps(fps: Any, source_name: str, motion_key: str) -> float:
    fps_arr = np.asarray(fps)
    if fps_arr.size != 1:
        _fail(source_name, motion_key, f"fps must be a scalar, got shape {fps_arr.shape}")
    try:
        fps_value = float(fps_arr.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid UFO motion data source={source_name}, motion_key={motion_key}: fps is not numeric") from exc
    if not np.isfinite(fps_value) or fps_value <= 0.0:
        _fail(source_name, motion_key, f"fps must be > 0, got {fps_value}")
    return fps_value


def validate_ufo_motion_dict(data: dict[str, dict[str, Any]] | Mapping[str, Mapping[str, Any]], source_name: str) -> dict[str, Any]:
    """Validate and return a standard UFO/HumanoidVerse motion dictionary.

    The validator intentionally checks only the canonical fields required to
    enter MotionLib: root translation, axis-angle pose, and per-motion fps.
    It does not resample fps or retarget skeletons.
    """

    if not isinstance(data, Mapping):
        raise ValueError(f"Invalid UFO motion data source={source_name}: expected dict[str, dict], got {type(data).__name__}")
    if len(data) == 0:
        raise ValueError(f"Invalid UFO motion data source={source_name}: motion dictionary is empty")

    validated: dict[str, Any] = {}
    for key, motion in data.items():
        motion_key = str(key)
        if not isinstance(motion, Mapping):
            _fail(source_name, motion_key, f"motion record must be a dict, got {type(motion).__name__}")

        for field in REQUIRED_MOTION_FIELDS:
            if field not in motion:
                _fail(source_name, motion_key, f"missing required field '{field}'")

        root_trans_offset = np.asarray(motion["root_trans_offset"])
        pose_aa = np.asarray(motion["pose_aa"])
        _coerce_fps(motion["fps"], source_name, motion_key)

        if root_trans_offset.ndim != 2 or root_trans_offset.shape[1] != 3:
            _fail(source_name, motion_key, f"root_trans_offset must have shape [T, 3], got {root_trans_offset.shape}")
        if pose_aa.ndim != 3 or pose_aa.shape[2] != 3:
            _fail(source_name, motion_key, f"pose_aa must have shape [T, J, 3], got {pose_aa.shape}")
        if root_trans_offset.shape[0] <= 0:
            _fail(source_name, motion_key, "motion must contain at least one frame")
        if root_trans_offset.shape[0] != pose_aa.shape[0]:
            _fail(
                source_name,
                motion_key,
                f"root_trans_offset and pose_aa must share T, got {root_trans_offset.shape[0]} and {pose_aa.shape[0]}",
            )

        validated[motion_key] = motion

    return validated


def motion_fps_values(data: Mapping[str, Mapping[str, Any]], source_name: str) -> list[float]:
    return [_coerce_fps(motion["fps"], source_name, str(key)) for key, motion in data.items()]


def format_fps_distribution(data: Mapping[str, Mapping[str, Any]], source_name: str) -> str:
    fps_values = motion_fps_values(data, source_name)
    counts = Counter(fps_values)
    parts: list[str] = []
    for fps_value in sorted(counts):
        if float(fps_value).is_integer():
            fps_text = str(int(fps_value))
        else:
            fps_text = f"{fps_value:.6g}"
        parts.append(f"{fps_text}: {counts[fps_value]}")
    return "{" + ", ".join(parts) + "}"
