"""Clip UFO motion dictionaries into fixed-duration training windows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict


def _slice_motion_record(record: dict[str, Any], start: int, end: int, total_frames: int) -> dict[str, Any]:
    clipped: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_frames:
            clipped[key] = value[start:end].copy()
        else:
            clipped[key] = deepcopy(value)
    return clipped


def clip_ufo_motion_dict(
    data: dict[str, dict[str, Any]],
    clip_seconds: float = 10.0,
    stride_seconds: float = 10.0,
    keep_short: bool = True,
    min_clip_seconds: float = 1.0,
    source_name: str = "",
) -> dict[str, Any]:
    """Clip each motion while preserving all synchronized time-series fields.

    No fps resampling is performed. Each motion uses its own fps to convert
    seconds into frame counts.
    """

    validated = validate_ufo_motion_dict(data, source_name or "clip")
    if clip_seconds <= 0.0:
        raise ValueError(f"clip_seconds must be > 0, got {clip_seconds}")
    if stride_seconds <= 0.0:
        raise ValueError(f"stride_seconds must be > 0, got {stride_seconds}")
    if min_clip_seconds < 0.0:
        raise ValueError(f"min_clip_seconds must be >= 0, got {min_clip_seconds}")

    clipped_data: dict[str, Any] = {}
    for motion_key, record in validated.items():
        total_frames = int(np.asarray(record["root_trans_offset"]).shape[0])
        fps = float(np.asarray(record["fps"]).reshape(-1)[0])
        clip_frames = max(1, int(round(clip_seconds * fps)))
        stride_frames = max(1, int(round(stride_seconds * fps)))
        min_frames = max(1, int(round(min_clip_seconds * fps))) if min_clip_seconds > 0.0 else 1

        windows: list[tuple[int, int]] = []
        if total_frames >= clip_frames:
            start = 0
            while start + clip_frames <= total_frames:
                windows.append((start, start + clip_frames))
                start += stride_frames
            if keep_short and start < total_frames and total_frames - start >= min_frames:
                windows.append((start, total_frames))
        elif keep_short and total_frames >= min_frames:
            windows.append((0, total_frames))

        for clip_idx, (start, end) in enumerate(windows):
            new_key = f"{motion_key}__clip{clip_idx:03d}"
            if new_key in clipped_data:
                raise ValueError(f"Duplicate clipped motion key: {new_key}")
            clipped = _slice_motion_record(record, start, end, total_frames)
            clipped["motion_key"] = new_key
            metadata = dict(clipped.get("metadata") or {})
            metadata.update(
                {
                    "source_motion_key": motion_key,
                    "clip_start_frame": start,
                    "clip_end_frame": end,
                    "clip_seconds": clip_seconds,
                    "stride_seconds": stride_seconds,
                }
            )
            clipped["metadata"] = metadata
            clipped_data[new_key] = clipped

    if not clipped_data:
        raise ValueError(
            f"No motion clips were generated for source={source_name!r}. "
            f"Try keep_short=true or min_clip_seconds <= shortest motion length."
        )
    return validate_ufo_motion_dict(clipped_data, source_name or "clip")
