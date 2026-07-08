"""Adapters that convert normalized motion sources into UFO motion dicts."""

from __future__ import annotations

import csv
import glob
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger

from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict


SUPPORTED_FORMATS = {"ufo_pkl", "ufo_npz", "csv_ufo"}


def _as_path_list(path_spec: str | os.PathLike[str] | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(path_spec, (list, tuple)):
        return [str(item) for item in path_spec]
    return [str(path_spec)]


def _candidate_patterns(raw_path: str, base_dir: Path | None) -> list[Path]:
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    path = Path(expanded)
    if path.is_absolute():
        return [path]
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(base_dir / path)
    candidates.append(Path.cwd() / path)
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def expand_motion_paths(
    path_spec: str | os.PathLike[str] | list[str] | tuple[str, ...],
    *,
    base_dir: Path | None = None,
    suffix: str,
) -> list[Path]:
    """Resolve a path, directory, glob, or list of those into concrete files."""

    resolved: list[Path] = []
    missing_patterns: list[str] = []
    for raw in _as_path_list(path_spec):
        found_for_raw: list[Path] = []
        for candidate in _candidate_patterns(raw, base_dir):
            candidate_str = str(candidate)
            if glob.has_magic(candidate_str):
                found_for_raw = [Path(item) for item in sorted(glob.glob(candidate_str))]
            elif candidate.is_dir():
                found_for_raw = sorted(candidate.glob(f"*{suffix}"))
            elif candidate.exists():
                found_for_raw = [candidate]
            if found_for_raw:
                break
        if not found_for_raw:
            missing_patterns.append(raw)
        resolved.extend(found_for_raw)

    filtered = [path.expanduser().resolve() for path in resolved if path.suffix == suffix]
    if not filtered:
        raise FileNotFoundError(f"No {suffix} motion files matched: {missing_patterns or path_spec}")
    return filtered


def _merge_motion_dicts(sources: list[tuple[Path, dict[str, Any]]], source_name: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path, data in sources:
        for motion_key, motion in data.items():
            if motion_key in merged:
                raise ValueError(f"Duplicate motion_key={motion_key} while merging source={source_name} from {path}")
            merged[motion_key] = motion
    return validate_ufo_motion_dict(merged, source_name)


def load_ufo_pkl(path_spec: str | os.PathLike[str] | list[str], *, source_name: str, base_dir: Path | None = None) -> dict[str, Any]:
    sources: list[tuple[Path, dict[str, Any]]] = []
    for path in expand_motion_paths(path_spec, base_dir=base_dir, suffix=".pkl"):
        data = joblib.load(path)
        validated = validate_ufo_motion_dict(data, f"{source_name}:{path.name}")
        sources.append((path, validated))
    return _merge_motion_dicts(sources, source_name)


def _scalar_from_npz(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def load_ufo_npz(path_spec: str | os.PathLike[str] | list[str], *, source_name: str, base_dir: Path | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for path in expand_motion_paths(path_spec, base_dir=base_dir, suffix=".npz"):
        with np.load(path, allow_pickle=True) as npz:
            missing = [field for field in ("root_trans_offset", "pose_aa", "fps") if field not in npz]
            if missing:
                raise ValueError(f"Invalid ufo_npz source={source_name}, file={path}: missing fields {missing}")
            motion_key = str(_scalar_from_npz(npz["motion_key"])) if "motion_key" in npz else path.stem
            if motion_key in data:
                raise ValueError(f"Duplicate motion_key={motion_key} while loading ufo_npz source={source_name}")
            record: dict[str, Any] = {
                "root_trans_offset": np.asarray(npz["root_trans_offset"]),
                "pose_aa": np.asarray(npz["pose_aa"]),
                "fps": _scalar_from_npz(npz["fps"]),
                "motion_key": motion_key,
                "source": source_name,
            }
            for optional_field in ("action", "dof_pos", "dof_vel", "joint_names", "body_names", "metadata"):
                if optional_field in npz:
                    record[optional_field] = _scalar_from_npz(npz[optional_field])
            data[motion_key] = record
    return validate_ufo_motion_dict(data, source_name)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"csv_ufo file has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"csv_ufo file is empty: {path}")
    return list(reader.fieldnames), rows


def _column_array(rows: list[dict[str, str]], column: str, path: Path) -> np.ndarray:
    try:
        return np.asarray([float(row[column]) for row in rows], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"csv_ufo file={path} missing required column '{column}'") from exc
    except ValueError as exc:
        raise ValueError(f"csv_ufo file={path} column '{column}' contains non-numeric values") from exc


def _infer_fps_from_time(rows: list[dict[str, str]], path: Path) -> float:
    try:
        times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"csv_ufo file={path} missing required column 'time'") from exc
    except ValueError as exc:
        raise ValueError(f"csv_ufo file={path} column 'time' contains non-numeric values") from exc
    if len(times) < 2:
        raise ValueError(f"csv_ufo file={path} needs at least two time samples to infer fps")
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError(f"csv_ufo file={path} time column must be strictly increasing")
    median_dt = float(np.median(dt))
    if median_dt <= 0.0 or not np.isfinite(median_dt):
        raise ValueError(f"csv_ufo file={path} has invalid median dt={median_dt}")
    rel_jitter = float(np.max(np.abs(dt - median_dt)) / median_dt)
    if rel_jitter > 0.05:
        logger.warning(f"csv_ufo file={path} has non-uniform time intervals; max relative jitter={rel_jitter:.3f}")
    return 1.0 / median_dt


def _pose_from_named_columns(fieldnames: list[str], rows: list[dict[str, str]], path: Path) -> np.ndarray | None:
    pattern = re.compile(r"^pose_aa_(\d+)_(x|y|z)$")
    grouped: dict[int, dict[str, str]] = {}
    for name in fieldnames:
        match = pattern.match(name)
        if match:
            joint_idx = int(match.group(1))
            axis = match.group(2)
            grouped.setdefault(joint_idx, {})[axis] = name
    if not grouped:
        return None

    joint_indices = sorted(grouped)
    pose = np.zeros((len(rows), len(joint_indices), 3), dtype=np.float32)
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    for out_joint_idx, joint_idx in enumerate(joint_indices):
        columns = grouped[joint_idx]
        if set(columns) != {"x", "y", "z"}:
            raise ValueError(f"csv_ufo file={path} pose_aa_{joint_idx}_* must contain x, y, and z columns")
        for axis, axis_idx in axis_to_idx.items():
            pose[:, out_joint_idx, axis_idx] = _column_array(rows, columns[axis], path)
    return pose


def _pose_from_flat_columns(fieldnames: list[str], rows: list[dict[str, str]], path: Path) -> np.ndarray | None:
    pattern = re.compile(r"^pose_aa_flat_(\d+)$")
    indexed_columns: list[tuple[int, str]] = []
    for name in fieldnames:
        match = pattern.match(name)
        if match:
            indexed_columns.append((int(match.group(1)), name))
    if not indexed_columns:
        return None
    indexed_columns.sort()
    if [idx for idx, _ in indexed_columns] != list(range(len(indexed_columns))):
        raise ValueError(f"csv_ufo file={path} pose_aa_flat_* columns must be contiguous from 0")
    if len(indexed_columns) % 3 != 0:
        raise ValueError(f"csv_ufo file={path} pose_aa_flat_* column count must be divisible by 3")
    flat = np.stack([_column_array(rows, column, path) for _, column in indexed_columns], axis=1)
    return flat.reshape(len(rows), len(indexed_columns) // 3, 3).astype(np.float32)


def load_csv_ufo(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    base_dir: Path | None = None,
    fps: float | int | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for path in expand_motion_paths(path_spec, base_dir=base_dir, suffix=".csv"):
        fieldnames, rows = _read_csv_rows(path)
        root = np.stack(
            [
                _column_array(rows, "root_trans_offset_x", path),
                _column_array(rows, "root_trans_offset_y", path),
                _column_array(rows, "root_trans_offset_z", path),
            ],
            axis=1,
        )
        pose = _pose_from_named_columns(fieldnames, rows, path)
        if pose is None:
            pose = _pose_from_flat_columns(fieldnames, rows, path)
        if pose is None:
            raise ValueError(
                f"csv_ufo source={source_name}, file={path} does not contain pose_aa columns. "
                "Raw G1 joint CSV requires a separate conversion/retarget step into pose_aa before UFO MotionLib can load it."
            )

        if fps is not None:
            motion_fps = float(fps)
        elif "time" in fieldnames:
            motion_fps = _infer_fps_from_time(rows, path)
        else:
            raise ValueError(f"csv_ufo source={source_name}, file={path} requires a time column or manifest fps")

        motion_key = path.stem
        if motion_key in data:
            raise ValueError(f"Duplicate motion_key={motion_key} while loading csv_ufo source={source_name}")
        data[motion_key] = {
            "root_trans_offset": root,
            "pose_aa": pose,
            "fps": motion_fps,
            "motion_key": motion_key,
            "source": source_name,
        }
    return validate_ufo_motion_dict(data, source_name)


def load_motion_data_by_format(
    fmt: str,
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    base_dir: Path | None = None,
    fps: float | int | None = None,
) -> dict[str, Any]:
    fmt = str(fmt)
    if fmt == "ufo_pkl":
        return load_ufo_pkl(path_spec, source_name=source_name, base_dir=base_dir)
    if fmt == "ufo_npz":
        return load_ufo_npz(path_spec, source_name=source_name, base_dir=base_dir)
    if fmt == "csv_ufo":
        return load_csv_ufo(path_spec, source_name=source_name, base_dir=base_dir, fps=fps)
    raise ValueError(f"Unsupported motion data format '{fmt}'. Supported formats: {sorted(SUPPORTED_FORMATS)}")


def dump_ufo_pkl(data: dict[str, Any], output_path: Path, source_name: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_ufo_motion_dict(data, source_name)
    joblib.dump(validated, output_path)
    return output_path
