"""Inspect robot_state CSV/NPZ files before building a UFO motion manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from humanoidverse.utils.motion_data.adapters import expand_motion_paths
from humanoidverse.utils.robot_spec import RobotSpec, load_robot_spec


ROOT_POS_ALIASES = [
    ["root_pos_x", "root_pos_y", "root_pos_z"],
    ["root_x", "root_y", "root_z"],
    ["base_pos_x", "base_pos_y", "base_pos_z"],
    ["base_x", "base_y", "base_z"],
    ["pelvis_pos_x", "pelvis_pos_y", "pelvis_pos_z"],
]

ROOT_QUAT_ALIASES = [
    ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"],
    ["root_qx", "root_qy", "root_qz", "root_qw"],
    ["base_quat_x", "base_quat_y", "base_quat_z", "base_quat_w"],
    ["base_qx", "base_qy", "base_qz", "base_qw"],
    ["pelvis_quat_x", "pelvis_quat_y", "pelvis_quat_z", "pelvis_quat_w"],
]


@dataclass(frozen=True)
class FileInspection:
    path: Path
    frames: int
    fps: float
    duration_seconds: float
    estimated_clip_count: int
    root_pos_columns: list[str] | None = None
    root_quat_columns: list[str] | None = None
    dof_pos_mode: str | None = None
    missing_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataInspection:
    format: str
    source_paths: list[Path]
    files: list[FileInspection]
    root_pos_columns: list[str] | None
    root_quat_columns: list[str] | None
    dof_pos_mode: str | None
    manifest_columns: dict[str, Any]
    suggested_manifest: dict[str, Any]

    @property
    def fps_values(self) -> list[float]:
        return [item.fps for item in self.files]

    @property
    def duration_range(self) -> tuple[float, float]:
        durations = [item.duration_seconds for item in self.files]
        return min(durations), max(durations)

    @property
    def estimated_clip_count(self) -> int:
        return sum(item.estimated_clip_count for item in self.files)


def _read_header_and_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    return list(reader.fieldnames), rows


def _first_alias(fieldnames: list[str], aliases: list[list[str]]) -> list[str] | None:
    names = set(fieldnames)
    for candidate in aliases:
        if all(column in names for column in candidate):
            return candidate
    return None


def _infer_fps_from_time(rows: list[dict[str, str]], path: Path) -> float:
    try:
        times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"CSV file={path} needs --fps when no time column is present") from exc
    except ValueError as exc:
        raise ValueError(f"CSV file={path} time column contains non-numeric values") from exc
    if times.shape[0] < 2:
        raise ValueError(f"CSV file={path} needs at least two time samples to infer fps")
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError(f"CSV file={path} time column must be strictly increasing")
    median_dt = float(np.median(dt))
    if not np.isfinite(median_dt) or median_dt <= 0.0:
        raise ValueError(f"CSV file={path} has invalid median dt={median_dt}")
    return 1.0 / median_dt


def _resolve_fps(fieldnames: list[str], rows: list[dict[str, str]], path: Path, fps: float | None) -> float:
    if "time" in fieldnames:
        return _infer_fps_from_time(rows, path)
    if fps is not None:
        value = float(fps)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid --fps value: {fps}")
        return value
    raise ValueError(f"CSV file={path} needs either a time column or --fps")


def estimate_clip_count(
    frames: int,
    fps: float,
    *,
    clip_seconds: float = 10.0,
    stride_seconds: float | None = None,
    keep_short: bool = True,
    min_clip_seconds: float = 1.0,
) -> int:
    if frames <= 0:
        return 0
    stride_seconds = clip_seconds if stride_seconds is None else stride_seconds
    clip_frames = max(1, int(round(clip_seconds * fps)))
    stride_frames = max(1, int(round(stride_seconds * fps)))
    min_frames = max(1, int(round(min_clip_seconds * fps))) if min_clip_seconds > 0 else 1
    if frames >= clip_frames:
        count = 0
        start = 0
        while start + clip_frames <= frames:
            count += 1
            start += stride_frames
        if keep_short and start < frames and frames - start >= min_frames:
            count += 1
        return count
    return 1 if keep_short and frames >= min_frames else 0


def detect_csv_columns(fieldnames: list[str], robot_spec: RobotSpec) -> tuple[dict[str, Any], list[str], list[str]]:
    missing: list[str] = []
    notes: list[str] = []
    root_pos = _first_alias(fieldnames, ROOT_POS_ALIASES)
    root_quat = _first_alias(fieldnames, ROOT_QUAT_ALIASES)
    if root_pos is None:
        missing.append("root_pos aliases: " + " or ".join("/".join(alias) for alias in ROOT_POS_ALIASES))
    if root_quat is None:
        missing.append("root_quat aliases: " + " or ".join("/".join(alias) for alias in ROOT_QUAT_ALIASES))

    names = set(fieldnames)
    missing_named = [joint for joint in robot_spec.control_joint_names if joint not in names]
    dof_columns = [f"dof_{idx}" for idx in range(len(robot_spec.control_joint_names))]
    missing_dof = [column for column in dof_columns if column not in names]

    has_named = not missing_named
    has_xml_order = not missing_dof
    if has_named:
        dof_mode = "auto_by_joint_name"
        if has_xml_order:
            notes.append("Both named joint columns and dof_0..dof_N columns exist; using auto_by_joint_name.")
    elif has_xml_order:
        dof_mode = "xml_order"
    else:
        dof_mode = None
        missing.append("missing named joints: " + ", ".join(missing_named))
        missing.append("missing xml_order dof columns: " + ", ".join(missing_dof))

    columns: dict[str, Any] = {}
    if root_pos is not None:
        columns["root_pos"] = root_pos
    if root_quat is not None:
        columns["root_quat"] = root_quat
    if dof_mode is not None:
        columns["dof_pos"] = dof_mode
    return columns, missing, notes


def _inspect_csv_file(
    path: Path,
    robot_spec: RobotSpec,
    *,
    fps: float | None,
    clip_seconds: float,
    stride_seconds: float | None,
    keep_short: bool,
    min_clip_seconds: float,
) -> FileInspection:
    fieldnames, rows = _read_header_and_rows(path)
    columns, missing, notes = detect_csv_columns(fieldnames, robot_spec)
    motion_fps = _resolve_fps(fieldnames, rows, path, fps)
    frames = len(rows)
    return FileInspection(
        path=path,
        frames=frames,
        fps=motion_fps,
        duration_seconds=frames / motion_fps,
        estimated_clip_count=estimate_clip_count(
            frames,
            motion_fps,
            clip_seconds=clip_seconds,
            stride_seconds=stride_seconds,
            keep_short=keep_short,
            min_clip_seconds=min_clip_seconds,
        ),
        root_pos_columns=columns.get("root_pos"),
        root_quat_columns=columns.get("root_quat"),
        dof_pos_mode=columns.get("dof_pos"),
        missing_columns=missing,
        notes=notes,
    )


def _string_list_from_npz(value: Any) -> list[str]:
    arr = np.asarray(value)
    result: list[str] = []
    for item in arr.reshape(-1):
        result.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return result


def _fps_from_npz(npz: Any, path: Path, fps: float | None) -> float:
    if fps is not None:
        value = float(fps)
    elif "fps" in npz:
        value = float(np.asarray(npz["fps"]).reshape(-1)[0])
    elif "time" in npz:
        times = np.asarray(npz["time"], dtype=np.float64)
        if times.ndim != 1 or times.shape[0] < 2:
            raise ValueError(f"NPZ file={path} time must have shape [T] with at least two samples")
        dt = np.diff(times)
        if np.any(dt <= 0.0):
            raise ValueError(f"NPZ file={path} time must be strictly increasing")
        value = 1.0 / float(np.median(dt))
    else:
        raise ValueError(f"NPZ file={path} requires fps, time, or --fps")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"NPZ file={path} has invalid fps={value}")
    return value


def _inspect_npz_file(
    path: Path,
    robot_spec: RobotSpec,
    *,
    fps: float | None,
    clip_seconds: float,
    stride_seconds: float | None,
    keep_short: bool,
    min_clip_seconds: float,
) -> FileInspection:
    missing: list[str] = []
    notes: list[str] = []
    with np.load(path, allow_pickle=True) as npz:
        for field_name in ("root_pos", "root_quat", "dof_pos"):
            if field_name not in npz:
                missing.append(field_name)
        if missing:
            raise ValueError(f"robot_state_npz file={path} missing fields: {missing}")
        root_pos = np.asarray(npz["root_pos"])
        root_quat = np.asarray(npz["root_quat"])
        dof_pos = np.asarray(npz["dof_pos"])
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            missing.append(f"root_pos shape [T,3], got {root_pos.shape}")
        if root_quat.ndim != 2 or root_quat.shape[1] != 4:
            missing.append(f"root_quat shape [T,4], got {root_quat.shape}")
        if dof_pos.ndim != 2:
            missing.append(f"dof_pos shape [T,N], got {dof_pos.shape}")
        if root_pos.shape[0] != root_quat.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
            missing.append("root_pos/root_quat/dof_pos must share T")
        if "joint_names" in npz:
            joint_names = _string_list_from_npz(npz["joint_names"])
            absent = [joint for joint in robot_spec.control_joint_names if joint not in set(joint_names)]
            if absent:
                missing.append("missing joint_names entries: " + ", ".join(absent))
            else:
                notes.append("joint_names found; dof_pos will be reordered into RobotSpec.control_joint_names.")
        else:
            expected = len(robot_spec.control_joint_names)
            if dof_pos.ndim == 2 and dof_pos.shape[1] != expected:
                missing.append(f"dof_pos width must be {expected} without joint_names, got {dof_pos.shape[1]}")
            notes.append("No joint_names found; assuming dof_pos already follows RobotSpec.control_joint_names.")
        motion_fps = _fps_from_npz(npz, path, fps)
        frames = int(root_pos.shape[0])
    return FileInspection(
        path=path,
        frames=frames,
        fps=motion_fps,
        duration_seconds=frames / motion_fps,
        estimated_clip_count=estimate_clip_count(
            frames,
            motion_fps,
            clip_seconds=clip_seconds,
            stride_seconds=stride_seconds,
            keep_short=keep_short,
            min_clip_seconds=min_clip_seconds,
        ),
        dof_pos_mode="npz",
        missing_columns=missing,
        notes=notes,
    )


def detect_npz_layout(path: Path, robot_spec: RobotSpec, *, fps: float | None = None) -> FileInspection:
    return _inspect_npz_file(
        path,
        robot_spec,
        fps=fps,
        clip_seconds=10.0,
        stride_seconds=None,
        keep_short=True,
        min_clip_seconds=1.0,
    )


def _infer_format(source: str, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    suffix = Path(source).suffix.lower()
    if suffix == ".csv":
        return "robot_state_csv"
    if suffix == ".npz":
        return "robot_state_npz"
    raise ValueError(f"Cannot infer format from source={source!r}; pass --format robot_state_csv or robot_state_npz")


def inspect_data_source(
    *,
    robot_config: str | Path,
    source: str | list[str],
    fmt: str = "auto",
    fps: float | None = None,
    clip_seconds: float = 10.0,
    stride_seconds: float | None = None,
    keep_short: bool = True,
    min_clip_seconds: float = 1.0,
    dataset_name: str = "dataset",
    weight: float = 1.0,
) -> DataInspection:
    robot_spec = load_robot_spec(robot_config)
    source_for_format = source[0] if isinstance(source, list) else source
    resolved_format = _infer_format(str(source_for_format), fmt)
    suffix = ".csv" if resolved_format == "robot_state_csv" else ".npz" if resolved_format == "robot_state_npz" else ""
    if not suffix:
        raise ValueError(f"data_inspect supports robot_state_csv and robot_state_npz, got {resolved_format}")
    paths = expand_motion_paths(source, suffix=suffix)

    files: list[FileInspection] = []
    manifest_columns: dict[str, Any] = {}
    if resolved_format == "robot_state_csv":
        for path in paths:
            files.append(
                _inspect_csv_file(
                    path,
                    robot_spec,
                    fps=fps,
                    clip_seconds=clip_seconds,
                    stride_seconds=stride_seconds,
                    keep_short=keep_short,
                    min_clip_seconds=min_clip_seconds,
                )
            )
        first = files[0]
        if first.root_pos_columns is not None:
            manifest_columns["root_pos"] = first.root_pos_columns
        if first.root_quat_columns is not None:
            manifest_columns["root_quat"] = first.root_quat_columns
        if first.dof_pos_mode is not None:
            manifest_columns["dof_pos"] = first.dof_pos_mode
    else:
        for path in paths:
            files.append(
                _inspect_npz_file(
                    path,
                    robot_spec,
                    fps=fps,
                    clip_seconds=clip_seconds,
                    stride_seconds=stride_seconds,
                    keep_short=keep_short,
                    min_clip_seconds=min_clip_seconds,
                )
            )

    missing = [entry for item in files for entry in item.missing_columns]
    if missing:
        raise ValueError("Motion source inspection failed: " + "; ".join(missing))

    source_path: str | list[str] = source
    suggested_dataset: dict[str, Any] = {
        "name": dataset_name,
        "format": resolved_format,
        "source_path": source_path,
        "weight": float(weight),
        "auto_build": {
            "train_clip_seconds": float(clip_seconds),
            "clip_stride_seconds": float(stride_seconds if stride_seconds is not None else clip_seconds),
            "keep_short": bool(keep_short),
            "min_clip_seconds": float(min_clip_seconds),
        },
    }
    if fps is not None:
        suggested_dataset["fps"] = float(fps)
    if manifest_columns:
        suggested_dataset["columns"] = manifest_columns
    suggested = {"robot_config": str(robot_config), "datasets": [suggested_dataset]}
    return DataInspection(
        format=resolved_format,
        source_paths=paths,
        files=files,
        root_pos_columns=manifest_columns.get("root_pos"),
        root_quat_columns=manifest_columns.get("root_quat"),
        dof_pos_mode=manifest_columns.get("dof_pos") or ("npz" if resolved_format == "robot_state_npz" else None),
        manifest_columns=manifest_columns,
        suggested_manifest=suggested,
    )


def _print_report(result: DataInspection) -> None:
    duration_min, duration_max = result.duration_range
    fps_values = result.fps_values
    print("Detected RobotState schema")
    print(f"  reader: {result.format}")
    print(f"  files: {len(result.source_paths)}")
    print(f"  fps range: {min(fps_values):.6g} - {max(fps_values):.6g}")
    print(f"  duration range: {duration_min:.3f}s - {duration_max:.3f}s")
    print(f"  estimated near10s clips: {result.estimated_clip_count}")
    if result.root_pos_columns:
        print(f"  root_pos columns: {result.root_pos_columns}")
    if result.root_quat_columns:
        print(f"  root_quat columns: {result.root_quat_columns}")
    if result.dof_pos_mode:
        print(f"  dof_pos mode: {result.dof_pos_mode}")
    notes = [note for item in result.files for note in item.notes]
    for note in sorted(set(notes)):
        print(f"  note: {note}")
    print("Suggested manifest snippet:")
    print(OmegaConf.to_yaml(OmegaConf.create(result.suggested_manifest)).rstrip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, help="Robot YAML path")
    parser.add_argument("--source", required=True, nargs="+", help="CSV/NPZ source path, directory, or glob")
    parser.add_argument("--format", default="auto", choices=["auto", "robot_state_csv", "robot_state_npz"])
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--keep-short", dest="keep_short", action="store_true", default=True)
    parser.add_argument("--drop-short", dest="keep_short", action="store_false")
    parser.add_argument("--min-clip-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    source: str | list[str] = args.source[0] if len(args.source) == 1 else args.source
    result = inspect_data_source(
        robot_config=args.robot,
        source=source,
        fmt=args.format,
        fps=args.fps,
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        keep_short=args.keep_short,
        min_clip_seconds=args.min_clip_seconds,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
