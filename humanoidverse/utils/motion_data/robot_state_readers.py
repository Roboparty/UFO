"""File readers that parse CSV/NPZ sources into RobotStateMotion records."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from humanoidverse.utils.motion_data.paths import expand_motion_paths
from humanoidverse.utils.motion_data.robot_state import RobotStateMotion, validate_robot_state_dict
from humanoidverse.utils.robot_spec import RobotSpec


def _scalar_from_npz(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def _motion_key_from_npz(value: Any) -> str:
    scalar = _scalar_from_npz(value)
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


DEFAULT_ROOT_POS_COLUMNS = ["root_pos_x", "root_pos_y", "root_pos_z"]
DEFAULT_ROOT_QUAT_COLUMNS = ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"]


def _is_numeric_row(row: list[str]) -> bool:
    if not row:
        return False
    try:
        for value in row:
            float(value)
    except ValueError:
        return False
    return True


def _headerless_csv_fieldnames(path: Path, width: int, dof_count: int) -> list[str]:
    dof_columns = [f"dof_{idx}" for idx in range(dof_count)]
    no_time_width = len(DEFAULT_ROOT_POS_COLUMNS) + len(DEFAULT_ROOT_QUAT_COLUMNS) + dof_count
    with_time_width = 1 + no_time_width
    if width == no_time_width:
        return [*DEFAULT_ROOT_POS_COLUMNS, *DEFAULT_ROOT_QUAT_COLUMNS, *dof_columns]
    if width == with_time_width:
        return ["time", *DEFAULT_ROOT_POS_COLUMNS, *DEFAULT_ROOT_QUAT_COLUMNS, *dof_columns]
    raise ValueError(
        f"CSV file={path} has no header and {width} columns; expected {no_time_width} "
        f"columns without time or {with_time_width} columns with a leading time column for {dof_count} DOFs"
    )


def _dict_rows(fieldnames: list[str], raw_rows: list[list[str]], path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    expected = len(fieldnames)
    for line_no, row in enumerate(raw_rows, start=1):
        if len(row) != expected:
            raise ValueError(f"CSV file={path} line {line_no} has {len(row)} columns, expected {expected}")
        rows.append(dict(zip(fieldnames, row)))
    return rows


def _read_csv_rows(path: Path, robot_spec: RobotSpec) -> tuple[list[str], list[dict[str, str]], bool]:
    with path.open("r", newline="") as f:
        raw_rows = [row for row in csv.reader(f) if any(cell.strip() for cell in row)]
    if not raw_rows:
        raise ValueError(f"CSV file is empty: {path}")
    first = raw_rows[0]
    if _is_numeric_row(first):
        fieldnames = _headerless_csv_fieldnames(path, len(first), len(robot_spec.control_joint_names))
        return fieldnames, _dict_rows(fieldnames, raw_rows, path), True
    fieldnames = [str(value) for value in first]
    rows = _dict_rows(fieldnames, raw_rows[1:], path)
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    return fieldnames, rows, False


def _column_array(rows: list[dict[str, str]], column: str, path: Path) -> np.ndarray:
    try:
        return np.asarray([float(row[column]) for row in rows], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"CSV file={path} missing required column '{column}'") from exc
    except ValueError as exc:
        raise ValueError(f"CSV file={path} column '{column}' contains non-numeric values") from exc


def _infer_fps_from_time(rows: list[dict[str, str]], path: Path) -> float:
    try:
        times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"CSV file={path} missing required column 'time'") from exc
    except ValueError as exc:
        raise ValueError(f"CSV file={path} column 'time' contains non-numeric values") from exc
    if len(times) < 2:
        raise ValueError(f"CSV file={path} needs at least two time samples to infer fps")
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError(f"CSV file={path} time column must be strictly increasing")
    median_dt = float(np.median(dt))
    if median_dt <= 0.0 or not np.isfinite(median_dt):
        raise ValueError(f"CSV file={path} has invalid median dt={median_dt}")
    rel_jitter = float(np.max(np.abs(dt - median_dt)) / median_dt)
    if rel_jitter > 0.05:
        logger.warning(f"CSV file={path} has non-uniform time intervals; max relative jitter={rel_jitter:.3f}")
    return 1.0 / median_dt


def _fps_from_rows(fieldnames: list[str], rows: list[dict[str, str]], path: Path, fps: float | int | None) -> float:
    if fps is not None:
        motion_fps = float(fps)
        if not np.isfinite(motion_fps) or motion_fps <= 0.0:
            raise ValueError(f"CSV file={path} has invalid manifest fps={fps}")
        return motion_fps
    if "time" in fieldnames:
        return _infer_fps_from_time(rows, path)
    raise ValueError(f"CSV file={path} requires a time column or manifest fps")


def _columns_matrix(rows: list[dict[str, str]], columns: list[str], path: Path) -> np.ndarray:
    return np.stack([_column_array(rows, column, path) for column in columns], axis=1).astype(np.float32)


def _robot_state_columns(columns: dict[str, Any] | None) -> tuple[list[str], list[str], Any]:
    config = dict(columns or {})
    root_pos_columns = config.get("root_pos", ["root_pos_x", "root_pos_y", "root_pos_z"])
    root_quat_columns = config.get("root_quat", ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"])
    dof_spec = config.get("dof_pos", "auto_by_joint_name")
    if not isinstance(root_pos_columns, list) or len(root_pos_columns) != 3:
        raise ValueError("robot_state_csv columns.root_pos must be a list of three column names")
    if not isinstance(root_quat_columns, list) or len(root_quat_columns) != 4:
        raise ValueError("robot_state_csv columns.root_quat must be a list of four column names")
    return [str(v) for v in root_pos_columns], [str(v) for v in root_quat_columns], dof_spec


def read_robot_state_csv(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
    columns: dict[str, Any] | None = None,
) -> dict[str, RobotStateMotion]:
    data: dict[str, RobotStateMotion] = {}
    configured_root_pos_columns, configured_root_quat_columns, configured_dof_spec = _robot_state_columns(columns)
    for path in expand_motion_paths(path_spec, base_dir=base_dir, suffix=".csv"):
        fieldnames, rows, headerless = _read_csv_rows(path, robot_spec)
        root_pos_columns = configured_root_pos_columns
        root_quat_columns = configured_root_quat_columns
        dof_spec = "xml_order" if headerless and columns is None else configured_dof_spec
        root_pos = _columns_matrix(rows, root_pos_columns, path)
        root_quat = _columns_matrix(rows, root_quat_columns, path)
        joint_names: list[str] | None = None
        if dof_spec == "auto_by_joint_name":
            missing = [joint for joint in robot_spec.control_joint_names if joint not in fieldnames]
            if missing:
                raise ValueError(f"robot_state_csv file={path} missing control joint columns: {missing}")
            joint_names = list(robot_spec.control_joint_names)
            dof_pos = _columns_matrix(rows, joint_names, path)
        elif dof_spec == "xml_order":
            dof_columns = [f"dof_{idx}" for idx in range(len(robot_spec.control_joint_names))]
            missing = [column for column in dof_columns if column not in fieldnames]
            if missing:
                raise ValueError(f"robot_state_csv file={path} missing xml_order dof columns: {missing}")
            dof_pos = _columns_matrix(rows, dof_columns, path)
        elif isinstance(dof_spec, list):
            if len(dof_spec) != len(robot_spec.control_joint_names):
                raise ValueError(
                    f"robot_state_csv columns.dof_pos list length must match control joints "
                    f"({len(robot_spec.control_joint_names)}), got {len(dof_spec)}"
                )
            dof_pos = _columns_matrix(rows, [str(column) for column in dof_spec], path)
        else:
            raise ValueError("robot_state_csv columns.dof_pos must be auto_by_joint_name, xml_order, or a column list")

        motion_key = path.stem
        if motion_key in data:
            raise ValueError(f"Duplicate motion_key={motion_key} while reading robot_state_csv source={source_name}")
        data[motion_key] = RobotStateMotion(
            motion_key=motion_key,
            root_pos=root_pos,
            root_quat=root_quat,
            dof_pos=dof_pos,
            fps=_fps_from_rows(fieldnames, rows, path, fps),
            joint_names=joint_names,
            source=source_name,
            metadata={
                "path": str(path),
                "reader": "robot_state_csv",
                "columns": {
                    "root_pos": root_pos_columns,
                    "root_quat": root_quat_columns,
                    "dof_pos": dof_spec,
                },
            },
        )
    return validate_robot_state_dict(data, robot_spec, source_name)


def _string_list_from_npz(value: Any) -> list[str]:
    arr = np.asarray(value)
    values: list[str] = []
    for item in arr.reshape(-1):
        if isinstance(item, bytes):
            values.append(item.decode("utf-8"))
        else:
            values.append(str(item))
    return values


def _fps_from_npz(npz: Any, path: Path, fps: float | int | None) -> float:
    if fps is not None:
        motion_fps = float(fps)
    elif "fps" in npz:
        motion_fps = float(_scalar_from_npz(npz["fps"]))
    elif "framerate" in npz:
        motion_fps = float(_scalar_from_npz(npz["framerate"]))
    elif "time" in npz:
        times = np.asarray(npz["time"], dtype=np.float64)
        if times.ndim != 1 or times.shape[0] < 2:
            raise ValueError(f"robot_state_npz file={path} time must have shape [T] with at least two samples")
        dt = np.diff(times)
        if np.any(dt <= 0.0):
            raise ValueError(f"robot_state_npz file={path} time must be strictly increasing")
        motion_fps = 1.0 / float(np.median(dt))
    else:
        raise ValueError(f"robot_state_npz file={path} requires fps, time, or manifest fps")
    if not np.isfinite(motion_fps) or motion_fps <= 0.0:
        raise ValueError(f"robot_state_npz file={path} has invalid fps={motion_fps}")
    return motion_fps


def read_robot_state_npz(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
) -> dict[str, RobotStateMotion]:
    data: dict[str, RobotStateMotion] = {}
    for path in expand_motion_paths(path_spec, base_dir=base_dir, suffix=".npz"):
        with np.load(path, allow_pickle=True) as npz:
            source_fields = {
                "root_pos": "root_pos" if "root_pos" in npz else "base_pos_w" if "base_pos_w" in npz else None,
                "root_quat": "root_quat" if "root_quat" in npz else "base_quat_w" if "base_quat_w" in npz else None,
                "dof_pos": "dof_pos" if "dof_pos" in npz else "joint_pos" if "joint_pos" in npz else None,
            }
            missing = [field for field, source_field in source_fields.items() if source_field is None]
            if missing:
                raise ValueError(f"robot_state_npz source={source_name}, file={path}: missing fields {missing}")
            joint_names = _string_list_from_npz(npz["joint_names"]) if "joint_names" in npz else None
            if joint_names is None:
                logger.warning(
                    f"robot_state_npz file={path} has no joint_names; assuming dof_pos is ordered as RobotSpec.control_joint_names"
                )
            motion_key = _motion_key_from_npz(npz["motion_key"]) if "motion_key" in npz else path.stem
            if motion_key in data:
                raise ValueError(f"Duplicate motion_key={motion_key} while reading robot_state_npz source={source_name}")
            root_quat = np.asarray(npz[source_fields["root_quat"]], dtype=np.float32)
            if source_fields["root_quat"] == "base_quat_w" and robot_spec.root_quat_order == "xyzw":
                root_quat = root_quat[:, [1, 2, 3, 0]]
            metadata = {"path": str(path), "reader": "robot_state_npz", "source_fields": source_fields}
            if "tpose_trimmed_frames" in npz:
                metadata["tpose_trimmed_frames"] = int(_scalar_from_npz(npz["tpose_trimmed_frames"]))
            data[motion_key] = RobotStateMotion(
                motion_key=motion_key,
                root_pos=np.asarray(npz[source_fields["root_pos"]], dtype=np.float32),
                root_quat=root_quat,
                dof_pos=np.asarray(npz[source_fields["dof_pos"]], dtype=np.float32),
                fps=_fps_from_npz(npz, path, fps),
                joint_names=joint_names,
                source=source_name,
                metadata=metadata,
            )
    return validate_robot_state_dict(data, robot_spec, source_name)
