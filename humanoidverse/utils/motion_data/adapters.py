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
from humanoidverse.utils.robot_spec import RobotSpec


SUPPORTED_FORMATS = {"ufo_pkl", "ufo_npz", "csv_ufo", "robot_state_csv", "robot_state_npz"}


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


def _fps_from_rows(fieldnames: list[str], rows: list[dict[str, str]], path: Path, fps: float | int | None) -> float:
    if fps is not None:
        motion_fps = float(fps)
        if not np.isfinite(motion_fps) or motion_fps <= 0.0:
            raise ValueError(f"CSV file={path} has invalid manifest fps={fps}")
        return motion_fps
    if "time" in fieldnames:
        return _infer_fps_from_time(rows, path)
    raise ValueError(f"CSV file={path} requires a time column or manifest fps")


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
    if joint_indices != list(range(len(joint_indices))):
        raise ValueError(f"csv_ufo file={path} pose_aa joint indices must be contiguous from 0")
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


def _require_robot_spec(fmt: str, robot_spec: RobotSpec | None) -> RobotSpec:
    if robot_spec is None:
        raise ValueError(f"{fmt} requires a robot_config/RobotSpec")
    return robot_spec


def _root_quat_to_xyzw(root_quat: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(root_quat, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"root_quat must have shape [T, 4], got {quat.shape}")
    if order == "xyzw":
        xyzw = quat.copy()
    elif order == "wxyz":
        xyzw = quat[:, [1, 2, 3, 0]].copy()
    else:
        raise ValueError(f"Unsupported root_quat_order={order!r}")
    norm = np.linalg.norm(xyzw, axis=1, keepdims=True)
    if np.any(norm <= 0.0) or np.any(~np.isfinite(norm)):
        raise ValueError("root_quat contains zero or non-finite quaternions")
    return xyzw / norm


def _quat_xyzw_to_axis_angle(root_quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(root_quat_xyzw, dtype=np.float64)
    vec = quat[:, :3]
    w = np.clip(quat[:, 3], -1.0, 1.0)
    vec_norm = np.linalg.norm(vec, axis=1)
    angle = 2.0 * np.arctan2(vec_norm, w)
    axis = np.zeros_like(vec)
    valid = vec_norm > 1e-8
    axis[valid] = vec[valid] / vec_norm[valid, None]
    return (axis * angle[:, None]).astype(np.float32)


def _as_float_matrix(value: Any, field_name: str, width: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{field_name} must have shape [T, {width}], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} contains non-finite values")
    return arr


def robot_state_to_ufo_motion(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    dof_pos: np.ndarray,
    robot_spec: RobotSpec,
    source_name: str,
    motion_key: str,
    *,
    fps: float | int,
) -> dict[str, Any]:
    """Convert root/dof state into the current UFO pose_aa motion schema.

    Existing G1 MotionLib data uses pose_aa[:, 0] as the floating-base body
    rotation and one axis-angle entry per non-world MuJoCo body. Hinge joint
    angles are therefore written to the body attached to each XML joint as
    joint_axis * joint_angle.
    """

    root = _as_float_matrix(root_pos, "root_pos", 3)
    quat = _as_float_matrix(root_quat, "root_quat", 4)
    dof = np.asarray(dof_pos, dtype=np.float32)
    expected_dofs = len(robot_spec.control_joint_names)
    if dof.ndim != 2 or dof.shape[1] != expected_dofs:
        raise ValueError(f"dof_pos must have shape [T, {expected_dofs}], got {dof.shape}")
    if root.shape[0] != quat.shape[0] or root.shape[0] != dof.shape[0]:
        raise ValueError(f"root_pos/root_quat/dof_pos must share T, got {root.shape[0]}, {quat.shape[0]}, {dof.shape[0]}")
    if robot_spec.dof_unit == "deg":
        dof = np.deg2rad(dof)
    elif robot_spec.dof_unit != "rad":
        raise ValueError(f"Unsupported dof_unit={robot_spec.dof_unit!r}; expected 'rad' or 'deg'")

    motion_fps = float(fps)
    if not np.isfinite(motion_fps) or motion_fps <= 0.0:
        raise ValueError(f"fps must be > 0, got {fps}")

    pose_aa = np.zeros((root.shape[0], len(robot_spec.body_names), 3), dtype=np.float32)
    body_to_idx = {name: idx for idx, name in enumerate(robot_spec.body_names)}
    if robot_spec.base_body not in body_to_idx:
        raise ValueError(f"RobotSpec base_body={robot_spec.base_body!r} is not present in body_names")
    pose_aa[:, body_to_idx[robot_spec.base_body]] = _quat_xyzw_to_axis_angle(
        _root_quat_to_xyzw(quat, robot_spec.root_quat_order)
    )

    for dof_idx, joint_name in enumerate(robot_spec.control_joint_names):
        joint_type = robot_spec.joint_types[joint_name]
        if joint_type != "hinge":
            raise ValueError(f"robot_state_to_ufo_motion currently supports hinge control joints only, got {joint_name}:{joint_type}")
        body_name = robot_spec.joint_body_names[joint_name]
        if body_name not in body_to_idx:
            raise ValueError(f"Control joint {joint_name} attaches to body {body_name}, which is not in RobotSpec.body_names")
        axis = np.asarray(robot_spec.joint_axes[joint_name], dtype=np.float32)
        pose_aa[:, body_to_idx[body_name]] = dof[:, dof_idx : dof_idx + 1] * axis[None, :]

    return {
        "root_trans_offset": root.astype(np.float32),
        "pose_aa": pose_aa,
        "fps": motion_fps,
        "dof_pos": dof.astype(np.float32),
        "root_quat": quat.astype(np.float32),
        "joint_names": list(robot_spec.control_joint_names),
        "body_names": list(robot_spec.body_names),
        "robot_name": robot_spec.name,
        "motion_key": motion_key,
        "source": source_name,
        "metadata": {
            "source_name": source_name,
            "motion_key": motion_key,
            "robot_name": robot_spec.name,
            "xml_path": robot_spec.xml_path,
            "root_quat_order": robot_spec.root_quat_order,
            "coordinate_system": robot_spec.coordinate_system,
            "dof_unit": robot_spec.dof_unit,
        },
    }


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


def load_robot_state_csv(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
    columns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    root_pos_columns, root_quat_columns, dof_spec = _robot_state_columns(columns)
    for path in expand_motion_paths(path_spec, base_dir=base_dir, suffix=".csv"):
        fieldnames, rows = _read_csv_rows(path)
        root_pos = _columns_matrix(rows, root_pos_columns, path)
        root_quat = _columns_matrix(rows, root_quat_columns, path)
        if dof_spec == "auto_by_joint_name":
            missing = [joint for joint in robot_spec.control_joint_names if joint not in fieldnames]
            if missing:
                raise ValueError(f"robot_state_csv file={path} missing control joint columns: {missing}")
            dof_pos = _columns_matrix(rows, list(robot_spec.control_joint_names), path)
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
            raise ValueError(f"Duplicate motion_key={motion_key} while loading robot_state_csv source={source_name}")
        data[motion_key] = robot_state_to_ufo_motion(
            root_pos,
            root_quat,
            dof_pos,
            robot_spec,
            source_name,
            motion_key,
            fps=_fps_from_rows(fieldnames, rows, path, fps),
        )
    return validate_ufo_motion_dict(data, source_name)


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


def load_robot_state_npz(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for path in expand_motion_paths(path_spec, base_dir=base_dir, suffix=".npz"):
        with np.load(path, allow_pickle=True) as npz:
            missing = [field for field in ("root_pos", "root_quat", "dof_pos") if field not in npz]
            if missing:
                raise ValueError(f"robot_state_npz source={source_name}, file={path}: missing fields {missing}")
            dof_pos = np.asarray(npz["dof_pos"], dtype=np.float32)
            if "joint_names" in npz:
                joint_names = _string_list_from_npz(npz["joint_names"])
                joint_to_idx = {name: idx for idx, name in enumerate(joint_names)}
                missing_joints = [joint for joint in robot_spec.control_joint_names if joint not in joint_to_idx]
                if missing_joints:
                    raise ValueError(f"robot_state_npz file={path} missing joint_names entries: {missing_joints}")
                dof_pos = dof_pos[:, [joint_to_idx[joint] for joint in robot_spec.control_joint_names]]
            else:
                logger.warning(
                    f"robot_state_npz file={path} has no joint_names; assuming dof_pos is ordered as RobotSpec.control_joint_names"
                )

            motion_key = str(_scalar_from_npz(npz["motion_key"])) if "motion_key" in npz else path.stem
            if motion_key in data:
                raise ValueError(f"Duplicate motion_key={motion_key} while loading robot_state_npz source={source_name}")
            data[motion_key] = robot_state_to_ufo_motion(
                np.asarray(npz["root_pos"], dtype=np.float32),
                np.asarray(npz["root_quat"], dtype=np.float32),
                dof_pos,
                robot_spec,
                source_name,
                motion_key,
                fps=_fps_from_npz(npz, path, fps),
            )
    return validate_ufo_motion_dict(data, source_name)


def load_motion_data_by_format(
    fmt: str,
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    base_dir: Path | None = None,
    fps: float | int | None = None,
    robot_spec: RobotSpec | None = None,
    columns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fmt = str(fmt)
    if fmt == "ufo_pkl":
        return load_ufo_pkl(path_spec, source_name=source_name, base_dir=base_dir)
    if fmt == "ufo_npz":
        return load_ufo_npz(path_spec, source_name=source_name, base_dir=base_dir)
    if fmt == "csv_ufo":
        return load_csv_ufo(path_spec, source_name=source_name, base_dir=base_dir, fps=fps)
    if fmt == "robot_state_csv":
        return load_robot_state_csv(
            path_spec,
            source_name=source_name,
            robot_spec=_require_robot_spec(fmt, robot_spec),
            base_dir=base_dir,
            fps=fps,
            columns=columns,
        )
    if fmt == "robot_state_npz":
        return load_robot_state_npz(
            path_spec,
            source_name=source_name,
            robot_spec=_require_robot_spec(fmt, robot_spec),
            base_dir=base_dir,
            fps=fps,
        )
    raise ValueError(f"Unsupported motion data format '{fmt}'. Supported formats: {sorted(SUPPORTED_FORMATS)}")


def dump_ufo_pkl(data: dict[str, Any], output_path: Path, source_name: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_ufo_motion_dict(data, source_name)
    joblib.dump(validated, output_path)
    return output_path
