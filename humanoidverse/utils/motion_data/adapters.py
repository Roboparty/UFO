"""Adapters that convert normalized motion sources into UFO motion dicts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib

from humanoidverse.utils.motion_data.paths import expand_motion_paths
from humanoidverse.utils.motion_data.robot_state_convert import robot_state_dict_to_ufo_motion_dict
from humanoidverse.utils.motion_data.robot_state_readers import read_robot_state_csv, read_robot_state_npz
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import RobotSpec


SUPPORTED_FORMATS = {"ufo_pkl", "robot_state_csv", "robot_state_npz"}


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


def _require_robot_spec(fmt: str, robot_spec: RobotSpec | None) -> RobotSpec:
    if robot_spec is None:
        raise ValueError(f"{fmt} requires a robot_config/RobotSpec")
    return robot_spec


def load_robot_state_csv(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
    columns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    robot_state = read_robot_state_csv(
        path_spec,
        source_name=source_name,
        robot_spec=robot_spec,
        base_dir=base_dir,
        fps=fps,
        columns=columns,
    )
    return robot_state_dict_to_ufo_motion_dict(robot_state, robot_spec, source_name)


def load_robot_state_npz(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
) -> dict[str, Any]:
    robot_state = read_robot_state_npz(
        path_spec,
        source_name=source_name,
        robot_spec=robot_spec,
        base_dir=base_dir,
        fps=fps,
    )
    return robot_state_dict_to_ufo_motion_dict(robot_state, robot_spec, source_name)


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
