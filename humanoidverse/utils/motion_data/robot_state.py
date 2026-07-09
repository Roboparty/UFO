"""RobotState schema used by UFO motion data readers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from humanoidverse.utils.robot_spec import RobotSpec


@dataclass(frozen=True)
class RobotStateMotion:
    motion_key: str
    root_pos: np.ndarray
    root_quat: np.ndarray
    dof_pos: np.ndarray
    fps: float
    joint_names: list[str] | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _context(source_name: str, motion_key: str | None = None) -> str:
    return f"{source_name}:{motion_key}" if motion_key else source_name


def _as_float_matrix(value: Any, field_name: str, width: int, context: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{context} {field_name} must have shape [T, {width}], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{context} {field_name} contains non-finite values")
    return arr


def reorder_dof_by_joint_names(
    dof_pos: np.ndarray,
    input_joint_names: list[str],
    target_joint_names: list[str],
    source_name: str,
) -> np.ndarray:
    dof = np.asarray(dof_pos, dtype=np.float32)
    if dof.ndim != 2:
        raise ValueError(f"{source_name} dof_pos must have shape [T, N], got {dof.shape}")
    if len(input_joint_names) != dof.shape[1]:
        raise ValueError(
            f"{source_name} joint_names length must match dof_pos width, "
            f"got {len(input_joint_names)} names for width {dof.shape[1]}"
        )
    if len(set(input_joint_names)) != len(input_joint_names):
        raise ValueError(f"{source_name} joint_names contains duplicates")
    extra = [joint for joint in input_joint_names if joint not in target_joint_names]
    if extra:
        raise ValueError(f"{source_name} unexpected joint_names entries: {extra}")
    joint_to_idx = {name: idx for idx, name in enumerate(input_joint_names)}
    missing = [joint for joint in target_joint_names if joint not in joint_to_idx]
    if missing:
        raise ValueError(f"{source_name} missing joint_names entries: {missing}")
    return dof[:, [joint_to_idx[joint] for joint in target_joint_names]].copy()


def validate_robot_state_motion(
    motion: RobotStateMotion,
    robot_spec: RobotSpec,
    source_name: str,
) -> RobotStateMotion:
    context = _context(source_name, motion.motion_key)
    root = _as_float_matrix(motion.root_pos, "root_pos", 3, context)
    quat = _as_float_matrix(motion.root_quat, "root_quat", 4, context)
    dof = np.asarray(motion.dof_pos, dtype=np.float32)
    if dof.ndim != 2:
        raise ValueError(f"{context} dof_pos must have shape [T, N], got {dof.shape}")
    if not np.all(np.isfinite(dof)):
        raise ValueError(f"{context} dof_pos contains non-finite values")
    if root.shape[0] != quat.shape[0] or root.shape[0] != dof.shape[0]:
        raise ValueError(
            f"{context} root_pos/root_quat/dof_pos must share T, "
            f"got {root.shape[0]}, {quat.shape[0]}, {dof.shape[0]}"
        )

    fps = float(motion.fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{context} fps must be > 0, got {motion.fps}")

    quat_norm = np.linalg.norm(quat, axis=1)
    if np.any(quat_norm <= 0.0) or np.any(~np.isfinite(quat_norm)):
        raise ValueError(f"{context} root_quat contains zero or non-finite quaternions")

    target_joint_names = list(robot_spec.control_joint_names)
    if motion.joint_names is not None:
        dof = reorder_dof_by_joint_names(dof, list(motion.joint_names), target_joint_names, context)
        joint_names: list[str] | None = target_joint_names
    else:
        expected_dofs = len(target_joint_names)
        if dof.shape[1] != expected_dofs:
            raise ValueError(f"{context} dof_pos must have shape [T, {expected_dofs}], got {dof.shape}")
        joint_names = None

    return RobotStateMotion(
        motion_key=str(motion.motion_key),
        root_pos=root,
        root_quat=quat,
        dof_pos=dof,
        fps=fps,
        joint_names=joint_names,
        source=motion.source or source_name,
        metadata=dict(motion.metadata or {}),
    )


def validate_robot_state_dict(
    data: dict[str, RobotStateMotion],
    robot_spec: RobotSpec,
    source_name: str,
) -> dict[str, RobotStateMotion]:
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{source_name} RobotState data must be a non-empty dict")
    validated: dict[str, RobotStateMotion] = {}
    for key, motion in data.items():
        if not isinstance(motion, RobotStateMotion):
            raise ValueError(f"{source_name}:{key} must be a RobotStateMotion")
        canonical = validate_robot_state_motion(motion, robot_spec, source_name)
        motion_key = str(canonical.motion_key)
        if motion_key in validated:
            raise ValueError(f"Duplicate RobotState motion_key={motion_key} in source={source_name}")
        if motion_key != str(key):
            raise ValueError(f"RobotState dict key={key!r} does not match motion_key={motion_key!r}")
        validated[motion_key] = canonical
    return validated
