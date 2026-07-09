"""Convert RobotState motions into UFO MotionLib motion dictionaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from humanoidverse.utils.motion_data.robot_state import RobotStateMotion, validate_robot_state_dict, validate_robot_state_motion
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import RobotSpec


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


def robot_state_to_ufo_motion(
    motion: RobotStateMotion,
    robot_spec: RobotSpec,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Convert one validated RobotStateMotion into the UFO pose_aa schema."""

    effective_source = source_name or motion.source or "robot_state"
    motion = validate_robot_state_motion(motion, robot_spec, effective_source)
    dof = motion.dof_pos.astype(np.float32)
    if robot_spec.dof_unit == "deg":
        dof = np.deg2rad(dof)
    elif robot_spec.dof_unit != "rad":
        raise ValueError(f"Unsupported dof_unit={robot_spec.dof_unit!r}; expected 'rad' or 'deg'")

    pose_aa = np.zeros((motion.root_pos.shape[0], len(robot_spec.body_names), 3), dtype=np.float32)
    body_to_idx = {name: idx for idx, name in enumerate(robot_spec.body_names)}
    if robot_spec.base_body not in body_to_idx:
        raise ValueError(f"RobotSpec base_body={robot_spec.base_body!r} is not present in body_names")
    pose_aa[:, body_to_idx[robot_spec.base_body]] = _quat_xyzw_to_axis_angle(
        _root_quat_to_xyzw(motion.root_quat, robot_spec.root_quat_order)
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

    metadata = dict(motion.metadata or {})
    metadata.update(
        {
            "source_name": effective_source,
            "motion_key": motion.motion_key,
            "robot_name": robot_spec.name,
            "xml_path": robot_spec.xml_path,
            "root_quat_order": robot_spec.root_quat_order,
            "coordinate_system": robot_spec.coordinate_system,
            "dof_unit": robot_spec.dof_unit,
        }
    )
    return {
        "root_trans_offset": motion.root_pos.astype(np.float32),
        "pose_aa": pose_aa,
        "fps": float(motion.fps),
        "dof_pos": dof.astype(np.float32),
        "root_quat": motion.root_quat.astype(np.float32),
        "joint_names": list(robot_spec.control_joint_names),
        "body_names": list(robot_spec.body_names),
        "robot_name": robot_spec.name,
        "motion_key": motion.motion_key,
        "source": motion.source or effective_source,
        "metadata": metadata,
    }


def robot_state_dict_to_ufo_motion_dict(
    data: dict[str, RobotStateMotion],
    robot_spec: RobotSpec,
    source_name: str,
) -> dict[str, Any]:
    validated = validate_robot_state_dict(data, robot_spec, source_name)
    converted = {
        motion_key: robot_state_to_ufo_motion(motion, robot_spec, source_name)
        for motion_key, motion in validated.items()
    }
    return validate_ufo_motion_dict(converted, source_name)
