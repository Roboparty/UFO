from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import mujoco as mj
import numpy as np
import yaml

from .params import resolve_robot_xml_path


UFO_EXPECTED_G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

UFO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_CONFIG = UFO_ROOT / "config" / "policy" / "g1_policy.yaml"


def _duplicates(names: Sequence[str]) -> list[str]:
    counts = Counter(str(name) for name in names)
    return sorted(name for name, count in counts.items() if count > 1)


def validate_joint_name_set(
    canonical_names: Sequence[str],
    output_names: Sequence[str],
) -> None:
    canonical = tuple(str(name) for name in canonical_names)
    output = tuple(str(name) for name in output_names)
    canonical_dupes = _duplicates(canonical)
    output_dupes = _duplicates(output)
    if canonical_dupes:
        raise ValueError(f"duplicate canonical joint names: {canonical_dupes}")
    if output_dupes:
        raise ValueError(f"duplicate UFO output joint names: {output_dupes}")

    canonical_set = set(canonical)
    output_set = set(output)
    missing = sorted(output_set - canonical_set)
    extra = sorted(canonical_set - output_set)
    if missing or extra:
        raise ValueError(f"joint name mismatch: missing={missing}, extra={extra}")


def build_joint_permutation(
    canonical_names: Sequence[str],
    output_names: Sequence[str] = UFO_EXPECTED_G1_JOINT_NAMES,
) -> np.ndarray:
    canonical = tuple(str(name) for name in canonical_names)
    output = tuple(str(name) for name in output_names)
    validate_joint_name_set(canonical, output)
    canonical_index = {name: idx for idx, name in enumerate(canonical)}
    return np.asarray([canonical_index[name] for name in output], dtype=np.int64)


def policy_joint_names(policy_config_path: str | Path = DEFAULT_POLICY_CONFIG) -> tuple[str, ...]:
    cfg_path = Path(policy_config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("policy_joint_names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"policy_joint_names missing or invalid in {cfg_path}")
    return tuple(str(name) for name in names)


def canonical_joint_names(target_robot: str = "g1") -> tuple[str, ...]:
    model = mj.MjModel.from_xml_path(str(resolve_robot_xml_path(target_robot)))
    names: list[str] = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mj.mjtJoint.mjJNT_FREE):
            continue
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            raise ValueError(f"failed to resolve canonical joint name at index {joint_id}")
        names.append(str(joint_name))
    return tuple(names)


def qpos_size(target_robot: str = "g1") -> int:
    model = mj.MjModel.from_xml_path(str(resolve_robot_xml_path(target_robot)))
    return int(model.nq)
