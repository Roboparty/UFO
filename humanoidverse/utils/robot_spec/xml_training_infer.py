"""Infer draft RobotTrainingSpec and Hydra robot configs from MuJoCo XML.

The generated configs are intentionally conservative drafts. They extract
structure from XML, but semantic groups and training parameters must be reviewed
before formal training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import mujoco


def _mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return str(name) if name else f"{obj_type.name.lower()}_{obj_id}"


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value)
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _contains_any(name: str, tokens: Iterable[str]) -> bool:
    lower = name.lower()
    return any(token in lower for token in tokens)


def _is_left(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("left") or "_left" in lower or lower.startswith("l_") or "_l_" in lower


def _is_right(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("right") or "_right" in lower or lower.startswith("r_") or "_r_" in lower


def load_mujoco_model(xml_path: str | Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser()))


def infer_control_joints_from_xml(model: mujoco.MjModel) -> tuple[list[str], list[str]]:
    """Return joint-transmission actuator joints in MuJoCo actuator order."""

    joints: list[str] = []
    warnings: list[str] = []
    non_joint_actuators: list[str] = []
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            non_joint_actuators.append(_mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id))
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            non_joint_actuators.append(_mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id))
            continue
        joints.append(_mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    joints = _ordered_unique(joints)
    if not joints:
        raise ValueError("MuJoCo XML has no joint-transmission actuators; cannot infer control joints for training.")
    if non_joint_actuators:
        warnings.append(f"Ignored non-joint actuators while inferring control joints: {non_joint_actuators}")
    if len(joints) < model.nu:
        warnings.append("Control joints were inferred only from joint-transmission actuators.")
    return joints, warnings


def infer_body_names_from_xml(model: mujoco.MjModel) -> list[str]:
    return [_mj_name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(1, model.nbody)]


def infer_freejoint_base_body(model: mujoco.MjModel) -> tuple[str, str | None, list[str]]:
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            joint_name = _mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            body_name = _mj_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[joint_id]))
            return body_name, joint_name, []
    body_names = infer_body_names_from_xml(model)
    fallback = body_names[0] if body_names else ""
    return fallback, None, [f"No freejoint found in XML; using first non-world body as base_body={fallback!r}. Training a floating humanoid usually requires a freejoint."]


def _joint_id_map(model: mujoco.MjModel) -> dict[str, int]:
    return {_mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, idx): idx for idx in range(model.njnt)}


def _actuator_for_joint(model: mujoco.MjModel, joint_name: str) -> int | None:
    joint_ids = _joint_id_map(model)
    joint_id = joint_ids[joint_name]
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        if int(model.actuator_trnid[actuator_id, 0]) == joint_id:
            return actuator_id
    return None


def infer_joint_ranges_from_xml(model: mujoco.MjModel, control_joints: list[str]) -> tuple[dict[str, list[float]], list[str], list[list[float]]]:
    joint_ids = _joint_id_map(model)
    ranges: dict[str, list[float]] = {}
    lower: list[float] = []
    upper: list[float] = []
    warnings: list[str] = []
    for joint_name in control_joints:
        joint_id = joint_ids[joint_name]
        if int(model.jnt_limited[joint_id]):
            lo, hi = (float(model.jnt_range[joint_id, 0]), float(model.jnt_range[joint_id, 1]))
        else:
            lo, hi = -3.141592653589793, 3.141592653589793
            warnings.append(f"Joint {joint_name!r} has no XML range; using [-pi, pi] draft limits.")
        ranges[joint_name] = [lo, hi]
        lower.append(lo)
        upper.append(hi)
    return ranges, warnings, [lower, upper]


def infer_actuator_limits_from_xml(
    model: mujoco.MjModel,
    control_joints: list[str],
    *,
    effort_limit_default: float = 80.0,
    velocity_limit_default: float = 20.0,
    armature_default: float = 0.01,
    friction_default: float = 0.0,
) -> tuple[list[float], list[float], dict[str, dict[str, float]], list[str]]:
    joint_ids = _joint_id_map(model)
    warnings: list[str] = []
    effort_limits: list[float] = []
    velocity_limits: list[float] = []
    actuator_joints: dict[str, dict[str, float]] = {}
    for joint_name in control_joints:
        joint_id = joint_ids[joint_name]
        actuator_id = _actuator_for_joint(model, joint_name)
        effort_limit: float | None = None
        if actuator_id is not None:
            force_range = [float(v) for v in model.actuator_forcerange[actuator_id].tolist()]
            if bool(model.actuator_forcelimited[actuator_id]) and force_range[0] != force_range[1]:
                effort_limit = max(abs(force_range[0]), abs(force_range[1]))
            else:
                ctrl_range = [float(v) for v in model.actuator_ctrlrange[actuator_id].tolist()]
                if bool(model.actuator_ctrllimited[actuator_id]) and ctrl_range[0] != ctrl_range[1]:
                    effort_limit = max(abs(ctrl_range[0]), abs(ctrl_range[1]))
                    warnings.append(f"Joint {joint_name!r} actuator has no force range; using ctrlrange as draft effort_limit.")
        if effort_limit is None or effort_limit <= 0.0:
            effort_limit = float(effort_limit_default)
            warnings.append(f"Joint {joint_name!r} has no actuator force/ctrl range; using effort_limit_default={effort_limit_default}.")
        velocity_limit = float(velocity_limit_default)
        warnings.append(f"Joint {joint_name!r} velocity limit is not reliably represented in MJCF; using velocity_limit_default={velocity_limit_default}.")
        dof_addr = int(model.jnt_dofadr[joint_id])
        armature = float(model.dof_armature[dof_addr]) if dof_addr >= 0 else float(armature_default)
        friction = float(model.dof_frictionloss[dof_addr]) if dof_addr >= 0 else float(friction_default)
        if armature == 0.0:
            armature = float(armature_default)
            warnings.append(f"Joint {joint_name!r} has zero XML armature; using armature_default={armature_default}.")
        if friction == 0.0 and friction_default != 0.0:
            friction = float(friction_default)
            warnings.append(f"Joint {joint_name!r} has zero XML frictionloss; using friction_default={friction_default}.")
        effort_limits.append(float(effort_limit))
        velocity_limits.append(velocity_limit)
        actuator_joints[joint_name] = {
            "effort_limit": float(effort_limit),
            "velocity_limit": velocity_limit,
            "armature": armature,
            "friction": friction,
        }
    return effort_limits, velocity_limits, actuator_joints, warnings


def _keyframe_qpos(model: mujoco.MjModel, keyframe_names: tuple[str, ...]) -> tuple[list[float], str | None]:
    requested = [name for name in keyframe_names if name]
    for preferred in requested:
        for key_id in range(model.nkey):
            name = _mj_name(model, mujoco.mjtObj.mjOBJ_KEY, key_id)
            if name == preferred:
                return [float(v) for v in model.key_qpos[key_id].tolist()], name
    for key_id in range(model.nkey):
        name = _mj_name(model, mujoco.mjtObj.mjOBJ_KEY, key_id)
        if name in {"stand", "default", "home", "init"}:
            return [float(v) for v in model.key_qpos[key_id].tolist()], name
    return [float(v) for v in model.qpos0.tolist()], None


def _freejoint_pose_from_qpos(model: mujoco.MjModel, qpos: list[float]) -> tuple[list[float], list[float], list[str]]:
    warnings: list[str] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        pos = [float(v) for v in qpos[qpos_addr : qpos_addr + 3]]
        quat_wxyz = [float(v) for v in qpos[qpos_addr + 3 : qpos_addr + 7]]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        return pos, quat_xyzw, warnings
    warnings.append("No freejoint qpos found; using default init_state pos/rot.")
    return [0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], warnings


def infer_default_pose_from_xml(
    model: mujoco.MjModel,
    control_joints: list[str],
    keyframe_names: tuple[str, ...] = ("stand", "default", "home", "init"),
) -> tuple[dict[str, float], dict[str, list[float]], list[str]]:
    qpos, keyframe_name = _keyframe_qpos(model, keyframe_names)
    warnings: list[str] = []
    if keyframe_name is None:
        warnings.append("No stand/default/home/init keyframe found; using model.qpos0 for default pose.")
    else:
        warnings.append(f"Default pose inferred from MuJoCo keyframe {keyframe_name!r}; review before training.")
    joint_ids = _joint_id_map(model)
    default_angles: dict[str, float] = {}
    for joint_name in control_joints:
        joint_id = joint_ids[joint_name]
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            warnings.append(f"Control joint {joint_name!r} is not hinge/slide; using first qpos element as draft default.")
        qpos_addr = int(model.jnt_qposadr[joint_id])
        default_angles[joint_name] = float(qpos[qpos_addr]) if qpos_addr < len(qpos) else 0.0
    pos, rot, pose_warnings = _freejoint_pose_from_qpos(model, qpos)
    warnings.extend(pose_warnings)
    init_state = {"pos": pos, "rot": rot, "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]}
    return default_angles, init_state, warnings


def infer_semantics_from_names(body_names: list[str], joint_names: list[str]) -> dict[str, Any]:
    feet = [name for name in body_names if _contains_any(name, ("foot", "toe", "ankle_roll", "ankle"))]
    hands = [name for name in body_names if _contains_any(name, ("hand", "wrist", "gripper", "palm"))]
    torso_candidates = [name for name in body_names if _contains_any(name, ("torso", "trunk", "chest"))]
    if not torso_candidates:
        torso_candidates = [name for name in body_names if "waist" in name.lower()]
    torso_name = torso_candidates[0] if torso_candidates else (body_names[0] if body_names else "")
    undesired = [
        name
        for name in body_names
        if _contains_any(name, ("pelvis", "hip", "shoulder", "torso", "trunk", "chest"))
    ]
    left_ankle = [name for name in joint_names if _is_left(name) and "ankle" in name.lower()]
    right_ankle = [name for name in joint_names if _is_right(name) and "ankle" in name.lower()]
    warnings: list[str] = []
    if len(feet) < 2:
        warnings.append("Fewer than two contact/feet bodies were inferred; biped rewards/terminations may need manual configuration.")
    if not hands:
        warnings.append("No hand bodies were inferred from names.")
    return {
        "feet": _ordered_unique(feet),
        "hands": _ordered_unique(hands),
        "contact_bodies": _ordered_unique(feet),
        "undesired_contact_bodies": _ordered_unique(undesired),
        "torso_name": torso_name,
        "left_ankle_dof_names": _ordered_unique(left_ankle),
        "right_ankle_dof_names": _ordered_unique(right_ankle),
        "warnings": warnings,
    }


def infer_dof_groups_from_names(control_joints: list[str]) -> dict[str, list[str] | dict[str, list[int]]]:
    lower_tokens = ("hip", "knee", "ankle", "leg", "toe", "foot")
    upper_tokens = ("waist", "shoulder", "elbow", "wrist", "arm", "hand")
    arm_tokens = ("shoulder", "elbow", "wrist", "arm", "hand")
    groups: dict[str, list[str] | dict[str, list[int]]] = {
        "lower_dof_names": [name for name in control_joints if _contains_any(name, lower_tokens)],
        "upper_dof_names": [name for name in control_joints if _contains_any(name, upper_tokens)],
        "waist_dof_names": [name for name in control_joints if "waist" in name.lower()],
        "arm_dof_names": [name for name in control_joints if _contains_any(name, arm_tokens)],
        "left_arm_dof_names": [name for name in control_joints if _is_left(name) and _contains_any(name, arm_tokens)],
        "right_arm_dof_names": [name for name in control_joints if _is_right(name) and _contains_any(name, arm_tokens)],
        "left_ankle_dof_names": [name for name in control_joints if _is_left(name) and "ankle" in name.lower()],
        "right_ankle_dof_names": [name for name in control_joints if _is_right(name) and "ankle" in name.lower()],
        "knee_dof_names": [name for name in control_joints if "knee" in name.lower()],
        "symmetric_dofs_idx": {},
    }
    if not groups["lower_dof_names"]:
        groups["lower_dof_names"] = list(control_joints)
    return groups


def _pd_gains(control_joints: list[str], pd_template: str) -> tuple[dict[str, float], dict[str, float], list[str]]:
    warnings = [f"PD gains generated from {pd_template!r} draft template; tune stiffness/damping before training."]
    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    for joint in control_joints:
        lower = joint.lower()
        if "waist" in lower or "torso" in lower:
            kp, kd = 120.0, 4.0
        elif "hip" in lower or "knee" in lower:
            kp, kd = 80.0, 3.0
        elif "ankle" in lower:
            kp, kd = 30.0, 1.5
        elif "shoulder" in lower or "elbow" in lower:
            kp, kd = 20.0, 1.0
        elif "wrist" in lower or "hand" in lower:
            kp, kd = 10.0, 0.5
        else:
            kp, kd = 40.0, 2.0
        stiffness[joint] = kp
        damping[joint] = kd
    return stiffness, damping, warnings


def build_robot_training_yaml_draft(
    *,
    model: mujoco.MjModel,
    name: str,
    xml_path: str | Path,
    hydra_robot: str,
    semantics: dict[str, Any] | None = None,
    default_keyframe: str = "stand",
    actuator_source: str = "yaml",
    action_scale: float = 0.25,
    action_clip_value: float = 5.0,
    normalize_action_to: float = 5.0,
    effort_limit_default: float = 80.0,
    velocity_limit_default: float = 20.0,
    armature_default: float = 0.01,
    friction_default: float = 0.0,
    pd_template: str = "humanoid",
    review_status: str = "draft",
) -> dict[str, Any]:
    warnings: list[str] = []
    control_joints, control_warnings = infer_control_joints_from_xml(model)
    warnings.extend(control_warnings)
    body_names = infer_body_names_from_xml(model)
    base_body, _free_joint, base_warnings = infer_freejoint_base_body(model)
    warnings.extend(base_warnings)
    semantic_guess = infer_semantics_from_names(body_names, control_joints)
    warnings.extend(semantic_guess.pop("warnings"))
    merged_semantics = dict(semantic_guess)
    if semantics:
        if semantics.get("feet"):
            merged_semantics["feet"] = list(semantics["feet"])
            merged_semantics["contact_bodies"] = list(semantics["feet"])
        if semantics.get("hands"):
            merged_semantics["hands"] = list(semantics["hands"])
        merged_semantics["torso_name"] = str(merged_semantics.get("torso_name") or base_body)
    key_bodies = _ordered_unique([base_body, merged_semantics.get("torso_name", ""), *(merged_semantics.get("feet") or []), *(merged_semantics.get("hands") or [])])
    if semantics and semantics.get("key_bodies"):
        key_bodies = list(semantics["key_bodies"])
    _joint_ranges, range_warnings, _lower_upper = infer_joint_ranges_from_xml(model, control_joints)
    warnings.extend(range_warnings)
    effort_limits, velocity_limits, actuator_joints, actuator_warnings = infer_actuator_limits_from_xml(
        model,
        control_joints,
        effort_limit_default=effort_limit_default,
        velocity_limit_default=velocity_limit_default,
        armature_default=armature_default,
        friction_default=friction_default,
    )
    warnings.extend(actuator_warnings)
    default_angles, init_state, pose_warnings = infer_default_pose_from_xml(
        model,
        control_joints,
        keyframe_names=(default_keyframe, "stand", "default", "home", "init"),
    )
    warnings.extend(pose_warnings)
    stiffness, damping, pd_warnings = _pd_gains(control_joints, pd_template)
    warnings.extend(pd_warnings)
    if actuator_source != "yaml":
        warnings.append(f"actuator.source={actuator_source!r} was requested; non-yaml sources are not recommended for new robots.")
    if len(merged_semantics.get("contact_bodies") or []) < 2:
        warnings.append("contact_bodies has fewer than two entries; smoke may run but biped auxiliary rewards can require manual contact body setup.")
    warnings.append("symmetric_dofs_idx was not generated because reliable symmetry inference is robot-specific.")
    warnings.append("This config is XML-derived draft output. Review semantic fields, PD gains, default pose, actuator parameters, contact bodies, and reward/termination-related fields before formal training.")
    return {
        "name": name,
        "xml_path": str(xml_path),
        "base_body": base_body,
        "root_quat_order": "xyzw",
        "coordinate_system": "z_up",
        "dof_unit": "rad",
        "control_joints": {"mode": "explicit", "names": control_joints},
        "feet": list(merged_semantics.get("feet") or []),
        "hands": list(merged_semantics.get("hands") or []),
        "key_bodies": key_bodies,
        "default_dof_pos": {},
        "training": {
            "hydra_robot": hydra_robot,
            "hydra_overrides": [],
            "semantics": {
                "contact_bodies": list(merged_semantics.get("contact_bodies") or []),
                "undesired_contact_bodies": list(merged_semantics.get("undesired_contact_bodies") or []),
                "torso_name": str(merged_semantics.get("torso_name") or base_body),
                "left_ankle_dof_names": list(merged_semantics.get("left_ankle_dof_names") or []),
                "right_ankle_dof_names": list(merged_semantics.get("right_ankle_dof_names") or []),
            },
            "init_state": {
                **init_state,
                "default_joint_angles": default_angles,
            },
            "control": {
                "action_scale": float(action_scale),
                "action_clip_value": float(action_clip_value),
                "normalize_action_to": float(normalize_action_to),
                "effort_limit_scale": 1.0,
                "effort_limit": effort_limits,
                "velocity_limit": velocity_limits,
                "stiffness": stiffness,
                "damping": damping,
            },
            "actuator": {
                "source": actuator_source,
                "joints": actuator_joints,
            },
        },
        "metadata": {
            "generated_from_xml": str(xml_path),
            "generated_by": "humanoidverse.tools.robot_inspect",
            "review_status": review_status,
            "warnings": _ordered_unique(warnings),
        },
    }


def _first_or_empty(values: list[str]) -> str:
    return values[0] if values else ""


def _suffix_pattern(values: list[str], suffix: str) -> str:
    for value in values:
        if suffix in value:
            return suffix
    return _first_or_empty(values)


def build_hydra_robot_config_draft(
    *,
    model: mujoco.MjModel,
    robot_config: dict[str, Any],
    pd_template: str = "humanoid",
    review_status: str = "draft",
) -> dict[str, Any]:
    control_joints = list(robot_config["control_joints"]["names"])
    body_names = infer_body_names_from_xml(model)
    training = dict(robot_config["training"])
    semantics = dict(training["semantics"])
    groups = infer_dof_groups_from_names(control_joints)
    _joint_ranges, _range_warnings, lower_upper = infer_joint_ranges_from_xml(model, control_joints)
    lower, upper = lower_upper
    stiffness = dict(training["control"]["stiffness"])
    damping = dict(training["control"]["damping"])
    integral = {joint: 0.0 for joint in control_joints}
    lower_count = len(groups["lower_dof_names"])
    upper_count = len(groups["upper_dof_names"])
    contact_bodies = list(semantics.get("contact_bodies") or [])
    left_contact = next((name for name in contact_bodies if _is_left(name)), _first_or_empty(contact_bodies))
    right_contact = next((name for name in contact_bodies if _is_right(name)), contact_bodies[1] if len(contact_bodies) > 1 else _first_or_empty(contact_bodies))
    knee_names = list(groups["knee_dof_names"])
    randomize = [
        name for name in body_names if _contains_any(name, ("pelvis", "hip", "thigh", "torso", "trunk", "waist", "knee"))
    ][:12]
    xml_path = Path(str(robot_config["xml_path"])).expanduser()
    return {
        "defaults": ["robot_base"],
        "robot": {
            "num_bodies": len(body_names),
            "dof_obs_size": len(control_joints),
            "actions_dim": len(control_joints),
            "lower_body_actions_dim": lower_count,
            "upper_body_actions_dim": upper_count,
            "body_keys": ["lower_body", "upper_body"],
            "key_bodies": list(robot_config.get("key_bodies") or []),
            "contact_bodies": contact_bodies,
            "num_feet": len(contact_bodies),
            "right_foot_name": right_contact,
            "left_foot_name": left_contact,
            "foot_name": _suffix_pattern(contact_bodies, "foot") or _suffix_pattern(contact_bodies, "ankle"),
            "knee_name": _suffix_pattern(knee_names, "knee"),
            "has_torso": bool(semantics.get("torso_name")),
            "torso_name": str(semantics.get("torso_name") or robot_config.get("base_body")),
            "dof_names": control_joints,
            "upper_dof_names": list(groups["upper_dof_names"]),
            "lower_dof_names": list(groups["lower_dof_names"]),
            "waist_dof_names": list(groups["waist_dof_names"]),
            "arm_dof_names": list(groups["arm_dof_names"]),
            "left_arm_dof_names": list(groups["left_arm_dof_names"]),
            "right_arm_dof_names": list(groups["right_arm_dof_names"]),
            "left_ankle_dof_names": list(groups["left_ankle_dof_names"]),
            "right_ankle_dof_names": list(groups["right_ankle_dof_names"]),
            "knee_dof_names": knee_names,
            "dof_pos_lower_limit_list": lower,
            "dof_pos_upper_limit_list": upper,
            "dof_vel_limit_list": list(training["control"]["velocity_limit"]),
            "dof_effort_limit_list": list(training["control"]["effort_limit"]),
            "dof_effort_limit_scale": 1.0,
            "body_names": body_names,
            "terminate_after_contacts_on": list(semantics.get("undesired_contact_bodies") or []),
            "penalize_contacts_on": list(semantics.get("undesired_contact_bodies") or []),
            "init_state": dict(training["init_state"]),
            "randomize_link_body_names": randomize,
            "control": {
                "control_type": "P",
                "stiffness": stiffness,
                "damping": damping,
                "integral": integral,
                "action_scale": float(training["control"]["action_scale"]),
                "action_clip_value": float(training["control"]["action_clip_value"]),
                "action_rescale": True,
                "clip_torques": True,
                "normalize_action": True,
                "normalize_action_from": 1.0,
                "normalize_action_to": float(training["control"]["normalize_action_to"]),
            },
            "asset": {
                "collapse_fixed_joints": True,
                "replace_cylinder_with_capsule": True,
                "flip_visual_attachments": False,
                "armature": 0.001,
                "thickness": 0.01,
                "max_angular_velocity": 1000.0,
                "max_linear_velocity": 1000.0,
                "density": 0.001,
                "angular_damping": 0.0,
                "linear_damping": 0.0,
                "urdf_file": None,
                "usd_file": None,
                "xml_file": str(robot_config["xml_path"]),
                "robot_type": str(robot_config["name"]),
                "self_collisions": 0,
                "default_dof_drive_mode": 3,
                "fix_base_link": False,
            },
            "motion": {
                "asset": {
                    "assetRoot": str(xml_path.parent),
                    "assetFileName": xml_path.name,
                    "urdfFileName": None,
                },
                "humanoid_type": str(robot_config["name"]),
                "bias_offset": False,
                "has_self_collision": True,
                "has_mesh": False,
                "has_jt_limit": False,
                "has_dof_subset": True,
                "has_upright_start": True,
                "has_smpl_pd_offset": False,
                "remove_toe": False,
                "motion_sym_loss": False,
                "sym_loss_coef": 1,
                "big_ankle": False,
                "has_shape_obs": False,
                "has_shape_obs_disc": False,
                "has_shape_variation": False,
                "uniform_sample": False,
                "motion_fps_scale": 1.0,
                "standardize_motion_length": True,
                "standardize_motion_length_value": 10,
                "reverse_motion": False,
            },
            "metadata": {
                "generated_by": "humanoidverse.tools.robot_inspect",
                "review_status": review_status,
                "warnings": [
                    "AUTO-GENERATED DRAFT. REVIEW BEFORE TRAINING.",
                    f"PD gains generated from {pd_template!r} draft template.",
                    "symmetric_dofs_idx was not generated because reliable symmetry inference is robot-specific.",
                ],
            },
        },
    }
