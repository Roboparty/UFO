"""MuJoCo XML/YAML parser for RobotSpec."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
from loguru import logger
from omegaconf import OmegaConf

from humanoidverse.utils.robot_spec.robot_spec import RobotSpec
from humanoidverse.utils.robot_spec.validate import ensure_known_names, ensure_unique


_JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


def _mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return str(name) if name else f"{obj_type.name.lower()}_{obj_id}"


def _resolve_xml_path(xml_path: str, config_path: Path) -> Path:
    raw = Path(str(xml_path)).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [config_path.parent / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def _list_from_config(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Robot config field '{field_name}' must be a list")
    return [str(item) for item in value]


def _control_joint_names(config: dict[str, Any], joint_names: list[str], actuator_joint_names: list[str]) -> list[str]:
    control_config = config.get("control_joints", {"mode": "all_actuated"})
    if isinstance(control_config, list):
        return ensure_unique("control_joint", control_config)
    if not isinstance(control_config, dict):
        raise ValueError("Robot config field 'control_joints' must be a mapping or list")

    mode = str(control_config.get("mode", "all_actuated"))
    if mode == "all_actuated":
        if not actuator_joint_names:
            raise ValueError("control_joints.mode=all_actuated but the MuJoCo XML has no joint actuators")
        return ensure_unique("control_joint", actuator_joint_names)
    if mode == "explicit":
        joints = control_config.get("names")
        if not isinstance(joints, list) or not joints:
            raise ValueError("control_joints.mode=explicit requires a non-empty 'names' list")
        return ensure_unique("control_joint", joints)
    raise ValueError(f"Unsupported control_joints.mode={mode!r}. Supported modes: all_actuated, explicit")


def load_robot_spec(config_path: str | Path) -> RobotSpec:
    """Load and validate a robot YAML backed by a MuJoCo XML file."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Robot config does not exist: {path}")
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Robot config must be a mapping: {path}")
    for field in ("name", "xml_path", "base_body"):
        if field not in config:
            raise ValueError(f"Robot config {path} is missing required field '{field}'")

    xml_path = _resolve_xml_path(str(config["xml_path"]), path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Robot XML does not exist: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    all_body_names = [_mj_name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(model.nbody)]
    # MotionLib pose_aa excludes MuJoCo's world body. Existing G1 data has
    # nbody-1 axis-angle entries with the floating base body at index 0.
    body_names = all_body_names[1:]
    joint_names = [_mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(model.njnt)]
    actuator_names = [_mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in range(model.nu)]

    joint_types: dict[str, str] = {}
    joint_axes: dict[str, list[float]] = {}
    joint_ranges: dict[str, list[float] | None] = {}
    joint_qpos_addr: dict[str, int] = {}
    joint_dof_addr: dict[str, int] = {}
    joint_body_names: dict[str, str] = {}
    free_joints: list[str] = []

    for idx, joint_name in enumerate(joint_names):
        joint_type = _JOINT_TYPE_NAMES.get(int(model.jnt_type[idx]), str(int(model.jnt_type[idx])))
        joint_types[joint_name] = joint_type
        joint_axes[joint_name] = [float(v) for v in model.jnt_axis[idx].tolist()]
        joint_ranges[joint_name] = [float(v) for v in model.jnt_range[idx].tolist()] if int(model.jnt_limited[idx]) else None
        joint_qpos_addr[joint_name] = int(model.jnt_qposadr[idx])
        joint_dof_addr[joint_name] = int(model.jnt_dofadr[idx])
        joint_body_names[joint_name] = _mj_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[idx]))
        if joint_type == "free":
            free_joints.append(joint_name)

    actuator_joint_names: list[str] = []
    for idx in range(model.nu):
        if int(model.actuator_trntype[idx]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        joint_id = int(model.actuator_trnid[idx, 0])
        if joint_id >= 0:
            actuator_joint_names.append(_mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))

    body_parent: dict[str, str | None] = {}
    for idx in range(1, model.nbody):
        body_name = all_body_names[idx]
        parent_id = int(model.body_parentid[idx])
        body_parent[body_name] = None if parent_id == 0 else _mj_name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

    known_bodies = set(body_names)
    known_joints = set(joint_names)
    base_body = str(config["base_body"])
    ensure_known_names("body", [base_body], known_bodies)

    control_joint_names = _control_joint_names(config, joint_names, actuator_joint_names)
    ensure_known_names("joint", control_joint_names, known_joints)

    key_bodies = ensure_known_names("body", _list_from_config(config.get("key_bodies"), field_name="key_bodies"), known_bodies)
    feet = ensure_known_names("body", _list_from_config(config.get("feet"), field_name="feet"), known_bodies)
    hands = ensure_known_names("body", _list_from_config(config.get("hands"), field_name="hands"), known_bodies)

    default_dof_pos_raw = config.get("default_dof_pos") or {}
    if not isinstance(default_dof_pos_raw, dict):
        raise ValueError("Robot config field 'default_dof_pos' must be a mapping")
    ensure_known_names("default_dof_pos joint", default_dof_pos_raw.keys(), known_joints)
    default_dof_pos = {joint: float(default_dof_pos_raw.get(joint, 0.0)) for joint in control_joint_names}

    root_joint_value = config.get("root_joint")
    root_joint = str(root_joint_value) if root_joint_value else (free_joints[0] if free_joints else None)
    if root_joint is not None:
        ensure_known_names("joint", [root_joint], known_joints)
    free_joint = root_joint if root_joint is not None and joint_types[root_joint] == "free" else (free_joints[0] if free_joints else None)

    root_quat_order = str(config.get("root_quat_order", "xyzw"))
    if root_quat_order not in {"xyzw", "wxyz"}:
        raise ValueError("root_quat_order must be either 'xyzw' or 'wxyz'")

    spec = RobotSpec(
        name=str(config["name"]),
        xml_path=str(xml_path),
        base_body=base_body,
        root_joint=root_joint,
        free_joint=free_joint,
        joint_names=joint_names,
        control_joint_names=control_joint_names,
        body_names=body_names,
        actuator_names=actuator_names,
        actuator_joint_names=actuator_joint_names,
        joint_types=joint_types,
        joint_axes=joint_axes,
        joint_ranges=joint_ranges,
        joint_qpos_addr=joint_qpos_addr,
        joint_dof_addr=joint_dof_addr,
        joint_body_names=joint_body_names,
        body_parent=body_parent,
        key_bodies=key_bodies,
        feet=feet,
        hands=hands,
        default_dof_pos=default_dof_pos,
        root_quat_order=root_quat_order,
        coordinate_system=str(config.get("coordinate_system", "z_up")),
        dof_unit=str(config.get("dof_unit", "rad")),
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
    )
    logger.info(
        "[robot-spec] name={name} xml={xml} nq={nq} nv={nv} nu={nu} bodies={bodies} joints={joints} control_joints={control}".format(
            name=spec.name,
            xml=spec.xml_path,
            nq=spec.nq,
            nv=spec.nv,
            nu=spec.nu,
            bodies=len(spec.body_names),
            joints=len(spec.joint_names),
            control=len(spec.control_joint_names),
        )
    )
    return spec
