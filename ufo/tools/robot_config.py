"""Robot XML/YAML loading and validation for training, inference and data tools."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import mujoco
from loguru import logger
from omegaconf import OmegaConf


@dataclass(frozen=True)
class RobotSpec:
    name: str
    xml_path: str
    base_body: str
    root_joint: str | None
    free_joint: str | None
    joint_names: list[str]
    control_joint_names: list[str]
    body_names: list[str]
    actuator_names: list[str]
    actuator_joint_names: list[str]
    joint_types: dict[str, str]
    joint_axes: dict[str, list[float]]
    joint_ranges: dict[str, list[float] | None]
    joint_qpos_addr: dict[str, int]
    joint_dof_addr: dict[str, int]
    joint_body_names: dict[str, str]
    body_parent: dict[str, str | None]
    key_bodies: list[str]
    feet: list[str]
    hands: list[str]
    default_dof_pos: dict[str, float]
    root_quat_order: str
    coordinate_system: str
    dof_unit: str
    nq: int
    nv: int
    nu: int

    @property
    def xml_file(self) -> Path:
        return Path(self.xml_path)


def ensure_known_names(kind: str, names: Iterable[str], known_names: set[str]) -> list[str]:
    values = [str(name) for name in names]
    missing = [name for name in values if name not in known_names]
    if missing:
        raise ValueError(f"Unknown {kind} names in robot config: {missing}. Known names include: {sorted(known_names)[:12]}")
    return values


def ensure_unique(kind: str, names: Iterable[str]) -> list[str]:
    values = [str(name) for name in names]
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in values:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"Duplicate {kind} names in robot config: {duplicates}")
    return values


_JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


def _mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return str(name) if name else f"{obj_type.name.lower()}_{obj_id}"


def resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    raw = Path(str(raw_path)).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [base_dir / raw, Path.cwd() / raw]
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
    return robot_spec_from_config(config, path)


def robot_spec_from_config(config: dict[str, Any], path: Path) -> RobotSpec:
    """Validate a resolved robot mapping and inspect its MuJoCo XML."""
    for field in ("name", "xml_path", "base_body"):
        if field not in config:
            raise ValueError(f"Robot config {path} is missing required field '{field}'")

    xml_path = resolve_path(str(config["xml_path"]), base_dir=path.parent)
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


def _required_mapping(config: dict[str, Any], key: str, *, context: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context} requires mapping field '{key}'")
    return dict(value)


def _required_float(config: dict[str, Any], key: str, *, context: str) -> float:
    if key not in config:
        raise ValueError(f"{context} requires field '{key}'")
    try:
        return float(config[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{key} must be numeric, got {config[key]!r}") from exc


def _required_float_list(config: dict[str, Any], key: str, *, length: int, context: str) -> list[float]:
    value = config.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context}.{key} must be a list with length {length}")
    return [float(item) for item in value]


def _required_name_list(config: dict[str, Any], key: str, *, context: str) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{context}.{key} must be a list")
    return [str(item) for item in value]


@dataclass(frozen=True)
class RobotTrainingSpec:
    config_path: Path
    robot: RobotSpec
    action_scale: float
    action_clip_value: float
    normalize_action_to: float
    init_state: dict[str, list[float]]
    default_joint_angles: dict[str, float]
    stiffness: dict[str, float]
    damping: dict[str, float]
    effort_limits: list[float]
    velocity_limits: list[float]
    effort_limit_scale: float
    actuator: dict[str, Any]
    contact_bodies: list[str]
    undesired_contact_bodies: list[str]
    torso_name: str
    left_ankle_dof_names: list[str]
    right_ankle_dof_names: list[str]

    action_rescale: bool = True
    task_overrides: dict[str, Any] = dataclass_field(default_factory=dict)
    motion_extensions: list[dict[str, Any]] = dataclass_field(default_factory=list)
    fb_agent_overrides: dict[str, Any] = dataclass_field(default_factory=dict)
    train_overrides: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_env_dict(self) -> dict[str, Any]:
        robot = self.robot
        return {
            "config_path": str(self.config_path),
            "motion_extensions": self.motion_extensions,
            "robot": {
                "name": robot.name,
                "xml_path": robot.xml_path,
                "base_body": robot.base_body,
                "control_joint_names": list(robot.control_joint_names),
                "body_names": list(robot.body_names),
                "key_bodies": list(robot.key_bodies),
                "feet": list(robot.feet),
                "hands": list(robot.hands),
                "joint_ranges": dict(robot.joint_ranges),
            },
            "action_scale": self.action_scale,
            "action_rescale": self.action_rescale,
            "action_clip_value": self.action_clip_value,
            "normalize_action_to": self.normalize_action_to,
            "init_state": self.init_state,
            "default_joint_angles": dict(self.default_joint_angles),
            "stiffness": dict(self.stiffness),
            "damping": dict(self.damping),
            "effort_limits": list(self.effort_limits),
            "velocity_limits": list(self.velocity_limits),
            "effort_limit_scale": self.effort_limit_scale,
            "actuator": dict(self.actuator),
            "contact_bodies": list(self.contact_bodies),
            "undesired_contact_bodies": list(self.undesired_contact_bodies),
            "torso_name": self.torso_name,
            "left_ankle_dof_names": list(self.left_ankle_dof_names),
            "right_ankle_dof_names": list(self.right_ankle_dof_names),
        }


def resolve_robot_config_path(raw_path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    base = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()
    path = resolve_path(raw_path, base_dir=base)
    if not path.exists():
        raise FileNotFoundError(f"Robot config does not exist: {path}")
    return path


def load_robot_training_spec(config_path: str | Path) -> RobotTrainingSpec:
    path = resolve_robot_config_path(config_path)
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Robot config must be a mapping: {path}")

    metadata = config.get("metadata")
    if isinstance(metadata, dict) and metadata.get("review_status") == "draft":
        print(
            "WARNING: Robot config is auto-generated draft. Review semantics, default pose, PD gains, "
            "actuator parameters, contact bodies, and reward/termination-related fields before formal training.",
            flush=True,
        )
    robot = robot_spec_from_config(config, path)
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError(
            f"Robot config {path} is missing required 'training' section for UFO training. "
            "Run robot_inspect to draft the robot config, then curate training control, init_state, and actuator fields."
        )
    training = dict(training)
    if training.get("hydra_robot") or training.get("hydra_overrides"):
        raise ValueError("Migrate training.hydra_robot/hydra_overrides to native task_overrides before training")
    ranges = _required_mapping(training, "joint_ranges", context="training") if "joint_ranges" in training else {}
    for joint, limits in ranges.items():
        if joint not in robot.control_joint_names:
            raise ValueError(f"training.joint_ranges contains unknown control joint: {joint}")
        if not isinstance(limits, list) or len(limits) != 2 or not all(math.isfinite(float(v)) for v in limits):
            raise ValueError(f"training.joint_ranges.{joint} must contain two finite limits")
        limits = [float(v) for v in limits]
        ranges[joint] = limits
        if limits[0] > limits[1]:
            raise ValueError(f"training.joint_ranges.{joint} lower limit exceeds upper limit")
    robot = replace(robot, joint_ranges={**robot.joint_ranges, **ranges})
    control = _required_mapping(training, "control", context="training")
    init_state = _required_mapping(training, "init_state", context="training")
    actuator = _required_mapping(training, "actuator", context="training")

    control_joints = list(robot.control_joint_names)
    joint_count = len(control_joints)
    default_joint_angles_raw = _required_mapping(init_state, "default_joint_angles", context="training.init_state")
    missing_default = [joint for joint in control_joints if joint not in default_joint_angles_raw]
    if missing_default:
        raise ValueError(f"training.init_state.default_joint_angles is missing joints: {missing_default}")

    stiffness = {str(k): float(v) for k, v in _required_mapping(control, "stiffness", context="training.control").items()}
    damping = {str(k): float(v) for k, v in _required_mapping(control, "damping", context="training.control").items()}

    effort_limits = _required_float_list(control, "effort_limit", length=joint_count, context="training.control")
    velocity_limits = _required_float_list(control, "velocity_limit", length=joint_count, context="training.control")

    actuator_source = str(actuator.get("source", ""))
    if actuator_source not in {"g1_mode15", "g1-mode_15", "yaml"}:
        raise ValueError("training.actuator.source must be one of: g1_mode15, g1-mode_15, yaml")
    actuator_model = actuator.get("model", "dc_motor")
    if actuator_model not in {"dc_motor", "roboparty"}:
        raise ValueError("training.actuator.model must be one of: dc_motor, roboparty")
    if actuator_model == "roboparty" and actuator_source != "yaml":
        raise ValueError("training.actuator.model=roboparty requires source=yaml")
    action_rescale = control.get("action_rescale", True)
    if not isinstance(action_rescale, bool):
        raise ValueError("training.control.action_rescale must be a boolean")
    if actuator_source == "yaml":
        joints = actuator.get("joints")
        if not isinstance(joints, dict):
            raise ValueError("training.actuator.source=yaml requires training.actuator.joints")
        for joint in control_joints:
            params = joints.get(joint)
            if not isinstance(params, dict):
                raise ValueError(f"training.actuator.joints is missing parameters for joint {joint!r}")
            for key in ("effort_limit", "velocity_limit", "armature", "friction"):
                if key not in params:
                    raise ValueError(f"training.actuator.joints.{joint} is missing '{key}'")
                float(params[key])
            if actuator_model == "roboparty":
                for key in ("effort_limit", "velocity_limit", "keep_velocity", "rated_effort", "rated_velocity"):
                    value = _required_float(params, key, context=f"training.actuator.joints.{joint}")
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError(f"training.actuator.joints.{joint}.{key} must be finite and positive")
                for key in ("armature", "friction", "viscous_friction"):
                    value = _required_float(params, key, context=f"training.actuator.joints.{joint}")
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(f"training.actuator.joints.{joint}.{key} must be finite and non-negative")
                if float(params["keep_velocity"]) >= float(params["velocity_limit"]):
                    raise ValueError(f"training.actuator.joints.{joint}.keep_velocity must be below velocity_limit")
                if float(params["rated_effort"]) > float(params["effort_limit"]):
                    raise ValueError(f"training.actuator.joints.{joint}.rated_effort exceeds effort_limit")
                if float(params["rated_velocity"]) > float(params["velocity_limit"]):
                    raise ValueError(f"training.actuator.joints.{joint}.rated_velocity exceeds velocity_limit")
        if actuator_model == "roboparty":
            for key in ("min_delay", "max_delay"):
                value = actuator.get(key)
                if type(value) is not int or value < 0:
                    raise ValueError(f"training.actuator.{key} must be a non-negative integer")
            if actuator["min_delay"] > actuator["max_delay"]:
                raise ValueError("training.actuator.min_delay must not exceed max_delay")

    semantic = dict(training.get("semantics") or {})
    contact_bodies = _required_name_list(semantic, "contact_bodies", context="training.semantics")
    undesired_contact_bodies = [str(item) for item in semantic.get("undesired_contact_bodies", [])]
    torso_name = str(semantic.get("torso_name") or robot.base_body)
    left_ankle_dof_names = [str(item) for item in semantic.get("left_ankle_dof_names", [])]
    right_ankle_dof_names = [str(item) for item in semantic.get("right_ankle_dof_names", [])]

    return RobotTrainingSpec(
        config_path=path,
        robot=robot,
        task_overrides=dict(training.get("task_overrides", {})),
        motion_extensions=list(training.get("motion_extensions", [])),
        fb_agent_overrides=_required_mapping(training, "fb_agent_overrides", context="training") if "fb_agent_overrides" in training else {},
        train_overrides=_required_mapping(training, "train_overrides", context="training") if "train_overrides" in training else {},
        action_scale=_required_float(control, "action_scale", context="training.control"),
        action_rescale=action_rescale,
        action_clip_value=_required_float(control, "action_clip_value", context="training.control"),
        normalize_action_to=_required_float(control, "normalize_action_to", context="training.control"),
        init_state={
            "pos": _required_float_list(init_state, "pos", length=3, context="training.init_state"),
            "rot": _required_float_list(init_state, "rot", length=4, context="training.init_state"),
            "lin_vel": _required_float_list(init_state, "lin_vel", length=3, context="training.init_state"),
            "ang_vel": _required_float_list(init_state, "ang_vel", length=3, context="training.init_state"),
        },
        default_joint_angles={joint: float(default_joint_angles_raw[joint]) for joint in control_joints},
        stiffness=stiffness,
        damping=damping,
        effort_limits=effort_limits,
        velocity_limits=velocity_limits,
        effort_limit_scale=float(control.get("effort_limit_scale", 1.0)),
        actuator=actuator,
        contact_bodies=contact_bodies,
        undesired_contact_bodies=undesired_contact_bodies,
        torso_name=torso_name,
        left_ankle_dof_names=left_ankle_dof_names,
        right_ankle_dof_names=right_ankle_dof_names,
    )


def assert_robot_configs_compatible(cli_config: str | Path, manifest_config: str | Path) -> Path:
    cli_path = resolve_robot_config_path(cli_config)
    manifest_path = resolve_robot_config_path(manifest_config)
    if cli_path == manifest_path:
        return cli_path
    cli_spec = load_robot_spec(cli_path)
    manifest_spec = load_robot_spec(manifest_path)
    if cli_spec.name == manifest_spec.name and Path(cli_spec.xml_path).resolve() == Path(manifest_spec.xml_path).resolve():
        return cli_path
    raise ValueError(
        "--robot-config does not match data manifest robot_config: "
        f"cli={cli_path} ({cli_spec.name}, {cli_spec.xml_path}), "
        f"manifest={manifest_path} ({manifest_spec.name}, {manifest_spec.xml_path})"
    )


BUILTIN_ROBOTS = {
    "g1": ("configs/robots/g1_29dof.yaml", "ufo/data/lafan_29dof_10s-clipped.pkl"),
    "rp1": ("configs/robots/rp1_3.yaml", "ufo/data/rp1_24dof_lafan_v7_10s-clipped.pkl"),
    "rpo": ("configs/robots/rpo.yaml", "ufo/data/rpo_23dof_lafan_grounded_10s-clipped.pkl"),
}


def builtin_robot_config(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / BUILTIN_ROBOTS[name][0]


def select_robot_config(cli_robot_config: str | Path | None, manifest_robot_config: str | Path | None = None) -> Path:
    if cli_robot_config is not None and manifest_robot_config is not None:
        return assert_robot_configs_compatible(cli_robot_config, manifest_robot_config)
    selected = cli_robot_config if cli_robot_config is not None else manifest_robot_config
    return builtin_robot_config("g1") if selected is None else resolve_robot_config_path(selected)


def default_robot_data(config: str | Path) -> str:
    for name, (_, data) in BUILTIN_ROBOTS.items():
        if Path(config).resolve() == builtin_robot_config(name):
            return str(Path(__file__).resolve().parents[2] / data)
    raise ValueError("Custom robots require --data-path or --data-manifest with matching retargeted motions.")
