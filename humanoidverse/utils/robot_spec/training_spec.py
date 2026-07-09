"""Robot training metadata loaded from robot YAML configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from humanoidverse.utils.robot_spec.mujoco_parser import load_robot_spec
from humanoidverse.utils.robot_spec.robot_spec import RobotSpec


def _resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    for candidate in (base_dir / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


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
    hydra_robot: str
    hydra_overrides: list[str]
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

    def to_env_dict(self) -> dict[str, Any]:
        robot = self.robot
        return {
            "config_path": str(self.config_path),
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
    path = _resolve_path(raw_path, base_dir=base)
    if not path.exists():
        raise FileNotFoundError(f"Robot config does not exist: {path}")
    return path


def load_robot_training_spec(config_path: str | Path) -> RobotTrainingSpec:
    path = resolve_robot_config_path(config_path)
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Robot config must be a mapping: {path}")

    robot = load_robot_spec(path)
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError(
            f"Robot config {path} is missing required 'training' section for UFO training. "
            "Run robot_inspect to draft the robot config, then curate training control, init_state, and actuator fields."
        )
    training = dict(training)
    if "hydra_robot" not in training:
        raise ValueError("training requires field 'hydra_robot'")
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

    semantic = dict(training.get("semantics") or {})
    contact_bodies = _required_name_list(semantic, "contact_bodies", context="training.semantics")
    undesired_contact_bodies = [str(item) for item in semantic.get("undesired_contact_bodies", [])]
    torso_name = str(semantic.get("torso_name") or robot.base_body)
    left_ankle_dof_names = [str(item) for item in semantic.get("left_ankle_dof_names", [])]
    right_ankle_dof_names = [str(item) for item in semantic.get("right_ankle_dof_names", [])]

    return RobotTrainingSpec(
        config_path=path,
        robot=robot,
        hydra_robot=str(training["hydra_robot"]),
        hydra_overrides=[str(item) for item in training.get("hydra_overrides", [])],
        action_scale=_required_float(control, "action_scale", context="training.control"),
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
