"""MJLab/MuJoCo-Warp bridge for UFO.

This module intentionally mirrors the public surface of
the old HumanoidVerse vector-env adapter so the existing FBcprAux training
loop can be reused without replacing the algorithm with MJLab/RSL-RL PPO.
MJLab owns batched physics stepping; this wrapper reconstructs the observation,
reward, reset and info dictionaries expected by the original UFO code.
"""

import os
import random
import typing as tp
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import gymnasium
import hydra
import humanoidverse
import numpy as np
import pydantic
import torch
from gymnasium import Env
from gymnasium.vector import VectorEnv
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

from humanoidverse.agents.base import BaseConfig
from humanoidverse.envs.env_utils.history_handler import HistoryHandler as HVHistoryHandler
from humanoidverse.envs.motion_observations import compute_humanoid_observations_max
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.torch_utils import (
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate_inverse,
    wxyz_to_xyzw,
    wrap_to_pi,
    xyzw_to_wxyz,
)

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = os.path.dirname(humanoidverse.__file__)
else:
    HUMANOIDVERSE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HYDRA_CONFIG_DIR = os.path.join(HUMANOIDVERSE_DIR, "config")
HYDRA_CONFIG_REL_PATH = os.path.join("exp", "bfm_zero", "bfm_zero")
G1_MJLAB_MJCF_PATH = "humanoidverse/data/robots/g1_mjlab/g1_29dof.xml"
G1_MJLAB_ACTUATOR_SOURCE = "g1-mode_15"


def _resolve_humanoidverse_path(path_value: str | os.PathLike[str]) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    text = str(path_value)
    if text == "humanoidverse" or text.startswith("humanoidverse/"):
        return str((Path(HUMANOIDVERSE_DIR).parent / path).resolve())
    return text


def _reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    """Compute reflected motor inertia constants for Unitree G1 motors."""
    return rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2 + rotor_inertia[1] * gear_ratio[2] ** 2 + rotor_inertia[2]


_ARMATURE_5020 = _reflected_inertia_from_two_stage_planetary((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + (46 / 18), 1 + (56 / 16)))
_ARMATURE_7520_14 = _reflected_inertia_from_two_stage_planetary((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + (48 / 22)))
_ARMATURE_7520_22 = _reflected_inertia_from_two_stage_planetary((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
_ARMATURE_5010 = _reflected_inertia_from_two_stage_planetary((0.084e-4, 0.015e-4, 0.068e-4), (1, 4, 4))


def _g1_mjlab_mode15_actuator_params(dof_names: tp.Sequence[str]) -> dict[str, list[float]]:
    """Return per-DOF G1 mode-15 motor params in UFO order.

    The constants are vendored here so training does not depend on an external
    asset package or download path. Kp/Kd remain UFO values; this only
    supplies motor effort, velocity reference, armature and dry friction.
    """

    efforts: list[float] = []
    velocities: list[float] = []
    armatures: list[float] = []
    frictions: list[float] = []

    for joint_name in dof_names:
        if "_hip_pitch_joint" in joint_name:
            effort, velocity, armature = 139.0, 20.0, _ARMATURE_7520_22
        elif "_hip_yaw_joint" in joint_name or joint_name == "waist_yaw_joint":
            effort, velocity, armature = 88.0, 32.0, _ARMATURE_7520_14
        elif "_hip_roll_joint" in joint_name or "_knee_joint" in joint_name:
            effort, velocity, armature = 139.0, 20.0, _ARMATURE_7520_22
        elif "_ankle_pitch_joint" in joint_name or "_ankle_roll_joint" in joint_name:
            effort, velocity, armature = 50.0, 37.0, 2.0 * _ARMATURE_5020
        elif joint_name in ("waist_pitch_joint", "waist_roll_joint"):
            effort, velocity, armature = 50.0, 37.0, 2.0 * _ARMATURE_5020
        elif (
            "_shoulder_pitch_joint" in joint_name
            or "_shoulder_roll_joint" in joint_name
            or "_shoulder_yaw_joint" in joint_name
            or "_elbow_joint" in joint_name
            or "_wrist_roll_joint" in joint_name
        ):
            effort, velocity, armature = 25.0, 37.0, _ARMATURE_5020
        elif "_wrist_pitch_joint" in joint_name or "_wrist_yaw_joint" in joint_name:
            effort, velocity, armature = 13.4, 27.0, _ARMATURE_5010
        else:
            raise ValueError(f"No G1 mode-15 actuator parameters for joint: {joint_name}")

        efforts.append(effort)
        velocities.append(velocity)
        armatures.append(armature)
        frictions.append(0.01)

    return {
        "effort_limit": efforts,
        "velocity_limit": velocities,
        "armature": armatures,
        "friction": frictions,
    }


def _obs_joint_pos(env):
    return env.scene["robot"].data.joint_pos


def _zero_reward(env):
    return torch.zeros(env.num_envs, device=env.device)


def _to_list(value) -> list:
    if value is None:
        return []
    return list(OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value)


def _to_float_dict(value) -> dict[str, float]:
    value = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return {str(k): float(v) for k, v in value.items()}


def _match_joint_value(joint_name: str, value_by_substring: dict[str, float], default: float = 0.0) -> float:
    for key, value in value_by_substring.items():
        if key in joint_name:
            return float(value)
    return float(default)


def _joint_limits_from_robot_training(robot_training: dict[str, Any], dof_names: tuple[str, ...]) -> tuple[list[float], list[float]]:
    joint_ranges = dict(robot_training.get("robot", {}).get("joint_ranges") or {})
    lower, upper = [], []
    for joint_name in dof_names:
        value = joint_ranges.get(joint_name)
        if value is None:
            lower.append(-3.14159)
            upper.append(3.14159)
        else:
            lower.append(float(value[0]))
            upper.append(float(value[1]))
    return lower, upper


def _patch_humanoidverse_robot_config(config, robot_training: dict[str, Any] | None) -> None:
    if not robot_training:
        return
    robot_info = dict(robot_training["robot"])
    dof_names = [str(name) for name in robot_info["control_joint_names"]]
    body_names = [str(name) for name in robot_info["body_names"]]
    feet = [str(name) for name in robot_info.get("feet") or []]
    lower, upper = _joint_limits_from_robot_training(robot_training, tuple(dof_names))

    config.robot.dof_names = dof_names
    config.robot.dof_obs_size = len(dof_names)
    config.robot.actions_dim = len(dof_names)
    config.robot.body_names = body_names
    config.robot.num_bodies = len(body_names)
    config.robot.key_bodies = list(robot_info.get("key_bodies") or [])
    config.robot.contact_bodies = list(robot_training.get("contact_bodies") or feet)
    config.robot.num_feet = len(config.robot.contact_bodies)
    config.robot.torso_name = str(robot_training.get("torso_name") or robot_info.get("base_body"))
    config.robot.penalize_contacts_on = list(robot_training.get("undesired_contact_bodies") or [])
    config.robot.terminate_after_contacts_on = list(robot_training.get("undesired_contact_bodies") or [])
    config.robot.left_ankle_dof_names = list(robot_training.get("left_ankle_dof_names") or [])
    config.robot.right_ankle_dof_names = list(robot_training.get("right_ankle_dof_names") or [])
    config.robot.dof_pos_lower_limit_list = lower
    config.robot.dof_pos_upper_limit_list = upper
    config.robot.dof_vel_limit_list = list(robot_training["velocity_limits"])
    config.robot.dof_effort_limit_list = list(robot_training["effort_limits"])
    config.robot.dof_effort_limit_scale = float(robot_training.get("effort_limit_scale", 1.0))

    if config.robot.get("init_state") is None:
        config.robot.init_state = OmegaConf.create({})
    if config.robot.get("control") is None:
        config.robot.control = OmegaConf.create({})

    config.robot.init_state.pos = list(robot_training["init_state"]["pos"])
    config.robot.init_state.rot = list(robot_training["init_state"]["rot"])
    config.robot.init_state.lin_vel = list(robot_training["init_state"]["lin_vel"])
    config.robot.init_state.ang_vel = list(robot_training["init_state"]["ang_vel"])
    config.robot.init_state.default_joint_angles = dict(robot_training["default_joint_angles"])
    config.robot.control.stiffness = dict(robot_training["stiffness"])
    config.robot.control.damping = dict(robot_training["damping"])
    config.robot.control.action_scale = float(robot_training["action_scale"])
    config.robot.control.action_clip_value = float(robot_training["action_clip_value"])
    config.robot.control.normalize_action_to = float(robot_training["normalize_action_to"])

    xml_path = Path(robot_info["xml_path"]).expanduser().resolve()
    if config.robot.get("asset") is None:
        config.robot.asset = OmegaConf.create({})
    config.robot.asset.asset_root = str(xml_path.parent)
    config.robot.asset.assetFileName = xml_path.name
    config.robot.asset.xml_file = str(xml_path)

    if config.robot.get("motion") is None:
        config.robot.motion = OmegaConf.create({})
    if config.robot.motion.get("asset") is None:
        config.robot.motion.asset = OmegaConf.create({})
    config.robot.motion.asset.assetRoot = str(xml_path.parent)
    config.robot.motion.asset.assetFileName = xml_path.name
    config.robot.motion.asset.urdfFileName = None

    extend_config = []
    for item in _to_list(config.robot.motion.get("extend_config", [])):
        parent_name = str(item.get("parent_name", ""))
        if parent_name in body_names:
            extend_config.append(dict(item))
    config.robot.motion.extend_config = extend_config
    config.robot.motion.nums_extend_bodies = len(extend_config)


def _actuator_params_from_training(dof_names: tp.Sequence[str], robot_training: dict[str, Any] | None) -> tuple[str, dict[str, list[float]]]:
    if not robot_training:
        return G1_MJLAB_ACTUATOR_SOURCE, _g1_mjlab_mode15_actuator_params(dof_names)
    actuator = dict(robot_training.get("actuator") or {})
    source = str(actuator.get("source", G1_MJLAB_ACTUATOR_SOURCE))
    if source in {"g1_mode15", "g1-mode_15"}:
        return source, _g1_mjlab_mode15_actuator_params(dof_names)
    if source != "yaml":
        raise ValueError(f"Unsupported training.actuator.source={source!r}")
    joints = actuator.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("training.actuator.source=yaml requires training.actuator.joints")
    params = {"effort_limit": [], "velocity_limit": [], "armature": [], "friction": []}
    for joint_name in dof_names:
        joint_params = joints.get(joint_name)
        if not isinstance(joint_params, dict):
            raise ValueError(f"training.actuator.joints is missing parameters for joint {joint_name!r}")
        for key in params:
            if key not in joint_params:
                raise ValueError(f"training.actuator.joints.{joint_name} is missing '{key}'")
            params[key].append(float(joint_params[key]))
    return source, params


def _default_joint_pos(config) -> torch.Tensor:
    values = [float(config.robot.init_state.default_joint_angles[name]) for name in config.robot.dof_names]
    return torch.tensor(values, dtype=torch.float32)


def _action_target_scale(config) -> torch.Tensor:
    dof_names = tuple(_to_list(config.robot.dof_names))
    stiffness = _to_float_dict(config.robot.control.stiffness)
    # UFO action_rescale uses the configured effort limits.  The Isaac
    # path does not apply dof_effort_limit_scale to those limits, so MJLab must
    # not do it either.
    effort_limits = [float(x) for x in _to_list(config.robot.dof_effort_limit_list)]
    scales = []
    for i, joint_name in enumerate(dof_names):
        kp = _match_joint_value(joint_name, stiffness)
        scale = float(config.robot.control.action_scale)
        if bool(config.robot.control.action_rescale):
            if kp <= 0.0:
                raise ValueError(f"Cannot action_rescale joint {joint_name}: stiffness={kp}")
            scale *= effort_limits[i] / kp
        scales.append(scale)
    return torch.tensor(scales, dtype=torch.float32)


def _small_random_quaternions(n: int, max_angle: float, device: str) -> torch.Tensor:
    axis = torch.randn((n, 3), device=device)
    axis = axis / torch.clamp(torch.norm(axis, dim=1, keepdim=True), min=1.0e-6)
    angles = max_angle * torch.rand((n, 1), device=device)
    sin_half_angle = torch.sin(angles / 2)
    cos_half_angle = torch.cos(angles / 2)
    return torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)


def _compose_humanoidverse_config(
    *,
    num_envs: int,
    relative_config_path: str,
    hydra_overrides: list[str],
    lafan_tail_path: str | list[str],
    data_mix_weights: list[float] | None,
    disable_obs_noise: bool,
    disable_domain_randomization: bool,
    max_episode_length_s: float | None,
    root_height_obs: bool,
    robot_training: dict[str, Any] | None = None,
):
    with hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
        cfg = hydra.compose(config_name=relative_config_path, overrides=hydra_overrides or [])
    unresolved_conf = OmegaConf.to_container(cfg, resolve=False)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))

    cfg.num_envs = num_envs
    cfg.exp_base = "__no_exp_base__"
    cfg.env.config.headless = True
    OmegaConf.set_struct(cfg, False)
    _patch_humanoidverse_robot_config(cfg, robot_training)
    cfg.robot.asset.asset_root = _resolve_humanoidverse_path(cfg.robot.asset.asset_root)
    cfg.robot.motion.asset.assetRoot = _resolve_humanoidverse_path(cfg.robot.motion.asset.assetRoot)
    cfg.robot.motion.motion_file = lafan_tail_path
    if data_mix_weights is not None:
        cfg.robot.motion.motion_file_weights = data_mix_weights

    pre_process_config(cfg)

    if disable_obs_noise:
        for key in cfg.obs.noise_scales.keys():
            cfg.obs.noise_scales[key] = 0.0
    cfg.obs.root_height_obs = root_height_obs

    if disable_domain_randomization:
        cfg.domain_rand.randomize_ctrl_delay = False
        cfg.domain_rand.randomize_pd_gain = False
        cfg.domain_rand.randomize_base_com = False
        cfg.domain_rand.randomize_link_mass = False
        cfg.domain_rand.randomize_friction = False
        cfg.domain_rand.randomize_torque_rfi = False
        cfg.domain_rand.randomize_rfi_lim = False
        cfg.domain_rand.randomize_push_robots = False
        cfg.domain_rand.push_robots = False
        cfg.domain_rand.randomize_default_dof_pos = False

    assert cfg.env.config.termination.terminate_when_close_to_dof_pos_limit is False
    assert cfg.env.config.termination.terminate_when_close_to_dof_vel_limit is False
    assert cfg.env.config.termination.terminate_when_close_to_torque_limit is False
    assert cfg.env.config.termination.terminate_by_contact is False
    assert cfg.env.config.termination.terminate_by_gravity is False
    assert cfg.env.config.termination.terminate_by_low_height is False
    assert cfg.env.config.termination.terminate_when_motion_end is False
    assert cfg.env.config.termination.terminate_when_motion_far is False
    assert cfg.env.config.robot.control.normalize_action_to == cfg.env.config.robot.control.action_clip_value

    if max_episode_length_s is not None:
        cfg.env.config.max_episode_length_s = max_episode_length_s

    return cfg.env.config, unresolved_conf


def make_mjlab_ufo_env_cfg(
    config,
    *,
    num_envs: int,
    seed: int | None,
    mjcf_path: str | None,
    auto_reset: bool,
    robot_training: dict[str, Any] | None = None,
):
    """Create an MJLab ManagerBasedRlEnvCfg with UFO robot metadata."""
    import mujoco
    from mjlab.actuator import DcMotorActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs import mdp as mjlab_mdp
    from mjlab.envs.mdp import dr as mjlab_dr
    from mjlab.envs.mdp import terminations as mjlab_terminations
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg

    dof_names = tuple(_to_list(config.robot.dof_names))
    body_names = tuple(_to_list(config.robot.body_names))
    xml_path = Path(mjcf_path) if mjcf_path is not None else Path(G1_MJLAB_MJCF_PATH)
    xml_path = xml_path if xml_path.is_absolute() else Path(HUMANOIDVERSE_DIR).parent / xml_path
    if not xml_path.exists():
        raise FileNotFoundError(f"MJCF asset not found: {xml_path}")
    if "actuatorfrcrange" in xml_path.read_text():
        raise ValueError(f"MJLab robot XML must not contain actuatorfrcrange: {xml_path}")

    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(xml_path))
        # The UFO Isaac path uses implicit position PD targets. XML motor
        # actuators are removed so MJLab adds equivalent position actuators.
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        return spec

    stiffness = _to_float_dict(config.robot.control.stiffness)
    damping = _to_float_dict(config.robot.control.damping)
    effort_scale = float(getattr(config.robot, "dof_effort_limit_scale", 1.0))
    bfm_effort_limits = [float(x) for x in _to_list(config.robot.dof_effort_limit_list)]
    actuator_source, actuator_params = _actuator_params_from_training(dof_names, robot_training)
    effort_limits = actuator_params["effort_limit"]
    velocity_limits = actuator_params["velocity_limit"]
    armature = actuator_params["armature"]
    friction = actuator_params["friction"]

    actuators = []
    action_scale = {}
    for i, joint_name in enumerate(dof_names):
        kp = _match_joint_value(joint_name, stiffness)
        kd = _match_joint_value(joint_name, damping)
        effort_limit = effort_limits[i]
        actuators.append(
            DcMotorActuatorCfg(
                target_names_expr=(joint_name,),
                stiffness=kp,
                damping=kd,
                effort_limit=effort_limit,
                saturation_effort=effort_limit,
                velocity_limit=velocity_limits[i],
                armature=armature[i] if i < len(armature) else None,
                frictionloss=friction[i] if i < len(friction) else None,
            )
        )

        scale = float(config.robot.control.action_scale)
        if bool(config.robot.control.action_rescale):
            if kp <= 0.0:
                raise ValueError(f"Cannot action_rescale joint {joint_name}: stiffness={kp}")
            scale *= bfm_effort_limits[i] / kp
        action_scale[joint_name] = scale

    if len(actuators) != len(dof_names):
        raise ValueError(f"Expected one MJLab actuator per UFO dof, got {len(actuators)} for {len(dof_names)} dofs")
    scaled_effort_limits = [float(x) * effort_scale for x in bfm_effort_limits]
    if effort_scale != 1.0 and any(abs(a - b) < 1.0e-6 for a, b in zip(effort_limits, scaled_effort_limits)):
        raise ValueError("MJLab actuator effort limits unexpectedly include dof_effort_limit_scale")
    print(
        "[INFO] MJLab asset: "
        f"xml_path={xml_path}, actuator_source={actuator_source}, "
        f"actuator_count={len(actuators)}, joint_order={list(dof_names)}, "
        f"action_scale={[action_scale[name] for name in dof_names]}, "
        f"kp={[_match_joint_value(name, stiffness) for name in dof_names]}, "
        f"kd={[_match_joint_value(name, damping) for name in dof_names]}, "
        f"effort_limit={effort_limits}, velocity_limit={velocity_limits}, "
        f"armature={armature}, friction={friction}, "
        f"dof_effort_limit_scale={effort_scale} ignored_for_mjlab_actuator_limits",
        flush=True,
    )

    init_rot_xyzw = tuple(float(x) for x in config.robot.init_state.rot)
    init_rot_wxyz = (init_rot_xyzw[3], init_rot_xyzw[0], init_rot_xyzw[1], init_rot_xyzw[2])
    init_state = EntityCfg.InitialStateCfg(
        pos=tuple(float(x) for x in config.robot.init_state.pos),
        rot=init_rot_wxyz,
        lin_vel=tuple(float(x) for x in config.robot.init_state.lin_vel),
        ang_vel=tuple(float(x) for x in config.robot.init_state.ang_vel),
        joint_pos={name: float(config.robot.init_state.default_joint_angles[name]) for name in dof_names},
        joint_vel={".*": 0.0},
    )

    robot_cfg = EntityCfg(
        spec_fn=spec_fn,
        init_state=init_state,
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=1.0),
        sort_actuators=True,
    )
    sensors = (
        ContactSensorCfg(
            name="body_contact",
            primary=ContactMatch(mode="body", pattern=body_names, entity="robot"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=int(config.simulator.config.sim.control_decimation),
        ),
    )
    observations = {
        "actor": ObservationGroupCfg(
            terms={"joint_pos": ObservationTermCfg(func=_obs_joint_pos)},
            concatenate_terms=True,
            enable_corruption=False,
        )
    }
    actions = {
        "actions": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=dof_names,
            preserve_order=True,
            scale=action_scale,
            use_default_offset=True,
        )
    }
    reward_keys = tuple(config.rewards.reward_scales.keys())
    rewards = {key: RewardTermCfg(func=_zero_reward, weight=0.0) for key in reward_keys}
    terminations = {
        "time_out": TerminationTermCfg(func=mjlab_terminations.time_out, time_out=True),
    }
    domain_rand = config.domain_rand
    events = {}
    if bool(domain_rand.get("push_robots", False)):
        max_push_vel_xy = float(domain_rand.max_push_vel_xy)
        max_push_ang_vel = float(domain_rand.get("max_push_ang_vel", 0.0))
        velocity_range = {
            "x": (-max_push_vel_xy, max_push_vel_xy),
            "y": (-max_push_vel_xy, max_push_vel_xy),
        }
        if max_push_ang_vel > 0.0:
            velocity_range.update(
                {
                    "roll": (-max_push_ang_vel, max_push_ang_vel),
                    "pitch": (-max_push_ang_vel, max_push_ang_vel),
                    "yaw": (-max_push_ang_vel, max_push_ang_vel),
                }
            )
        events["push_robots"] = EventTermCfg(
            func=mjlab_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=tuple(float(x) for x in _to_list(domain_rand.push_interval_s)),
            params={"velocity_range": velocity_range},
        )
    if bool(domain_rand.get("randomize_base_com", False)):
        base_com_range = domain_rand.base_com_range
        events["random_base_com"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(str(config.robot.torso_name),)),
                "operation": "add",
                "ranges": {
                    0: tuple(float(x) for x in _to_list(base_com_range.x)),
                    1: tuple(float(x) for x in _to_list(base_com_range.y)),
                    2: tuple(float(x) for x in _to_list(base_com_range.z)),
                },
            },
        )
    if bool(domain_rand.get("randomize_link_mass", False)):
        events["random_link_mass"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "operation": "scale",
                "ranges": tuple(float(x) for x in _to_list(domain_rand.link_mass_range)),
            },
        )
    if bool(domain_rand.get("randomize_friction", False)):
        events["random_geom_friction"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "operation": "abs",
                "axes": [0],
                "ranges": tuple(float(x) for x in _to_list(domain_rand.friction_range)),
            },
        )

    return ManagerBasedRlEnvCfg(
        decimation=int(config.simulator.config.sim.control_decimation),
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=float(config.env_spacing),
            terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=float(config.env_spacing)),
            entities={"robot": robot_cfg},
            sensors=sensors,
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        seed=seed,
        sim=SimulationCfg(
            nconmax=512,
            njmax=4096,
            contact_sensor_maxmatch=256,
            mujoco=MujocoCfg(timestep=1.0 / float(config.simulator.config.sim.fps)),
        ),
        episode_length_s=float(config.max_episode_length_s),
        auto_reset=auto_reset,
        scale_rewards_by_dt=False,
    )


class _MjlabSimulatorView:
    """Compatibility view for code that expects ``env._env.simulator``."""

    def __init__(self, core: "HumanoidVerseMjlabCore") -> None:
        self._core = core
        self._body_list = list(core.body_names)
        self.__class__.__name__ = "MJLab"

    def refresh(self) -> None:
        core = self._core
        self.dof_pos = core.dof_pos
        self.dof_vel = core.dof_vel
        self.dof_state = torch.stack((core.dof_pos, core.dof_vel), dim=-1)
        self.robot_root_states = core.robot_root_states
        self.base_quat = core.base_quat
        self._rigid_body_pos = core.body_pos
        self._rigid_body_rot = core.body_rot
        self._rigid_body_vel = core.body_vel
        self._rigid_body_ang_vel = core.body_ang_vel
        self.contact_forces = core.contact_forces
        self.dof_pos_limits = core.dof_pos_limits
        self.hard_dof_pos_limits = core.hard_dof_pos_limits

    def render(self):
        return self._core.mjlab_env.render()


class HumanoidVerseMjlabCore:
    def __init__(self, hv_config, mjlab_env, *, creation_config: "HumanoidVerseMjlabConfig") -> None:
        self.config = hv_config
        self.mjlab_env = mjlab_env
        self.robot = mjlab_env.scene["robot"]
        self.device = str(mjlab_env.device)
        self.num_envs = int(mjlab_env.num_envs)
        self.dt = float(mjlab_env.step_dt)
        self.sim_dt = float(mjlab_env.physics_dt)
        self._creation_config = creation_config

        self.dof_names = tuple(_to_list(hv_config.robot.dof_names))
        self.body_names = tuple(_to_list(hv_config.robot.body_names))
        self.num_dof = len(self.dof_names)
        self.num_dofs = self.num_dof
        self.num_bodies = len(self.body_names)
        self.dim_actions = self.num_dof
        self.env_origins = mjlab_env.scene.env_origins

        mjlab_joint_names = tuple(self.robot.joint_names)
        mjlab_body_names = tuple(self.robot.body_names)
        missing_joints = [name for name in self.dof_names if name not in mjlab_joint_names]
        missing_bodies = [name for name in self.body_names if name not in mjlab_body_names]
        if missing_joints:
            raise ValueError(f"MJLab robot asset is missing joints from HumanoidVerse config: {missing_joints}")
        if missing_bodies:
            raise ValueError(f"MJLab robot asset is missing bodies from HumanoidVerse config: {missing_bodies}")
        self._joint_ids = torch.tensor([mjlab_joint_names.index(name) for name in self.dof_names], device=self.device, dtype=torch.long)
        self._body_ids = torch.tensor([mjlab_body_names.index(name) for name in self.body_names], device=self.device, dtype=torch.long)
        qpos_dof_names = tuple(name for name in mjlab_joint_names if name in self.dof_names)
        if len(qpos_dof_names) != self.num_dof or set(qpos_dof_names) != set(self.dof_names):
            raise ValueError(
                "MJLab qpos joints do not match HumanoidVerse dof_names: "
                f"qpos_dof_names={list(qpos_dof_names)}, dof_names={list(self.dof_names)}"
            )
        self._qpos_dof_indices = torch.tensor([self.dof_names.index(name) for name in qpos_dof_names], device=self.device, dtype=torch.long)

        action_term = self.mjlab_env.action_manager.get_term("actions")
        action_target_names = tuple(action_term.target_names)
        if len(action_target_names) != self.num_dof or set(action_target_names) != set(self.dof_names):
            raise ValueError(
                "MJLab action target joints do not match HumanoidVerse dof_names: "
                f"target_names={list(action_target_names)}, dof_names={list(self.dof_names)}"
            )
        self._action_term_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in action_target_names], device=self.device, dtype=torch.long
        )
        if action_target_names != self.dof_names:
            print(
                "[INFO] MJLab action target order differs from HumanoidVerse dof order: "
                f"action_target_names={list(action_target_names)}",
                flush=True,
            )

        self.default_dof_pos = _default_joint_pos(hv_config).to(self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.default_dof_pos_offset = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.action_target_scale = _action_target_scale(hv_config).to(self.device).unsqueeze(0)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        lower = torch.tensor(_to_list(hv_config.robot.dof_pos_lower_limit_list), dtype=torch.float32, device=self.device)
        upper = torch.tensor(_to_list(hv_config.robot.dof_pos_upper_limit_list), dtype=torch.float32, device=self.device)
        self.hard_dof_pos_limits = torch.stack((lower, upper), dim=-1)
        limit_scale = float(hv_config.rewards.reward_limit.soft_dof_pos_limit)
        center = (lower + upper) * 0.5
        radius = (upper - lower) * 0.5 * limit_scale
        self.dof_pos_limits = torch.stack((center - radius, center + radius), dim=-1)
        self.torque_limits = torch.tensor(_to_list(hv_config.robot.dof_effort_limit_list), device=self.device, dtype=torch.float32)
        self.dof_vel_limits = torch.tensor(_to_list(hv_config.robot.dof_vel_limit_list), device=self.device, dtype=torch.float32)

        self.actions = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.extras: dict[str, Any] = {"aux_rewards": {}}

        self._init_reward_scales()
        self._validate_aux_reward_semantics(hv_config)
        self.feet_indices = torch.tensor([self.body_names.index(name) for name in hv_config.robot.contact_bodies], device=self.device, dtype=torch.long)
        self.torso_index = self.body_names.index(hv_config.robot.torso_name)
        penalized = []
        for pattern in _to_list(hv_config.robot.penalize_contacts_on):
            penalized.extend([i for i, name in enumerate(self.body_names) if pattern in name])
        self.penalised_contact_indices = torch.tensor(sorted(set(penalized)), device=self.device, dtype=torch.long)
        self.left_ankle_dof_indices = torch.tensor([self.dof_names.index(n) for n in hv_config.robot.left_ankle_dof_names], device=self.device)
        self.right_ankle_dof_indices = torch.tensor([self.dof_names.index(n) for n in hv_config.robot.right_ankle_dof_names], device=self.device)

        self._init_motion_extend()
        self.is_evaluating = False
        self.average_episode_length = 0.0
        self.last_episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.num_compute_average_epl = float(self.config.rewards.num_compute_average_epl)
        self.add_noise_currculum = bool(self.config.obs.get("add_noise_currculum", False))
        self.current_noise_curriculum_value = float(self.config.obs.get("noise_initial_value", 1.0))
        self._init_motion_lib()
        self.history_handler = HVHistoryHandler(self.num_envs, hv_config.obs.obs_auxiliary, hv_config.obs.obs_dims, self.device)
        self.use_contact_in_obs_max = bool(hv_config.get("use_contact_in_obs_max", False))
        self.simulator = _MjlabSimulatorView(self)

        self._refresh_state()
        self.simulator.refresh()

    def _init_reward_scales(self) -> None:
        self.reward_scales = dict(OmegaConf.to_container(self.config.rewards.reward_scales, resolve=True))
        self.reward_scales = {k: float(v) * self.dt for k, v in self.reward_scales.items() if float(v) != 0.0}
        self.reward_names = list(self.reward_scales.keys())
        self.reward_penalty_reward_names = set(_to_list(self.config.rewards.reward_penalty_reward_names))
        self.use_reward_penalty_curriculum = bool(self.config.rewards.reward_penalty_curriculum)
        self.reward_penalty_scale = float(self.config.rewards.reward_initial_penalty_scale)

    def _validate_aux_reward_semantics(self, hv_config) -> None:
        contact_bodies = _to_list(hv_config.robot.get("contact_bodies", None))
        if len(contact_bodies) < 2:
            raise ValueError(
                "robot.contact_bodies must contain at least 2 bodies because the current MJLab reward "
                "implementation computes biped foot auxiliary terms unconditionally"
            )

        if "penalty_ankle_roll" in self.reward_scales:
            missing_fields = []
            if len(_to_list(hv_config.robot.get("left_ankle_dof_names", None))) < 2:
                missing_fields.append("robot.left_ankle_dof_names")
            if len(_to_list(hv_config.robot.get("right_ankle_dof_names", None))) < 2:
                missing_fields.append("robot.right_ankle_dof_names")
            if missing_fields:
                raise ValueError(
                    f"{', '.join(missing_fields)} must contain at least 2 joints because reward 'penalty_ankle_roll' is enabled"
                )

    def _update_average_episode_length(self, env_ids: torch.Tensor) -> None:
        if self.is_evaluating or len(env_ids) == 0:
            return
        current = torch.mean(self.last_episode_length_buf[env_ids].float()).item()
        ratio = min(float(len(env_ids)) / max(self.num_compute_average_epl, 1.0), 1.0)
        self.average_episode_length = self.average_episode_length * (1.0 - ratio) + current * ratio

    def _update_reward_penalty_curriculum(self) -> None:
        if not self.use_reward_penalty_curriculum:
            return
        if self.average_episode_length < float(self.config.rewards.reward_penalty_level_down_threshold):
            self.reward_penalty_scale *= 1.0 - float(self.config.rewards.reward_penalty_degree)
        elif self.average_episode_length > float(self.config.rewards.reward_penalty_level_up_threshold):
            self.reward_penalty_scale *= 1.0 + float(self.config.rewards.reward_penalty_degree)
        self.reward_penalty_scale = float(
            np.clip(
                self.reward_penalty_scale,
                float(self.config.rewards.reward_min_penalty_scale),
                float(self.config.rewards.reward_max_penalty_scale),
            )
        )

    def _update_obs_noise_curriculum(self) -> None:
        if not self.add_noise_currculum:
            return
        if self.average_episode_length < float(self.config.obs.soft_dof_pos_curriculum_level_down_threshold):
            self.current_noise_curriculum_value *= 1.0 - float(self.config.obs.soft_dof_pos_curriculum_degree)
        elif self.average_episode_length > float(self.config.obs.soft_dof_pos_curriculum_level_up_threshold):
            self.current_noise_curriculum_value *= 1.0 + float(self.config.obs.soft_dof_pos_curriculum_degree)
        self.current_noise_curriculum_value = float(
            np.clip(
                self.current_noise_curriculum_value,
                float(self.config.obs.noise_value_min),
                float(self.config.obs.noise_value_max),
            )
        )

    def _apply_obs_scale_noise(self, key: str, value: torch.Tensor) -> torch.Tensor:
        obs_scales = self.config.obs.obs_scales
        noise_scales = self.config.obs.noise_scales
        scale = float(obs_scales.get(key, 1.0))
        noise_scale = 0.0 if self.is_evaluating else float(noise_scales.get(key, 0.0))
        if self.add_noise_currculum:
            noise_scale *= self.current_noise_curriculum_value
        if noise_scale != 0.0:
            value = value + (torch.rand_like(value) * 2.0 - 1.0) * noise_scale
        return value * scale

    def _init_motion_lib(self) -> None:
        self.config.robot.motion.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        self._motion_lib.load_motions_for_training(max_num_seqs=self.num_envs)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions
        self._resample_motion_time_and_ids(torch.arange(self.num_envs, device=self.device))

    def _init_motion_extend(self) -> None:
        extend_parent_ids, extend_pos, extend_rot = [], [], []
        for item in _to_list(self.config.robot.motion.extend_config):
            extend_parent_ids.append(self.body_names.index(item["parent_name"]))
            extend_pos.append(item["pos"])
            extend_rot.append(item["rot"])
        self.num_extend_bodies = len(extend_parent_ids)
        if self.num_extend_bodies:
            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1, 1)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1, 1)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.body_names = tuple(list(self.body_names) + [item["joint_name"] for item in _to_list(self.config.robot.motion.extend_config)])
        else:
            self.extend_body_parent_ids = torch.empty(0, device=self.device, dtype=torch.long)
        self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, device=self.device)

    def _resample_motion_time_and_ids(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_start_times[env_ids] = 0.0
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])

    def _randomize_default_dof_pos_offset(self, env_ids: torch.Tensor) -> None:
        if bool(self.config.domain_rand.get("randomize_default_dof_pos", False)):
            offset_range = self.config.domain_rand.default_dof_pos_noise_range
            self.default_dof_pos_offset[env_ids] = torch.empty(
                len(env_ids), self.num_dof, device=self.device, dtype=torch.float32
            ).uniform_(float(offset_range[0]), float(offset_range[1]))
        else:
            self.default_dof_pos_offset[env_ids] = 0.0

    def _refresh_state(self) -> None:
        data = self.robot.data
        self.dof_pos = data.joint_pos[:, self._joint_ids].clone()
        self.dof_vel = data.joint_vel[:, self._joint_ids].clone()
        root_pose_w = data.root_link_pose_w.clone()
        root_vel_w = data.root_link_vel_w.clone()
        self.base_quat = wxyz_to_xyzw(root_pose_w[:, 3:7])
        self.robot_root_states = torch.cat([root_pose_w[:, :3], self.base_quat, root_vel_w], dim=-1)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, root_vel_w[:, :3], w_last=True)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, root_vel_w[:, 3:6], w_last=True)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec, w_last=True)
        body_pose = data.body_link_pose_w[:, self._body_ids].clone()
        body_vel = data.body_link_vel_w[:, self._body_ids].clone()
        self.body_pos = body_pose[..., :3]
        self.body_rot = wxyz_to_xyzw(body_pose[..., 3:7])
        self.body_vel = body_vel[..., :3]
        self.body_ang_vel = body_vel[..., 3:6]
        self.torques = data.qfrc_actuator[:, self._joint_ids].clone()
        self.contact_forces = self._read_contact_forces()
        self.episode_length_buf = self.mjlab_env.episode_length_buf.clone()

    def _read_contact_forces(self) -> torch.Tensor:
        forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        sensor = self.mjlab_env.scene.sensors.get("body_contact")
        if sensor is None:
            return forces
        contact_data = sensor.data
        if contact_data.force is None:
            return forces
        names = [name.split("/")[-1] for name in sensor.primary_names]
        for i, name in enumerate(names):
            if name in self.body_names[: self.num_bodies]:
                forces[:, self.body_names.index(name), :] = contact_data.force[:, i, :]
        return forces

    def _extend_body_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.num_extend_bodies == 0:
            return self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel
        rotated_pos = my_quat_rotate(
            self.body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3),
        ).view(self.num_envs, -1, 3)
        extend_pos = rotated_pos + self.body_pos[:, self.extend_body_parent_ids]
        extend_rot = quat_mul(
            self.body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            w_last=True,
        ).view(self.num_envs, -1, 4)
        extend_ang_vel = self.body_ang_vel[:, self.extend_body_parent_ids]
        extend_vel = self.body_vel[:, self.extend_body_parent_ids] + torch.cross(
            extend_ang_vel, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2
        )
        return (
            torch.cat([self.body_pos, extend_pos], dim=1),
            torch.cat([self.body_rot, extend_rot], dim=1),
            torch.cat([self.body_vel, extend_vel], dim=1),
            torch.cat([self.body_ang_vel, extend_ang_vel], dim=1),
        )

    def _compute_reference_and_privileged_obs(self) -> None:
        body_pos, body_rot, body_vel, body_ang_vel = self._extend_body_state()
        self._rigid_body_pos_extend = body_pos
        self._rigid_body_rot_extend = body_rot
        self._rigid_body_vel_extend = body_vel
        self._rigid_body_ang_vel_extend = body_ang_vel

        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=self.env_origins)
        self.ref_body_pos_extend = motion_res["rg_pos_t"]
        self.ref_body_rot_extend = motion_res["rg_rot_t"]
        self.ref_body_vel_extend = motion_res["body_vel_t"]
        self.ref_body_ang_vel_extend = motion_res["body_ang_vel_t"]
        self.dif_global_body_pos = self.ref_body_pos_extend - body_pos
        self.dif_joint_angles = motion_res["dof_pos"] - self.dof_pos
        self.dif_joint_velocities = motion_res["dof_vel"] - self.dof_vel
        obs_dict = compute_humanoid_observations_max(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            local_root_obs=True,
            root_height_obs=bool(self.config.obs.get("root_height_obs", True)),
        )
        self._max_local_self = torch.cat([v for v in obs_dict.values()], dim=-1)

    def _raw_actor_obs(self) -> dict[str, torch.Tensor]:
        self._compute_reference_and_privileged_obs()
        dof_pos_rel = self.dof_pos - (self.default_dof_pos + self.default_dof_pos_offset)
        obs_data = {
            "actions": self._apply_obs_scale_noise("actions", self.actions),
            "base_ang_vel": self._apply_obs_scale_noise("base_ang_vel", self.base_ang_vel),
            "dof_pos": self._apply_obs_scale_noise("dof_pos", dof_pos_rel),
            "dof_vel": self._apply_obs_scale_noise("dof_vel", self.dof_vel),
            "projected_gravity": self._apply_obs_scale_noise("projected_gravity", self.projected_gravity),
            "max_local_self": self._apply_obs_scale_noise("max_local_self", self._max_local_self),
        }
        history_config = self.config.obs.obs_auxiliary["history_actor"]
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensors.append(history_tensor.reshape(history_tensor.shape[0], -1))
        history_actor = torch.cat(history_tensors, dim=1)
        history_actor = self._apply_obs_scale_noise("history_actor", history_actor)
        raw = {
            **obs_data,
            "history_actor": history_actor,
        }
        self.obs_buf_dict_raw = {"actor_obs": raw}
        for key in history_config.keys():
            value = obs_data[key]
            self.history_handler.add(key, value)
        return raw

    def get_observation(self, *, to_numpy: bool = True, include_last_action: bool = True, include_history_actor: bool = True):
        raw_obs = self._raw_actor_obs()
        obs = {
            "state": torch.cat([raw_obs["dof_pos"], raw_obs["dof_vel"], raw_obs["projected_gravity"], raw_obs["base_ang_vel"]], dim=-1),
            "privileged_state": raw_obs["max_local_self"],
        }
        if include_last_action:
            obs["last_action"] = raw_obs["actions"]
        obs["time"] = self.episode_length_buf.unsqueeze(-1)
        if include_history_actor:
            obs["history_actor"] = raw_obs["history_actor"]
        if to_numpy:
            obs = tree_map(lambda x: x.detach().cpu().numpy(), obs)
        return obs

    def _compute_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        contact = self.contact_forces
        foot_contact = contact[:, self.feet_indices, 2] > 1.0
        aux["penalty_torques"] = torch.sum(torch.square(self.torques), dim=1)
        aux["penalty_action_rate"] = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        lower, upper = self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1]
        aux["limits_dof_pos"] = torch.sum((-(self.dof_pos - lower).clip(max=0.0)) + ((self.dof_pos - upper).clip(min=0.0)), dim=1)
        vel_limit = self.dof_vel_limits * float(self.config.rewards.reward_limit.soft_dof_vel_limit)
        aux["limits_dof_vel"] = torch.sum((torch.abs(self.dof_vel) - vel_limit).clip(min=0.0, max=1.0), dim=1)
        torque_limit = self.torque_limits * float(self.config.rewards.reward_limit.soft_torque_limit)
        aux["limits_torque"] = torch.sum((torch.abs(self.torques) - torque_limit).clip(min=0.0), dim=1)
        if len(self.penalised_contact_indices) > 0:
            undesired = torch.any(torch.abs(contact[:, self.penalised_contact_indices, :]) > 1.0, dim=(1, 2))
        else:
            undesired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        aux["penalty_undesired_contact"] = undesired.float()
        left_ankle_roll = self.dof_pos[:, self.left_ankle_dof_indices[1:2]]
        right_ankle_roll = self.dof_pos[:, self.right_ankle_dof_indices[1:2]]
        aux["penalty_ankle_roll"] = torch.sum(torch.square(left_ankle_roll) + torch.square(right_ankle_roll), dim=1)
        left_quat = self.body_rot[:, self.feet_indices[0]]
        right_quat = self.body_rot[:, self.feet_indices[1]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec, w_last=True)
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec, w_last=True)
        aux["penalty_feet_ori"] = (
            torch.sum(torch.square(left_gravity[:, :2]), dim=1).sqrt() * foot_contact[:, 0]
            + torch.sum(torch.square(right_gravity[:, :2]), dim=1).sqrt() * foot_contact[:, 1]
        )
        foot_vel = self.body_vel[:, self.feet_indices]
        aux["penalty_slippage"] = torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(contact[:, self.feet_indices, :], dim=-1) > 1.0), dim=1)
        forward_left = my_quat_rotate(left_quat, self.forward_vec)
        forward_right = my_quat_rotate(right_quat, self.forward_vec)
        root_forward = my_quat_rotate(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])
        aux["feet_heading_alignment"] = torch.abs(wrap_to_pi(torch.atan2(forward_left[:, 1], forward_left[:, 0]) - heading_root)) + torch.abs(
            wrap_to_pi(torch.atan2(forward_right[:, 1], forward_right[:, 0]) - heading_root)
        )

        reward = torch.zeros(self.num_envs, device=self.device)
        for name, scale in self.reward_scales.items():
            if name not in aux:
                continue
            rew = aux[name] * scale
            if name in self.reward_penalty_reward_names and self.use_reward_penalty_curriculum:
                rew *= self.reward_penalty_scale
            reward += rew
        return reward, aux

    def _normalized_action(self, actions: torch.Tensor) -> torch.Tensor:
        if bool(self.config.robot.control.normalize_action):
            actions = actions * float(self.config.robot.control.normalize_action_to) / float(self.config.robot.control.normalize_action_from)
        return torch.clamp(actions, -float(self.config.robot.control.action_clip_value), float(self.config.robot.control.action_clip_value))

    def _mjlab_action_input(self) -> torch.Tensor:
        action_indices = self._action_term_dof_indices
        return self.actions[:, action_indices] + self.default_dof_pos_offset[:, action_indices] / torch.clamp(
            self.action_target_scale[:, action_indices], min=1.0e-6
        )

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device, dtype=torch.float32)
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.actions[:] = self._normalized_action(actions)
        mjlab_actions = self._mjlab_action_input()
        _, _, terminated, time_outs, _ = self.mjlab_env.step(mjlab_actions)
        self._refresh_state()
        reward, aux = self._compute_reward()
        reset = torch.logical_or(terminated.bool(), time_outs.bool())
        self.reset_buf = reset
        self.time_out_buf = time_outs.bool()
        self.rew_buf = reward
        self.extras["aux_rewards"] = {k: v.clone().detach() for k, v in aux.items()}
        if self.use_reward_penalty_curriculum:
            self.extras["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float32, device=self.device)
            self.extras["average_episode_length"] = torch.tensor(self.average_episode_length, dtype=torch.float32, device=self.device)
        if self.add_noise_currculum:
            self.extras["current_noise_curriculum_value"] = torch.tensor(
                self.current_noise_curriculum_value, dtype=torch.float32, device=self.device
            )
        if torch.any(reset):
            reset_ids = reset.nonzero(as_tuple=False).flatten()
            self.last_episode_length_buf[reset_ids] = self.episode_length_buf[reset_ids]
            self._update_average_episode_length(reset_ids)
            self._update_reward_penalty_curriculum()
            self._update_obs_noise_curriculum()
            self.reset_idx(reset_ids)
        else:
            self.simulator.refresh()
        return None, reward, reset, {"time_outs": time_outs.bool(), "aux_rewards": self.extras["aux_rewards"]}

    def reset_all(self, target_states: dict[str, torch.Tensor] | None = None):
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(env_ids, target_states=target_states)
        return None, {}

    def reset_idx(self, env_ids: torch.Tensor, target_states: dict[str, torch.Tensor] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.mjlab_env.reset(env_ids=env_ids)
        self._randomize_default_dof_pos_offset(env_ids)
        if target_states is not None:
            root_xyzw = target_states["root_states"][env_ids].to(self.device, dtype=torch.float32)
            dof_state = target_states["dof_states"][env_ids].to(self.device, dtype=torch.float32)
            joint_pos = dof_state[..., 0]
            joint_vel = dof_state[..., 1]
        else:
            self._resample_motion_time_and_ids(env_ids)
            motion_times = self.motion_start_times[env_ids]
            motion_res = self._motion_lib.get_motion_state(self.motion_ids[env_ids], motion_times, offset=self.env_origins[env_ids])
            root_pos = motion_res["root_pos"]
            root_rot = motion_res["root_rot"]
            root_vel = motion_res["root_vel"]
            root_ang_vel = motion_res["root_ang_vel"]
            if self.config.get("lie_down_init", False):
                mask = torch.rand(len(env_ids), device=self.device) < float(getattr(self.config, "lie_down_init_prob", 0.0))
                if torch.any(mask):
                    root_pos = root_pos.clone()
                    root_rot = root_rot.clone()
                    root_pos[mask, 2] = 0.5
                    sign = 1 if random.random() < 0.5 else -1
                    rot_quat = quat_from_angle_axis(
                        torch.tensor(sign * (-torch.pi / 2), device=self.device),
                        torch.tensor([1.0, 0.0, 0.0], device=self.device),
                        w_last=True,
                    )
                    root_rot[mask] = quat_mul(rot_quat.expand_as(root_rot[mask]), root_rot[mask], w_last=True)
            root_pos = root_pos + torch.randn_like(root_pos) * float(self.config.init_noise_scale.root_pos) * float(self.config.noise_to_initial_level)
            root_rot = quat_mul(
                _small_random_quaternions(
                    len(env_ids),
                    float(self.config.init_noise_scale.root_rot) * 3.14 / 180.0 * float(self.config.noise_to_initial_level),
                    self.device,
                ),
                root_rot,
                w_last=True,
            )
            root_vel = root_vel + torch.randn_like(root_vel) * float(self.config.init_noise_scale.root_vel) * float(self.config.noise_to_initial_level)
            root_ang_vel = root_ang_vel + torch.randn_like(root_ang_vel) * float(self.config.init_noise_scale.root_ang_vel) * float(
                self.config.noise_to_initial_level
            )
            root_xyzw = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
            joint_pos = motion_res["dof_pos"] + torch.randn_like(motion_res["dof_pos"]) * float(self.config.init_noise_scale.dof_pos) * float(
                self.config.noise_to_initial_level
            )
            joint_vel = motion_res["dof_vel"] + torch.randn_like(motion_res["dof_vel"]) * float(self.config.init_noise_scale.dof_vel) * float(
                self.config.noise_to_initial_level
            )

        root_wxyz = torch.cat([root_xyzw[:, :3], xyzw_to_wxyz(root_xyzw[:, 3:7]), root_xyzw[:, 7:13]], dim=-1)
        self.robot.write_root_state_to_sim(root_wxyz, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self._joint_ids, env_ids=env_ids)
        self.mjlab_env.scene.write_data_to_sim()
        self.mjlab_env.sim.forward()
        self.mjlab_env._manual_reset_pending[env_ids] = False
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.history_handler.reset(env_ids)
        self._refresh_state()
        self.simulator.refresh()

    def set_is_evaluating(self, global_rank: int = 0):
        self.is_evaluating = True
        self.begin_seq_motion_samples(global_rank)

    def begin_seq_motion_samples(self, global_rank: int = 0):
        self._motion_lib.load_motions_for_evaluation(start_idx=global_rank * self.num_envs)
        self.reset_all()

    def set_is_training(self):
        self.is_evaluating = False
        self.resample_motion()

    def resample_motion(self):
        self._motion_lib.load_motions_for_training(max_num_seqs=self.num_envs)
        self.reset_all()

    def close(self):
        return self.mjlab_env.close()


class HumanoidVerseMjlabVectorEnv(VectorEnv):
    """Gymnasium VectorEnv wrapper matching HumanoidVerseIsaacVectorEnv."""

    def __init__(
        self,
        env: HumanoidVerseMjlabCore,
        *,
        add_time_aware_observation: bool = True,
        include_last_action: bool = True,
        context_length: int | None = None,
        include_history_actor: bool = True,
        include_history_noaction: bool = False,
    ):
        super().__init__()
        self._env = env
        self.spec = None
        self.num_envs = env.num_envs
        self.add_time_aware_observation = add_time_aware_observation
        self.include_last_action = include_last_action
        self.context_length = context_length
        self.include_history_actor = include_history_actor
        self.include_history_noaction = include_history_noaction
        self.history_handler = None

        self.single_action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(env.num_dof,), dtype=np.float32)
        action_space_shape = (self.num_envs,) + self.single_action_space.shape
        self.action_space = gymnasium.spaces.Box(
            low=np.tile(self.single_action_space.low, (self.num_envs, 1)),
            high=np.tile(self.single_action_space.high, (self.num_envs, 1)),
            shape=action_space_shape,
            dtype=np.float32,
        )
        example_observation, _ = self.reset()
        observation_spaces = {}
        for key, value in example_observation.items():
            observation_spaces[key] = gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=value.shape, dtype=value.dtype)
        self.observation_space = gymnasium.spaces.Dict(observation_spaces)

    @property
    def single_observation_space(self):
        single_obs_spaces = {}
        for key, space in self.observation_space.spaces.items():
            single_obs_spaces[key] = gymnasium.spaces.Box(low=space.low[0], high=space.high[0], shape=space.shape[1:], dtype=space.dtype)
        return gymnasium.spaces.Dict(single_obs_spaces)

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> Env:
        return self._env

    @property
    def unwrapped(self):
        return self.base_env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        to_numpy: bool = True,
        reset_to_default_pose: bool = False,
        target_states: dict[str, torch.Tensor] | None = None,
    ):
        del seed, options, reset_to_default_pose
        self.base_env.reset_all(target_states=target_states)
        observation = self.base_env.get_observation(
            to_numpy=to_numpy,
            include_last_action=self.include_last_action,
            include_history_actor=self.include_history_actor,
        )
        qpos, qvel = self._get_qpos_qvel(to_numpy=to_numpy)
        return observation, {"qpos": qpos, "qvel": qvel}

    def _get_qpos_qvel(self, to_numpy: bool = True):
        base_pos_wxyz = torch.cat([self._env.robot_root_states[:, :3], xyzw_to_wxyz(self._env.robot_root_states[:, 3:7])], dim=-1)
        qpos_dof = self._env.dof_pos[:, self._env._qpos_dof_indices]
        qvel_dof = self._env.dof_vel[:, self._env._qpos_dof_indices]
        qpos = torch.cat([base_pos_wxyz, qpos_dof], dim=-1)
        qvel = torch.cat([self._env.robot_root_states[:, 7:10], self._env.base_ang_vel, qvel_dof], dim=-1)
        if to_numpy:
            return qpos.detach().cpu().numpy(), qvel.detach().cpu().numpy()
        return qpos, qvel

    def step(self, actions: Union[torch.Tensor, np.ndarray, Dict], to_numpy: bool = True):
        if isinstance(actions, dict):
            actions = actions["actions"]
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, device=self._env.device, dtype=torch.float32)
        _, reward, reset, new_info = self.base_env.step(actions)
        time_outs = new_info["time_outs"].bool()
        terminated = torch.logical_and(reset.bool(), ~time_outs)
        truncated = time_outs
        observation = self.base_env.get_observation(
            to_numpy=to_numpy,
            include_last_action=self.include_last_action,
            include_history_actor=self.include_history_actor,
        )
        qpos, qvel = self._get_qpos_qvel(to_numpy=to_numpy)
        new_info["qpos"] = qpos
        new_info["qvel"] = qvel
        if to_numpy:
            reward = reward.detach().cpu().numpy()
            terminated = terminated.detach().cpu().numpy()
            truncated = truncated.detach().cpu().numpy()
            new_info["aux_rewards"] = {k: v.detach().cpu().numpy() for k, v in new_info["aux_rewards"].items()}
        return observation, reward, terminated, truncated, new_info

    def close(self):
        return self.base_env.close()

    def render(self):
        return self.base_env.mjlab_env.render()


class HumanoidVerseMjlabConfig(BaseConfig):
    name: tp.Literal["humanoidverse_mjlab"] = "humanoidverse_mjlab"

    device: str = "cuda:0"
    lafan_tail_path: str | list[str]
    data_mix_weights: list[float] | None = None
    mjcf_path: str | None = None
    robot_config_path: str | None = None
    robot_training: dict[str, Any] | None = None
    max_episode_length_s: float | None = None
    disable_obs_noise: bool = False
    disable_domain_randomization: bool = False
    relative_config_path: str = HYDRA_CONFIG_REL_PATH
    include_last_action: bool = True
    hydra_overrides: tp.List[str] = pydantic.Field(default_factory=list)
    context_length: int | None = None
    include_history_actor: bool = False
    include_history_noaction: bool = False
    root_height_obs: bool = False
    auto_reset: bool = False
    seed: int | None = None

    def build(self, num_envs: int = 1) -> tp.Tuple[HumanoidVerseMjlabVectorEnv, tp.Any]:
        assert num_envs >= 1
        from mjlab.envs import ManagerBasedRlEnv

        hv_config, unresolved_conf = _compose_humanoidverse_config(
            num_envs=num_envs,
            relative_config_path=self.relative_config_path,
            hydra_overrides=list(self.hydra_overrides),
            lafan_tail_path=self.lafan_tail_path,
            data_mix_weights=self.data_mix_weights,
            disable_obs_noise=self.disable_obs_noise,
            disable_domain_randomization=self.disable_domain_randomization,
            max_episode_length_s=self.max_episode_length_s,
            root_height_obs=self.root_height_obs,
            robot_training=self.robot_training,
        )
        mjlab_cfg = make_mjlab_ufo_env_cfg(
            hv_config,
            num_envs=num_envs,
            seed=self.seed,
            mjcf_path=self.mjcf_path,
            auto_reset=self.auto_reset,
            robot_training=self.robot_training,
        )
        mjlab_env = ManagerBasedRlEnv(mjlab_cfg, device=self.device)
        core = HumanoidVerseMjlabCore(hv_config, mjlab_env, creation_config=self)
        env = HumanoidVerseMjlabVectorEnv(
            core,
            include_last_action=self.include_last_action,
            context_length=self.context_length,
            include_history_actor=self.include_history_actor,
            include_history_noaction=self.include_history_noaction,
        )
        env._creation_config = self
        return env, {"unresolved_conf": unresolved_conf, "mjlab_env_cfg": mjlab_cfg}
