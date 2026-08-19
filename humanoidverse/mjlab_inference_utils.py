"""Shared utilities for MJLab inference entrypoints.

These helpers intentionally avoid importing IsaacLab/IsaacSim code.  They
centralize MJLab env construction, checkpoint device handling and pure MuJoCo
qpos rendering used by tracking, goal and reward inference scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco
import numpy as np
import torch

import humanoidverse
from humanoidverse.agents.envs.humanoidverse_mjlab import (
    HumanoidVerseMjlabConfig,
    G1_MJLAB_MJCF_PATH,
)
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from humanoidverse.utils.robot_spec import (
    assert_robot_configs_compatible,
    load_robot_training_spec,
    resolve_robot_config_path,
)


if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = HUMANOIDVERSE_DIR.parent
DEFAULT_ROBOT_CONFIG = "configs/robots/g1_29dof.yaml"
DEFAULT_INFERENCE_DATA_PATH = Path("humanoidverse/data/lafan_29dof_10s-clipped.pkl")
G1_MJLAB_DOF_NAMES = (
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


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        name,
        nargs="?",
        const=True,
        default=default,
        type=str2bool,
        help=help_text,
    )


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def add_robot_config_manifest_args(parser: argparse.ArgumentParser, *, purpose: str) -> None:
    parser.add_argument("--robot-config", type=Path, default=None, help="Robot YAML for rollout and rendering.")
    parser.add_argument("--data-manifest", type=Path, default=None, help=f"Motion data manifest. Use with --dataset for {purpose}.")
    parser.add_argument("--dataset", default=None, help="Dataset name inside --data-manifest.")
    parser.add_argument("--rebuild-motion-cache", action="store_true", help="Rebuild manifest-generated motion pkl cache.")


def resolve_inference_robot_config(
    cli_robot_config: str | Path | None,
    manifest_robot_config: str | Path | None,
) -> Path:
    if cli_robot_config is not None and manifest_robot_config is not None:
        return assert_robot_configs_compatible(cli_robot_config, manifest_robot_config)
    if cli_robot_config is not None:
        return resolve_robot_config_path(cli_robot_config)
    if manifest_robot_config is not None:
        return resolve_robot_config_path(manifest_robot_config)
    return resolve_robot_config_path(DEFAULT_ROBOT_CONFIG)


def resolve_inference_data_and_robot_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        if args.dataset is None:
            parser.error("--dataset is required when --data-manifest is provided")
        manifest_robot_config = prepare_manifest_robot_config_path(args.data_manifest)
        args.data_path = Path(
            prepare_manifest_dataset_path(
                args.data_manifest,
                args.dataset,
                split="inference",
                rebuild_cache=bool(args.rebuild_motion_cache),
            )
        )
    args.robot_config = resolve_inference_robot_config(args.robot_config, manifest_robot_config)
    return args


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def _first_worldbody_body(root: ET.Element) -> ET.Element | None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return None
    return worldbody.find("body")


def _absolutize_compiler_paths(root: ET.Element, source_xml: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        return
    for attr in ("meshdir", "texturedir", "assetdir"):
        raw_path = compiler.get(attr)
        if not raw_path:
            continue
        asset_path = Path(raw_path)
        if not asset_path.is_absolute():
            compiler.set(attr, str((source_xml.parent / asset_path).resolve()))


def write_mjlab_relabel_xml(
    robot_xml: Path,
    output_dir: Path,
    control_joint_names: list[str],
    robot_name: str,
    *,
    root_body_name: str | None = None,
) -> Path:
    """Create a relabel-only MuJoCo XML with one motor per controlled joint."""

    if not control_joint_names:
        raise ValueError("control_joint_names must contain at least one joint")

    source_xml = Path(robot_xml).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_xml)
    root = tree.getroot()
    _absolutize_compiler_paths(root, source_xml)

    pelvis_body = _find_body(root, "pelvis")
    if pelvis_body is None and root_body_name is not None:
        pelvis_body = _find_body(root, root_body_name)
    if pelvis_body is None:
        pelvis_body = _first_worldbody_body(root)
    if pelvis_body is None:
        raise ValueError(f"Robot XML has no worldbody body for reward relabeling: {source_xml}")
    if pelvis_body.get("name") != "pelvis" and _find_body(root, "pelvis") is None:
        pelvis_body.set("name", "pelvis")

    site_name = "ufo_relabel_imu_site"
    if not any(site.get("name") == site_name for site in root.iter("site")):
        ET.SubElement(
            pelvis_body,
            "site",
            {
                "name": site_name,
                "pos": "0 0 0",
                "size": "0.01",
            },
        )

    for sensor in list(root.findall("sensor")):
        root.remove(sensor)
    sensor_root = ET.SubElement(root, "sensor")
    ET.SubElement(sensor_root, "subtreelinvel", {"name": "torso_link_subtreelinvel", "body": "pelvis"})
    ET.SubElement(sensor_root, "framezaxis", {"name": "upvector_torso", "objtype": "site", "objname": site_name})
    ET.SubElement(sensor_root, "gyro", {"name": "imu-angular-velocity", "site": site_name})

    for actuator in list(root.findall("actuator")):
        root.remove(actuator)
    actuator_root = ET.SubElement(root, "actuator")
    for joint_name in control_joint_names:
        ET.SubElement(
            actuator_root,
            "motor",
            {
                "name": f"{joint_name}_motor",
                "joint": joint_name,
                "gear": "1",
            },
        )

    safe_robot_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(robot_name)).strip("._") or "robot"
    output_path = output_dir / f"{safe_robot_name}_reward_relabel.xml"
    tree.write(output_path, encoding="utf-8", xml_declaration=False)

    model = mujoco.MjModel.from_xml_path(str(output_path))
    if int(model.nu) != len(control_joint_names):
        raise RuntimeError(
            f"Expected relabel XML ctrl dim {len(control_joint_names)}, got model.nu={model.nu}: {output_path}"
        )
    data = mujoco.MjData(model)
    if int(data.ctrl.size) != len(control_joint_names):
        raise RuntimeError(
            f"Expected relabel XML data.ctrl dim {len(control_joint_names)}, got {data.ctrl.size}: {output_path}"
        )
    return output_path


def write_g1_mjlab_relabel_xml(source_xml: Path, output_dir: Path) -> Path:
    """Create a G1 MuJoCo XML with 29 ctrl slots for reward relabeling.

    MJLab adds DC motor actuators from Python config at env construction time,
    so the raw G1 XML intentionally has ``nu == 0``. The reward
    relabel path calls ``data.ctrl[:] = action`` directly; it therefore needs a
    pure MuJoCo model with one actuator per policy action. These gear=1 motors
    are used only to size/populate ``data.ctrl`` and do not change the qpos/qvel
    samples loaded from the replay buffer.
    """

    source_xml = Path(source_xml).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir:
            meshdir_path = Path(meshdir)
            if not meshdir_path.is_absolute():
                compiler.set("meshdir", str((source_xml.parent / meshdir_path).resolve()))

    sensor_root = root.find("sensor")
    if sensor_root is None:
        sensor_root = ET.SubElement(root, "sensor")
    sensor_names = {sensor.get("name") for sensor in list(sensor_root) if sensor.get("name")}

    def add_sensor(tag: str, name: str, **attrs: str) -> None:
        if name in sensor_names:
            return
        ET.SubElement(sensor_root, tag, {"name": name, **attrs})
        sensor_names.add(name)

    add_sensor("subtreelinvel", "torso_link_subtreelinvel", body="torso_link")
    add_sensor("framelinvel", "frame_vel", objtype="site", objname="imu_in_torso")
    add_sensor("framezaxis", "upvector_torso", objtype="site", objname="imu_in_torso")
    add_sensor("gyro", "imu-angular-velocity", site="imu_in_torso")

    for actuator in list(root.findall("actuator")):
        root.remove(actuator)
    actuator_root = ET.SubElement(root, "actuator")
    for joint_name in G1_MJLAB_DOF_NAMES:
        ET.SubElement(
            actuator_root,
            "motor",
            {
                "name": f"{joint_name}_motor",
                "joint": joint_name,
                "gear": "1",
            },
        )

    output_path = output_dir / "g1_mjlab_reward_relabel.xml"
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return output_path


def checkpoint_load_device(device: str) -> str:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        return "cuda"
    if torch_device.type == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported inference device: {device}")


def replace_hydra_override(overrides: list[str], key: str, value: Any) -> list[str]:
    prefix = f"{key}="
    return [item for item in overrides if not item.startswith(prefix)] + [f"{key}={value}"]


def load_mjlab_env_cfg(
    model_folder: Path,
    *,
    data_path: Path | None,
    robot_config: Path | None = None,
    device: str,
    headless: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    max_episode_length_s: float,
) -> tuple[HumanoidVerseMjlabConfig, bool]:
    with (model_folder / "config.json").open("r") as f:
        config = json.load(f)

    env_config = dict(config["env"])
    use_root_height_obs = bool(env_config.get("root_height_obs", False))
    env_config["device"] = device
    if robot_config is not None:
        training_spec = load_robot_training_spec(robot_config)
        env_config["mjcf_path"] = training_spec.robot.xml_path
        env_config["robot_config_path"] = str(training_spec.config_path)
        env_config["robot_training"] = training_spec.to_env_dict()
    elif env_config.get("robot_training"):
        env_config["mjcf_path"] = env_config["robot_training"]["robot"]["xml_path"]
    else:
        env_config["mjcf_path"] = G1_MJLAB_MJCF_PATH
    env_config["disable_domain_randomization"] = disable_dr
    env_config["disable_obs_noise"] = disable_obs_noise
    env_config["auto_reset"] = False
    env_config["max_episode_length_s"] = max_episode_length_s

    if data_path is not None:
        env_config["lafan_tail_path"] = str(data_path.expanduser().resolve())
    elif DEFAULT_INFERENCE_DATA_PATH.exists():
        env_config["lafan_tail_path"] = str(DEFAULT_INFERENCE_DATA_PATH)
    else:
        motion_path = Path(env_config.get("lafan_tail_path", ""))
        if not motion_path.is_absolute() and not motion_path.exists():
            candidate = PROJECT_ROOT / motion_path
            if candidate.exists():
                env_config["lafan_tail_path"] = str(candidate)

    overrides = list(env_config.get("hydra_overrides") or [])
    overrides = replace_hydra_override(overrides, "env.config.max_episode_length_s", max_episode_length_s)
    overrides = replace_hydra_override(overrides, "env.config.headless", str(headless))
    env_config["hydra_overrides"] = overrides

    return HumanoidVerseMjlabConfig(**env_config), use_root_height_obs


def to_rgb_uint8(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        for key in ("rgb", "image", "frame"):
            if key in frame:
                frame = frame[key]
                break
        else:
            raise ValueError(f"Cannot find RGB image in render dict keys={sorted(frame)}")
    if isinstance(frame, (list, tuple)):
        if len(frame) == 0:
            raise ValueError("render returned an empty frame list")
        frame = frame[0]
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    array = np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected an RGB image, got shape {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected 3 color channels, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
    else:
        array = np.clip(array, 0, 255)
    return np.ascontiguousarray(array.astype(np.uint8))


def _limit_absolute_near_clip(model: mujoco.MjModel, max_near: float = 0.015) -> float:
    """Keep large merged scenes from pushing the near plane through the robot."""
    extent = max(float(model.stat.extent), 1.0e-6)
    model.vis.map.znear = min(float(model.vis.map.znear), float(max_near) / extent)
    return float(model.vis.map.znear) * extent


def _inference_scene_option() -> mujoco.MjvOption:
    """Show collision terrain, which MJLab stores in geom group 5."""
    option = mujoco.MjvOption()
    option.geomgroup[5] = 1
    return option


def _style_untextured_terrain(model: mujoco.MjModel) -> None:
    """Give default achromatic terrain a readable neutral render color."""
    terrain = np.asarray(model.geom_group) == 5
    rgb = np.asarray(model.geom_rgba[:, :3])
    default_gray = (np.ptp(rgb, axis=1) < 0.02) & (np.mean(rgb, axis=1) >= 0.45)
    model.geom_rgba[terrain & default_gray] = np.array([0.24, 0.30, 0.27, 1.0])
    model.geom_matid[terrain] = -1


def _camera_lookat_from_root(root_pos: np.ndarray) -> np.ndarray:
    """Track the pelvis in world coordinates without assuming ground z=0."""
    root_pos = np.asarray(root_pos, dtype=np.float64).reshape(-1)
    if root_pos.size < 3 or not np.isfinite(root_pos[:3]).all():
        raise ValueError(f"Expected a finite root xyz for camera tracking, got {root_pos}")
    return root_pos[:3].copy()


class MujocoQposRenderer:
    """Pure MuJoCo renderer for qpos playback from an MJCF."""

    def __init__(
        self,
        xml_path: Path | None,
        render_size: int = 480,
        *,
        scene_spec: mujoco.MjSpec | None = None,
        source_xml_path: Path | None = None,
        add_floor: bool = True,
        camera_distance: float = 3.0,
        camera_azimuth: float = 135.0,
        camera_elevation: float = -18.0,
        expected_qpos_size: int | None = None,
    ):
        if (xml_path is None) == (scene_spec is None):
            raise ValueError("Provide exactly one of xml_path or scene_spec")
        spec = scene_spec.copy() if scene_spec is not None else mujoco.MjSpec.from_file(str(xml_path))
        if add_floor:
            spec.worldbody.add_geom(
                name="inference_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0.0, 0.0, 0.0],
                size=[20.0, 20.0, 0.02],
                rgba=[0.45, 0.47, 0.50, 1.0],
                contype=0,
                conaffinity=0,
            )
        spec.worldbody.add_light(
            name="inference_key_light",
            pos=[0.0, -3.0, 4.0],
            dir=[0.2, 0.5, -1.0],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.45, 0.45, 0.45],
            ambient=[0.15, 0.15, 0.15],
            specular=[0.05, 0.05, 0.05],
        )
        self.model = spec.compile()
        _style_untextured_terrain(self.model)
        self._absolute_near_clip = _limit_absolute_near_clip(self.model)
        self._qpos_mapping: list[tuple[slice, slice]] | None = None
        self._input_nq = int(self.model.nq)
        self._camera_input_qpos_adr = 0
        self._target_root_body_id: int | None = None
        self._target_root_qpos_adr: int | None = None
        if source_xml_path is not None:
            source_model = mujoco.MjModel.from_xml_path(str(source_xml_path))
            self._input_nq = int(source_model.nq)
            source_joints = {
                mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id): joint_id
                for joint_id in range(source_model.njnt)
            }
            mapping = []
            mapped_source_qpos = set()
            for target_joint_id in range(self.model.njnt):
                target_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, target_joint_id)
                source_name = target_name.split("/", 1)[-1] if target_name is not None else None
                if source_name not in source_joints:
                    continue
                source_joint_id = source_joints[source_name]
                source_adr = int(source_model.jnt_qposadr[source_joint_id])
                target_adr = int(self.model.jnt_qposadr[target_joint_id])
                joint_type = int(source_model.jnt_type[source_joint_id])
                width = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
                mapping.append((slice(source_adr, source_adr + width), slice(target_adr, target_adr + width)))
                mapped_source_qpos.update(range(source_adr, source_adr + width))
                if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                    self._camera_input_qpos_adr = source_adr
                    self._target_root_body_id = int(self.model.jnt_bodyid[target_joint_id])
                    self._target_root_qpos_adr = target_adr
            if mapped_source_qpos != set(range(source_model.nq)):
                missing = sorted(set(range(source_model.nq)) - mapped_source_qpos)
                raise ValueError(f"Scene renderer could not map all source qpos addresses: missing={missing}")
            self._qpos_mapping = mapping
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), int(render_size))
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), int(render_size))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self.scene_option = _inference_scene_option()
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = float(camera_distance)
        self.camera.azimuth = float(camera_azimuth)
        self.camera.elevation = float(camera_elevation)
        if expected_qpos_size is not None and self._input_nq != int(expected_qpos_size):
            raise ValueError(f"Expected renderer input nq={expected_qpos_size}, got nq={self._input_nq}")

    @property
    def input_nq(self) -> int:
        return self._input_nq

    def render_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != self._input_nq:
            raise ValueError(f"Expected qpos size {self._input_nq}, got {qpos.size}")
        if self._qpos_mapping is None:
            self.data.qpos[:] = qpos
        else:
            self.data.qpos[:] = self.model.qpos0
            for source_slice, target_slice in self._qpos_mapping:
                self.data.qpos[target_slice] = qpos[source_slice]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self._target_root_body_id is not None:
            root_pos = self.data.xpos[self._target_root_body_id]
        else:
            root_adr = self._camera_input_qpos_adr
            root_pos = qpos[root_adr : root_adr + 3]
        self.camera.lookat[:] = _camera_lookat_from_root(root_pos)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return to_rgb_uint8(self.renderer.render())

    def render_debug_state(self) -> dict[str, Any]:
        root_body_pos = None
        root_qpos = None
        robot_geom_bounds = None
        if self._target_root_body_id is not None:
            root_body_pos = self.data.xpos[self._target_root_body_id].tolist()
            descendants = np.zeros(self.model.nbody, dtype=bool)
            descendants[self._target_root_body_id] = True
            for body_id in range(self._target_root_body_id + 1, self.model.nbody):
                descendants[body_id] = descendants[int(self.model.body_parentid[body_id])]
            robot_geom_mask = descendants[self.model.geom_bodyid]
            robot_geom_pos = self.data.geom_xpos[robot_geom_mask]
            if robot_geom_pos.size:
                robot_geom_bounds = {
                    "min": robot_geom_pos.min(axis=0).tolist(),
                    "max": robot_geom_pos.max(axis=0).tolist(),
                }
        if self._target_root_qpos_adr is not None:
            root_qpos = self.data.qpos[self._target_root_qpos_adr : self._target_root_qpos_adr + 7].tolist()
        scene_cameras = []
        for camera in self.renderer.scene.camera:
            scene_cameras.append(
                {
                    "pos": camera.pos.tolist(),
                    "forward": camera.forward.tolist(),
                    "up": camera.up.tolist(),
                    "frustum_top": float(camera.frustum_top),
                    "frustum_bottom": float(camera.frustum_bottom),
                }
            )
        return {
            "camera_lookat": self.camera.lookat.tolist(),
            "camera_distance": float(self.camera.distance),
            "root_body_pos": root_body_pos,
            "root_qpos": root_qpos,
            "robot_geom_bounds": robot_geom_bounds,
            "model_center": self.model.stat.center.tolist(),
            "model_extent": float(self.model.stat.extent),
            "absolute_near_clip": self._absolute_near_clip,
            "scene_cameras": scene_cameras,
        }

    def close(self) -> None:
        self.renderer.close()


def policy_qpos_from_env(wrapped_env: Any, *, expected_qpos_size: int) -> np.ndarray:
    qpos, _qvel = wrapped_env._get_qpos_qvel(to_numpy=True)
    qpos = np.asarray(qpos)
    if qpos.ndim == 2:
        qpos = qpos[0]
    qpos = qpos.reshape(-1)
    if qpos.size != int(expected_qpos_size):
        raise ValueError(f"Expected MJLab policy qpos size {expected_qpos_size}, got shape {qpos.shape}")
    return qpos


def render_policy_frame(
    wrapped_env: Any,
    renderer: MujocoQposRenderer,
    *,
    use_env_render: bool,
) -> tuple[np.ndarray, bool]:
    if use_env_render:
        try:
            return to_rgb_uint8(wrapped_env.render()), True
        except ValueError as exc:
            print(
                "[INFO] wrapped_env.render() did not return an RGB frame; "
                f"falling back to MJLab qpos rendering for policy frames ({exc})."
            )
    return renderer.render_qpos(policy_qpos_from_env(wrapped_env, expected_qpos_size=renderer.input_nq)), False
