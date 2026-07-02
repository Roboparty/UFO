"""Tracking inference and video export for the MJLab backend.

This entrypoint intentionally avoids the legacy Isaac inference environment.
Policy rollout is rendered from the MJLab environment used for training, while
the reference motion is rendered from the same local MJLab G1 MJCF with pure
MuJoCo qpos playback.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import mediapy as media
import mujoco
import numpy as np
import torch
from torch.utils._pytree import tree_map

import humanoidverse
from humanoidverse.agents.envs.humanoidverse_mjlab import (
    HumanoidVerseMjlabConfig,
    G1_MJLAB_MJCF_PATH,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import get_backward_observation


if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = HUMANOIDVERSE_DIR.parent
DEFAULT_INFERENCE_DATA_PATH = Path("humanoidverse/data/lafan_29dof_10s-clipped.pkl")


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        name,
        nargs="?",
        const=True,
        default=default,
        type=_str2bool,
        help=help_text,
    )


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _checkpoint_load_device(device: str) -> str:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        return "cuda"
    if torch_device.type == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported inference device: {device}")


def _replace_hydra_override(overrides: list[str], key: str, value: Any) -> list[str]:
    prefix = f"{key}="
    return [item for item in overrides if not item.startswith(prefix)] + [f"{key}={value}"]


def _load_env_cfg(
    model_folder: Path,
    *,
    data_path: Path | None,
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
    overrides = _replace_hydra_override(overrides, "env.config.max_episode_length_s", max_episode_length_s)
    overrides = _replace_hydra_override(overrides, "env.config.headless", str(headless))
    env_config["hydra_overrides"] = overrides

    return HumanoidVerseMjlabConfig(**env_config), use_root_height_obs


def _to_rgb_uint8(frame: Any) -> np.ndarray:
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


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    y_idx = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
    return frame[y_idx[:, None], x_idx[None, :]]


class MujocoQposRenderer:
    """Pure MuJoCo renderer for 36-D qpos from the MJLab G1 MJCF."""

    def __init__(
        self,
        xml_path: Path,
        render_size: int = 512,
        *,
        camera_distance: float = 3.0,
        camera_azimuth: float = 135.0,
        camera_elevation: float = -18.0,
    ):
        spec = mujoco.MjSpec.from_file(str(xml_path))
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
            diffuse=[0.8, 0.8, 0.8],
            ambient=[0.35, 0.35, 0.35],
            specular=[0.1, 0.1, 0.1],
        )
        self.model = spec.compile()
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), int(render_size))
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), int(render_size))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = float(camera_distance)
        self.camera.azimuth = float(camera_azimuth)
        self.camera.elevation = float(camera_elevation)
        if self.model.nq != 36:
            raise ValueError(f"Expected G1 nq=36, got nq={self.model.nq}")

    def render_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != self.model.nq:
            raise ValueError(f"Expected qpos size {self.model.nq}, got {qpos.size}")
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = [float(qpos[0]), float(qpos[1]), max(float(qpos[2]), 0.75)]
        self.renderer.update_scene(self.data, camera=self.camera)
        return _to_rgb_uint8(self.renderer.render())

    def close(self) -> None:
        self.renderer.close()


def _expert_qpos_from_obs(obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
    root_pos = obs_dict["ref_body_pos"][:, 0].detach().cpu().numpy()
    root_quat_wxyz = np.roll(obs_dict["ref_body_rots"][:, 0].detach().cpu().numpy(), 1, axis=-1)
    dof_pos = obs_dict["dof_pos"].detach().cpu().numpy()
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=-1)
    if qpos.shape[-1] != 36:
        raise ValueError(f"Expected expert qpos shape (*, 36), got {qpos.shape}")
    return qpos


def _target_states_from_obs(obs_dict: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    root_state_xyzw = torch.cat(
        [
            obs_dict["ref_body_pos"][0, 0],
            obs_dict["ref_body_rots"][0, 0],
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ],
        dim=-1,
    ).to(device=device, dtype=torch.float32)
    dof_state = torch.zeros((29, 2), device=device, dtype=torch.float32)
    dof_state[:, 0] = obs_dict["dof_pos"][0].to(device=device, dtype=torch.float32)
    dof_state[:, 1] = obs_dict["ref_dof_vel"][0].to(device=device, dtype=torch.float32)
    return {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}


def _policy_qpos_from_env(wrapped_env: Any) -> np.ndarray:
    qpos, _qvel = wrapped_env._get_qpos_qvel(to_numpy=True)
    qpos = np.asarray(qpos)
    if qpos.ndim == 2:
        qpos = qpos[0]
    qpos = qpos.reshape(-1)
    if qpos.size != 36:
        raise ValueError(f"Expected MJLab policy qpos size 36, got shape {qpos.shape}")
    return qpos


@torch.no_grad()
def _tracking_z(model: torch.nn.Module, obs: Any) -> torch.Tensor:
    z = model.backward_map(obs)
    for step in range(z.shape[0]):
        z[step] = z[step : step + 1].mean(dim=0)
    return model.project_z(z)


def run_tracking_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    headless: bool,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    motion_list: list[int],
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    fps: int,
    max_steps: int | None,
    log_every_steps: int,
    max_episode_length_s: float,
) -> None:
    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    G1_xml = _resolve_project_path(G1_MJLAB_MJCF_PATH)
    if not G1_xml.exists():
        raise FileNotFoundError(f"Missing MJLab G1 XML: {G1_xml}")

    model_load_device = _checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()

    env_cfg, use_root_height_obs = _load_env_cfg(
        model_folder,
        data_path=data_path,
        device=device,
        headless=headless,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=max_episode_length_s,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env

    output_dir = model_folder / "tracking_inference_mjlab"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] MJLab tracking inference model_folder={model_folder}")
    print(f"[INFO] MJLab rollout XML={env_cfg.mjcf_path}")
    print(f"[INFO] Motion data={env_cfg.lafan_tail_path}")
    print(f"[INFO] Expert renderer XML={G1_xml}")
    print(f"[INFO] device={device} disable_dr={disable_dr} disable_obs_noise={disable_obs_noise} save_mp4={save_mp4}")

    env._motion_lib.load_all_motions()
    env.is_evaluating = True
    expert_renderer = (
        MujocoQposRenderer(
            G1_xml,
            render_size=render_size,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
        )
        if save_mp4
        else None
    )
    try:
        for motion_id in motion_list:
            backward_obs, obs_dict = get_backward_observation(env, motion_id, use_root_height_obs=use_root_height_obs)
            z = _tracking_z(model, tree_map(lambda x: x[1:].to(device) if hasattr(x, "to") else x, backward_obs))
            joblib.dump(z.detach().cpu().numpy(), output_dir / f"zs_{motion_id}.pkl")
            print(f"[INFO] Saved z embedding: {output_dir / f'zs_{motion_id}.pkl'}")

            target_states = _target_states_from_obs(obs_dict, device=device)
            observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)
            episode_len = int(z.shape[0])
            if max_steps is not None:
                episode_len = min(episode_len, int(max_steps))
            expert_qpos = _expert_qpos_from_obs(obs_dict)
            frames: list[np.ndarray] = []
            use_env_render = True

            print(f"[INFO] Running MJLab policy rollout for motion_id={motion_id}, steps={episode_len}", flush=True)
            for step in range(episode_len):
                action = model.act(observation, z[step].unsqueeze(0), mean=True)
                observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)

                if save_mp4:
                    if use_env_render:
                        try:
                            policy_frame = _to_rgb_uint8(wrapped_env.render())
                        except ValueError as exc:
                            print(
                                "[INFO] wrapped_env.render() did not return an RGB frame; "
                                f"falling back to MJLab qpos rendering for policy frames ({exc})."
                            )
                            use_env_render = False
                            policy_frame = expert_renderer.render_qpos(_policy_qpos_from_env(wrapped_env))
                    else:
                        policy_frame = expert_renderer.render_qpos(_policy_qpos_from_env(wrapped_env))
                    expert_frame = expert_renderer.render_qpos(expert_qpos[min(step + 1, len(expert_qpos) - 1)])
                    expert_frame = _resize_nearest(expert_frame, policy_frame.shape[0], policy_frame.shape[1])
                    frames.append(np.concatenate([expert_frame, policy_frame], axis=1))

                if step == 0 or (step + 1) == episode_len or (log_every_steps > 0 and (step + 1) % log_every_steps == 0):
                    print(f"[INFO] motion_id={motion_id} rollout/render progress {step + 1}/{episode_len}", flush=True)

                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    print(f"[INFO] Episode ended at step={step}; stopping rollout for motion_id={motion_id}")
                    break

            if save_mp4:
                video_path = output_dir / f"tracking_mjlab_{motion_id}.mp4"
                if not frames:
                    raise RuntimeError(f"No frames were rendered for motion_id={motion_id}")
                media.write_video(str(video_path), frames, fps=fps)
                print(f"[INFO] Saved side-by-side video: {video_path}")
    finally:
        if expert_renderer is not None:
            expert_renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MJLab tracking inference with MuJoCo expert rendering.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    _bool_arg(parser, "--headless", True, "Run MuJoCo/MJLab in headless mode.")
    parser.add_argument("--device", default="cuda:0")
    _bool_arg(parser, "--save-mp4", False, "Save side-by-side expert/policy MP4.")
    _bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    _bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--motion-list", type=int, nargs="+", default=[20])
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rollout/video frames for quick previews.")
    parser.add_argument("--log-every-steps", type=int, default=100, help="Print rollout/render progress every N steps; 0 disables periodic logs.")
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tracking_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        headless=args.headless,
        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        motion_list=args.motion_list,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        fps=args.fps,
        max_steps=args.max_steps,
        log_every_steps=args.log_every_steps,
        max_episode_length_s=args.max_episode_length_s,
    )


if __name__ == "__main__":
    main()
