"""Tracking inference and video export for UFO policies.

Policy rollout is rendered from the training environment, while the reference
motion is rendered from the configured robot MJCF with pure MuJoCo qpos playback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import mediapy as media
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    add_bool_arg,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    render_policy_frame,
    resolve_project_path,
)
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, get_backward_observation
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    y_idx = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
    return frame[y_idx[:, None], x_idx[None, :]]


def _expert_qpos_from_obs(obs_dict: dict[str, torch.Tensor], *, num_dof: int) -> np.ndarray:
    root_pos = obs_dict["ref_body_pos"][:, 0].detach().cpu().numpy()
    root_quat_wxyz = np.roll(obs_dict["ref_body_rots"][:, 0].detach().cpu().numpy(), 1, axis=-1)
    dof_pos = obs_dict["dof_pos"].detach().cpu().numpy()
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=-1)
    expected = 7 + int(num_dof)
    if qpos.shape[-1] != expected:
        raise ValueError(f"Expected expert qpos shape (*, {expected}), got {qpos.shape}")
    return qpos


def _target_states_from_obs(obs_dict: dict[str, torch.Tensor], device: str, *, num_dof: int) -> dict[str, torch.Tensor]:
    root_state_xyzw = torch.cat(
        [
            obs_dict["ref_body_pos"][0, 0],
            obs_dict["ref_body_rots"][0, 0],
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ],
        dim=-1,
    ).to(device=device, dtype=torch.float32)
    dof_state = torch.zeros((int(num_dof), 2), device=device, dtype=torch.float32)
    dof_state[:, 0] = obs_dict["dof_pos"][0].to(device=device, dtype=torch.float32)
    dof_state[:, 1] = obs_dict["ref_dof_vel"][0].to(device=device, dtype=torch.float32)
    return {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}


@torch.no_grad()
def _tracking_z(model: torch.nn.Module, obs: Any) -> torch.Tensor:
    z = model.backward_map(obs)
    for step in range(z.shape[0]):
        z[step] = z[step : step + 1].mean(dim=0)
    return model.project_z(z)


def _export_model(model: torch.nn.Module, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.__class__.__name__
    output_name = f"{model_name}.onnx"
    export_meta_policy_as_onnx(
        model,
        output_dir,
        output_name,
        {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + model.cfg.archi.z_dim)},
        z_dim=model.cfg.archi.z_dim,
        history=("history_actor" in model.cfg.archi.actor.input_filter.key),
        use_29dof=True,
    )
    print(f"[INFO] Exported model to {output_dir / output_name}")


def run_tracking_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    robot_config: Path | None,
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
    export_onnx: bool,
) -> None:
    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    default_robot_config = resolve_project_path("configs/robots/g1_29dof.yaml")
    robot_training = load_robot_training_spec(robot_config or default_robot_config)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    if not robot_xml.exists():
        raise FileNotFoundError(f"Missing robot XML: {robot_xml}")
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()

    if export_onnx and num_dof != 29:
        raise ValueError("ONNX export currently supports only G1 29-DOF policies; rerun with --no-export-onnx for other robots.")
    if export_onnx:
        _export_model(model, model_folder / "exported")

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=device,
        headless=headless,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=max_episode_length_s,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env

    output_dir = model_folder / "tracking_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] UFO tracking inference model_folder={model_folder}")
    print(f"[INFO] Rollout XML={env_cfg.mjcf_path}")
    print(f"[INFO] Motion data={env_cfg.lafan_tail_path}")
    print(f"[INFO] Expert renderer XML={robot_xml}")
    print(f"[INFO] device={device} disable_dr={disable_dr} disable_obs_noise={disable_obs_noise} save_mp4={save_mp4}")

    env._motion_lib.load_all_motions()
    env.is_evaluating = True
    expert_renderer = (
        MujocoQposRenderer(
            robot_xml,
            render_size=render_size,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            expected_qpos_size=7 + num_dof,
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

            target_states = _target_states_from_obs(obs_dict, device=device, num_dof=num_dof)
            observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)
            episode_len = int(z.shape[0])
            if max_steps is not None:
                episode_len = min(episode_len, int(max_steps))
            expert_qpos = _expert_qpos_from_obs(obs_dict, num_dof=num_dof)
            frames: list[np.ndarray] = []
            use_env_render = True

            print(f"[INFO] Running policy rollout for motion_id={motion_id}, steps={episode_len}", flush=True)
            for step in range(episode_len):
                action = model.act(observation, z[step].unsqueeze(0), mean=True)
                observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)

                if save_mp4:
                    policy_frame, use_env_render = render_policy_frame(
                        wrapped_env,
                        expert_renderer,
                        use_env_render=use_env_render,
                    )
                    expert_frame = expert_renderer.render_qpos(expert_qpos[min(step + 1, len(expert_qpos) - 1)])
                    expert_frame = _resize_nearest(expert_frame, policy_frame.shape[0], policy_frame.shape[1])
                    frames.append(np.concatenate([expert_frame, policy_frame], axis=1))

                if step == 0 or (step + 1) == episode_len or (log_every_steps > 0 and (step + 1) % log_every_steps == 0):
                    print(f"[INFO] motion_id={motion_id} rollout/render progress {step + 1}/{episode_len}", flush=True)

                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    print(f"[INFO] Episode ended at step={step}; stopping rollout for motion_id={motion_id}")
                    break

            if save_mp4:
                video_path = output_dir / f"tracking_{motion_id}.mp4"
                if not frames:
                    raise RuntimeError(f"No frames were rendered for motion_id={motion_id}")
                media.write_video(str(video_path), frames, fps=fps)
                print(f"[INFO] Saved side-by-side video: {video_path}")
    finally:
        if expert_renderer is not None:
            expert_renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFO tracking inference with MuJoCo expert rendering.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=None, help="Robot YAML for rollout and expert rendering.")
    parser.add_argument("--data-manifest", type=Path, default=None, help="Motion data manifest. Use with --dataset.")
    parser.add_argument("--dataset", default=None, help="Dataset name inside --data-manifest for tracking inference.")
    parser.add_argument("--rebuild-motion-cache", action="store_true", help="Rebuild manifest-generated motion pkl cache.")
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save side-by-side expert/policy MP4.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--motion-list", type=int, nargs="+", default=[20])
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rollout/video frames for quick previews.")
    parser.add_argument("--log-every-steps", type=int, default=100, help="Print rollout/render progress every N steps; 0 disables periodic logs.")
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    add_bool_arg(parser, "--export-onnx", True, "Export ONNX next to the checkpoint before inference.")
    args = parser.parse_args()
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        if args.dataset is None:
            parser.error("--dataset is required when --data-manifest is provided")
        args.data_path = Path(
            prepare_manifest_dataset_path(
                args.data_manifest,
                args.dataset,
                split="inference",
                rebuild_cache=bool(args.rebuild_motion_cache),
            )
        )
    return args


def main() -> None:
    args = parse_args()
    run_tracking_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,
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
        export_onnx=args.export_onnx,
    )


if __name__ == "__main__":
    main()
