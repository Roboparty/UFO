"""Goal inference and video export for UFO policies.

This entrypoint avoids the legacy Isaac inference path.  Goal embeddings are
computed from MJLab motion observations, and optional videos are rendered from
MJLab rollout state with pure MuJoCo qpos fallback.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import joblib
import mediapy as media
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humanoidverse.agents.envs.humanoidverse_mjlab import G1_MJLAB_MJCF_PATH
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, get_backward_observation
from humanoidverse.mjlab_inference_utils import (
    HUMANOIDVERSE_DIR,
    add_bool_arg,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    render_policy_frame,
    resolve_project_path,
    MujocoQposRenderer,
)


def _find_goal_json(goal_json: Path | None) -> Path:
    if goal_json is not None:
        goal_json = goal_json.expanduser().resolve()
        if not goal_json.exists():
            raise FileNotFoundError(f"Missing goal JSON: {goal_json}")
        return goal_json

    candidates = [
        HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "goal_frames_lafan29dof.json",
        HUMANOIDVERSE_DIR / "data" / "goal_frames_lafan29dof.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find goal_frames_lafan29dof.json. Searched in: {candidates}")


def _export_model(model: torch.nn.Module, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.__class__.__name__
    export_meta_policy_as_onnx(
        model,
        output_dir,
        f"{model_name}.onnx",
        z_dim=model.cfg.archi.z_dim,
    )
    print(f"[INFO] Exported model to {output_dir / f'{model_name}.onnx'}")


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


def run_goal_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    goal_json: Path | None,
    headless: bool,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    episode_len: int,
    goal_switch_interval: int,
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    fps: int,
    max_episode_length_s: float,
    export_onnx: bool,
) -> None:
    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    G1_xml = resolve_project_path(G1_MJLAB_MJCF_PATH)
    if not G1_xml.exists():
        raise FileNotFoundError(f"Missing G1 XML: {G1_xml}")

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
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

    output_dir = model_folder / "goal_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] UFO goal inference model_folder={model_folder}")
    print(f"[INFO] Rollout XML={env_cfg.mjcf_path}")
    print(f"[INFO] Motion data={env_cfg.lafan_tail_path}")
    print(f"[INFO] Goal JSON={_find_goal_json(goal_json)}")
    print(f"[INFO] device={device} disable_dr={disable_dr} disable_obs_noise={disable_obs_noise} save_mp4={save_mp4}")

    try:
        if export_onnx:
            _export_model(model, model_folder / "exported")

        with _find_goal_json(goal_json).open("r") as f:
            goals_to_evaluate = json.load(f)
        if not goals_to_evaluate:
            raise RuntimeError("Goal JSON is empty.")
        first_goal = goals_to_evaluate[0]

        z_dict: dict[str, object] = {}
        with torch.no_grad():
            pbar = tqdm(goals_to_evaluate, leave=False, disable=False)
            for goal in pbar:
                motion_id = int(goal["motion_id"])
                env.set_is_evaluating(motion_id)
                gobs, _gobs_dict = get_backward_observation(
                    env,
                    0,
                    use_root_height_obs=use_root_height_obs,
                    velocity_multiplier=0,
                )
                num_frames = next(iter(gobs.values())).shape[0]
                frame_pbar = tqdm(goal["frames"], leave=False, disable=False, desc="frames")
                for frame_idx in frame_pbar:
                    frame_idx = int(frame_idx)
                    if frame_idx >= num_frames:
                        pbar.write(f"  Skipping frame_idx {frame_idx} (motion has {num_frames} frames)")
                        continue
                    goal_name = f"{goal['motion_name']}_{frame_idx}"
                    goal_observation = {key: value[frame_idx][None, ...] for key, value in gobs.items()}
                    goal_observation = tree_map(
                        lambda x: torch.as_tensor(x, device=device, dtype=torch.float32),
                        goal_observation,
                    )
                    z_dict[goal_name] = model.goal_inference(goal_observation).detach().cpu().numpy()

        output_path = output_dir / "goal_reaching.pkl"
        joblib.dump(z_dict, output_path)
        print(f"[INFO] Saved goal embeddings: {output_path} ({len(z_dict)} goals)")
        if not z_dict:
            raise RuntimeError("No goal embeddings were generated.")

        if not save_mp4:
            return

        renderer = MujocoQposRenderer(
            G1_xml,
            render_size=render_size,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
        )
        try:
            first_motion_id = int(first_goal["motion_id"])
            env.set_is_evaluating(first_motion_id)
            _first_backward_obs, first_obs_dict = get_backward_observation(
                env,
                0,
                use_root_height_obs=use_root_height_obs,
                velocity_multiplier=0,
            )
            target_states = _target_states_from_obs(first_obs_dict, device=device)
            observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)
            first_motion_name = first_goal.get("motion_name", "unknown")
            print(
                f"[INFO] Reset goal rollout to demo start: "
                f"motion_id={first_motion_id}, motion_name={first_motion_name}, frame=0"
            )
            frames = []
            goal_idx = -1
            goal_names = list(z_dict.keys())
            z = None
            use_env_render = True
            for step in tqdm(range(int(episode_len)), desc="steps", leave=False):
                if step % int(goal_switch_interval) == 0:
                    goal_idx = (goal_idx + 1) % len(goal_names)
                    print(f"[INFO] Switching to goal {goal_names[goal_idx]} at step {step}")
                    z = torch.as_tensor(z_dict[goal_names[goal_idx]], device=device, dtype=torch.float32)

                action = model.act(observation, z.repeat(1, 1), mean=True)
                observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                frame, use_env_render = render_policy_frame(wrapped_env, renderer, use_env_render=use_env_render)
                frames.append(frame)

                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    print(f"[INFO] Episode ended at step={step}; stopping goal rollout video.")
                    break

            video_path = video_dir / "goal.mp4"
            media.write_video(str(video_path), frames, fps=fps)
            print(f"[INFO] Saved goal video: {video_path}")
        finally:
            renderer.close()
    finally:
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFO goal inference.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--goal-json", type=Path, default=None)
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save policy rollout MP4.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--episode-len", type=int, default=2100)
    parser.add_argument("--goal-switch-interval", type=int, default=100)
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    add_bool_arg(parser, "--export-onnx", False, "Export ONNX next to the checkpoint before inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_goal_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        goal_json=args.goal_json,
        headless=args.headless,
        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        episode_len=args.episode_len,
        goal_switch_interval=args.goal_switch_interval,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        fps=args.fps,
        max_episode_length_s=args.max_episode_length_s,
        export_onnx=args.export_onnx,
    )


if __name__ == "__main__":
    main()
