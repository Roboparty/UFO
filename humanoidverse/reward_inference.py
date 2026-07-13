"""Reward inference for the MJLab backend.

This entrypoint avoids the legacy Isaac inference environment. Reward relabeling
uses the selected robot XML, G1 keeps the full default task set, and non-G1
robots currently support root/locomotion reward tasks only.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import re
import time
from pathlib import Path

import joblib
import mediapy as media
import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.mjlab_reward_relabel import RewardWrapperHV
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    add_robot_config_manifest_args,
    add_bool_arg,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    render_policy_frame,
    resolve_inference_data_and_robot_args,
    resolve_inference_robot_config,
    write_mjlab_relabel_xml,
    write_g1_mjlab_relabel_xml,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec


DEFAULT_TASKS = [
    "move-ego-0-0",
    "move-ego-low0.5-0-0",
    "move-ego-0-0.7",
    "move-ego-0-0.3",
    "move-ego-90-0.3",
    "move-ego-180-0.3",
    "move-ego--90-0.3",
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
    "raisearms-l-l",
    "raisearms-l-m",
    "raisearms-m-l",
    "raisearms-m-m",
    "move-arms-0-0.7-m-m",
    "move-arms-90-0.7-m-m",
    "move-arms-180-0.4-m-m",
    "move-arms--90-0.7-m-m",
    "move-arms-0-0.7-l-m",
    "move-arms-90-0.7-l-m",
    "move-arms-180-0.4-l-m",
    "move-arms--90-0.7-l-m",
    "move-arms-0-0.7-m-l",
    "move-arms-90-0.7-m-l",
    "move-arms-180-0.4-m-l",
    "move-arms--90-0.7-m-l",
    "move-arms-0-0.7-l-l",
    "move-arms-90-0.7-l-l",
    "move-arms-180-0.4-l-l",
    "move-arms--90-0.7-l-l",
    "spin-arms-5-l-l",
    "spin-arms--5-l-l",
    "spin-arms-5-l-m",
    "spin-arms--5-l-m",
    "spin-arms-5-m-l",
    "spin-arms--5-m-l",
    "crouch-0",
    "crouch-0.25",
    "sitonground",
]

NON_G1_LOCOMOTION_TASKS = [task for task in DEFAULT_TASKS if task.startswith("move-ego-") or task.startswith("rotate-z-")]


def _is_g1_reward_robot(robot_training) -> bool:
    return robot_training.robot.name == "g1_29dof" and len(robot_training.robot.control_joint_names) == 29


def _is_non_g1_locomotion_task(task: str) -> bool:
    patterns = (
        r"^move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)$",
        r"^move-ego-low(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$",
        r"^rotate-z-(-?\d+\.*\d*)-(\d+\.*\d*)$",
    )
    return any(re.search(pattern, task) for pattern in patterns)


def _resolve_reward_tasks(tasks: list[str] | None, robot_training) -> tuple[list[str], str]:
    if _is_g1_reward_robot(robot_training):
        return list(tasks or DEFAULT_TASKS), "G1 full tasks"

    selected_tasks = list(tasks or NON_G1_LOCOMOTION_TASKS)
    for task in selected_tasks:
        if not _is_non_g1_locomotion_task(task):
            raise ValueError(
                f"Task {task} requires G1-style arm/body semantics and is not enabled for robot {robot_training.robot.name}."
            )
    return selected_tasks, "non-G1 locomotion-only tasks"


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


def _default_standing_target_states(wrapped_env, device: str) -> dict[str, torch.Tensor]:
    core_env = wrapped_env._env
    num_envs = int(core_env.num_envs)
    env_device = core_env.device

    init_state = core_env.config.robot.init_state
    root_pos = torch.as_tensor(init_state.pos, device=env_device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    if hasattr(core_env, "env_origins"):
        root_pos = root_pos + core_env.env_origins.to(device=env_device, dtype=torch.float32)
    root_rot_xyzw = torch.as_tensor(init_state.rot, device=env_device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    root_lin_vel = torch.zeros((num_envs, 3), device=env_device, dtype=torch.float32)
    root_ang_vel = torch.zeros((num_envs, 3), device=env_device, dtype=torch.float32)
    root_state_xyzw = torch.cat([root_pos, root_rot_xyzw, root_lin_vel, root_ang_vel], dim=-1)

    dof_state = torch.zeros((num_envs, core_env.num_dof, 2), device=env_device, dtype=torch.float32)
    dof_state[..., 0] = core_env.default_dof_pos.to(device=env_device, dtype=torch.float32)
    return {
        "root_states": root_state_xyzw.to(device=device, dtype=torch.float32),
        "dof_states": dof_state.to(device=device, dtype=torch.float32),
    }


def _load_replay_buffer(
    model_folder: Path,
    *,
    buffer_rank: int,
    buffer_path: Path | None,
) -> tuple[object, Path]:
    if buffer_path is not None:
        buffer_path = buffer_path.expanduser().resolve()
        if not buffer_path.is_dir():
            raise FileNotFoundError(f"Missing replay buffer path: {buffer_path}")
    else:
        buffers_dir = model_folder / "checkpoint" / "buffers"
        reduced = buffers_dir / "train_reduced"
        old_single_rank = buffers_dir / "train"
        rank_shard = buffers_dir / f"train_rank_{buffer_rank}"
        if reduced.is_dir():
            buffer_path = reduced
        elif rank_shard.is_dir():
            buffer_path = rank_shard
        elif old_single_rank.is_dir():
            buffer_path = old_single_rank
        else:
            raise FileNotFoundError(
                "Could not find replay buffer. Tried "
                f"{reduced}, {rank_shard}, and {old_single_rank}."
            )

    config_path = buffer_path / "config.json"
    if config_path.exists() and "TrajectoryDictBufferMultiDim" in config_path.read_text():
        dataset = TrajectoryDictBufferMultiDim.load(buffer_path, device="cpu")
    else:
        dataset = DictBuffer.load(buffer_path, device="cpu")
    return dataset, buffer_path


def run_reward_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    robot_config: Path | None,
    headless: bool,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    episode_length: int,
    num_samples: int,
    n_inferences: int,
    skip_rollouts: bool,
    tasks: list[str],
    buffer_rank: int,
    buffer_path: Path | None,
    max_workers: int,
    process_executor: bool,
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

    robot_config = resolve_inference_robot_config(robot_config, None)
    robot_training = load_robot_training_spec(robot_config)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    if not robot_xml.exists():
        raise FileNotFoundError(f"Missing robot XML: {robot_xml}")
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    is_g1 = _is_g1_reward_robot(robot_training)
    tasks, task_support_mode = _resolve_reward_tasks(tasks, robot_training)

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()

    if export_onnx:
        _export_model(model, model_folder / "exported")

    print("[INFO] Loading replay buffer...", end=" ", flush=True)
    start_t = time.time()
    dataset, loaded_buffer_path = _load_replay_buffer(model_folder, buffer_rank=buffer_rank, buffer_path=buffer_path)
    print(f"done in {time.time() - start_t:.2f}s")
    print(f"[INFO] Replay buffer={loaded_buffer_path}")
    if hasattr(dataset, "size"):
        print(f"[INFO] Replay buffer sampled transition count={dataset.size()}")

    output_dir = model_folder / "reward_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    if is_g1:
        relabel_xml = write_g1_mjlab_relabel_xml(robot_xml, output_dir)
    else:
        relabel_xml = write_mjlab_relabel_xml(
            robot_xml,
            output_dir,
            control_joint_names,
            robot_training.robot.name,
            root_body_name=robot_training.robot.base_body,
        )

    reward_eval_agent = RewardWrapperHV(
        model=model,
        inference_dataset=dataset,
        num_samples_per_inference=int(num_samples),
        inference_function="reward_wr_inference",
        max_workers=int(max_workers),
        process_executor=bool(process_executor),
        env_model=str(relabel_xml),
    )

    print(f"[INFO] UFO reward inference model_folder={model_folder}")
    print(f"[INFO] Robot={robot_training.robot.name}")
    print(f"[INFO] Robot config={Path(robot_training.config_path).expanduser().resolve()}")
    print(f"[INFO] num_dof={num_dof}")
    print(f"[INFO] Reward source XML={robot_xml}")
    print(f"[INFO] Reward relabel XML={relabel_xml}")
    print(f"[INFO] task support mode={task_support_mode}")
    print(f"[INFO] device={device} save_mp4={save_mp4} skip_rollouts={skip_rollouts}")
    print(f"[INFO] tasks={tasks}")

    z_dict: dict[str, list[torch.Tensor]] = {}
    output_path = output_dir / "reward_locomotion.pkl"
    for inference_idx in range(int(n_inferences)):
        for task in tasks:
            print(f"[INFO] Started reward inference {inference_idx + 1}/{n_inferences} for {task}...", end=" ", flush=True)
            start_t = time.time()
            z = reward_eval_agent.reward_inference(task=task)
            z_dict.setdefault(task, []).append(z.detach().cpu())
            print(f"done in {time.time() - start_t:.2f}s")
            joblib.dump(z_dict, output_path)
            print(f"[INFO] Saved reward embeddings: {output_path}")

    if skip_rollouts:
        return

    env_cfg, _use_root_height_obs = load_mjlab_env_cfg(
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
    renderer = None
    try:
        print(f"[INFO] Generating rollout videos with XML={env_cfg.mjcf_path}")
        if save_mp4:
            renderer = MujocoQposRenderer(
                robot_xml,
                render_size=render_size,
                camera_distance=camera_distance,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                expected_qpos_size=7 + num_dof,
            )
        for task in tasks:
            frames = []
            for z_cpu in z_dict[task]:
                z = z_cpu.to(device).repeat(1, 1)
                target_states = _default_standing_target_states(wrapped_env, device=device)
                observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)
                print("[INFO] Reset reward rollout to default standing pose.")
                use_env_render = True
                if save_mp4:
                    frame, use_env_render = render_policy_frame(wrapped_env, renderer, use_env_render=use_env_render)
                    frames.append(frame)
                for step in range(int(episode_length)):
                    action = model.act(observation, z, mean=True)
                    observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                    if save_mp4:
                        frame, use_env_render = render_policy_frame(wrapped_env, renderer, use_env_render=use_env_render)
                        frames.append(frame)
                    if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                        print(f"[INFO] Task {task} episode ended at step={step}; stopping this rollout.")
                        break
            if save_mp4:
                video_path = video_dir / f"{task}.mp4"
                media.write_video(str(video_path), frames, fps=fps)
                print(f"[INFO] Saved reward rollout video for {task}: {video_path}")
    finally:
        if renderer is not None:
            renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFO reward inference.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    add_robot_config_manifest_args(parser, purpose="reward inference")
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save policy rollout MP4s.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=150_000)
    parser.add_argument("--n-inferences", type=int, default=1)
    add_bool_arg(parser, "--skip-rollouts", False, "Only compute reward embeddings; do not create rollout videos.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task subset. Defaults to the full locomotion task list.")
    parser.add_argument("--buffer-rank", type=int, default=0, help="Rank-local replay buffer shard to use, e.g. train_rank_0.")
    parser.add_argument("--buffer-path", type=Path, default=None, help="Explicit replay buffer directory; overrides --buffer-rank.")
    parser.add_argument("--max-workers", type=int, default=24)
    add_bool_arg(parser, "--process-executor", True, "Use ProcessPoolExecutor for reward relabel workers.")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    add_bool_arg(parser, "--export-onnx", False, "Export ONNX next to the checkpoint before inference.")
    return resolve_inference_data_and_robot_args(parser.parse_args(), parser)


def main() -> None:
    args = parse_args()
    run_reward_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,
        headless=args.headless,
        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        episode_length=args.episode_length,
        num_samples=args.num_samples,
        n_inferences=args.n_inferences,
        skip_rollouts=args.skip_rollouts,
        tasks=args.tasks,
        buffer_rank=args.buffer_rank,
        buffer_path=args.buffer_path,
        max_workers=args.max_workers,
        process_executor=args.process_executor,
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
