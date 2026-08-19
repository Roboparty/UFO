"""Same-z evaluation across physical flat, slope, stairs, and rough terrain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import mediapy as media
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.goal_inference import _find_goal_json, load_and_validate_goal_json
from humanoidverse.mjlab_inference_utils import (
    DEFAULT_INFERENCE_DATA_PATH,
    MujocoQposRenderer,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    policy_qpos_from_env,
    replace_hydra_override,
    resolve_inference_robot_config,
)
from humanoidverse.terrain_transfer import clone_same_z_for_terrains, tensor_checksum
from humanoidverse.tracking_inference import _target_states_from_obs, _tracking_z
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.robot_spec import load_robot_training_spec

SUPPORTED_TERRAINS = ("flat", "slope", "stairs", "rough", "course")


def _stairs_step_center_offset(step: int, *, platform_width: float, step_depth: float) -> float:
    """Return the local radial offset to the center of a one-indexed stair band."""
    if step < 0:
        raise ValueError("stairs start step must be non-negative")
    if step == 0:
        return 0.0
    if platform_width <= 0.0 or step_depth <= 0.0:
        raise ValueError("stairs platform width and step depth must be positive")
    return platform_width / 2.0 + (step - 0.5) * step_depth


def _course_completion_radius(course_cfg, *, final_flat_margin: float = 0.30) -> float:
    """Return the radius just inside the final flat annulus."""
    return (
        float(course_cfg.flat_run)
        + 2.0 * int(course_cfg.num_steps) * float(course_cfg.step_depth)
        + float(course_cfg.top_platform_length)
        + float(course_cfg.connector_length)
        + float(course_cfg.ramp_length)
        + final_flat_margin
    )


def _terrain_env_cfg(base_cfg, terrain: str, seed: int, *, dense_terrain: bool = False):
    overrides = list(base_cfg.hydra_overrides)
    overrides = replace_hydra_override(overrides, "terrain", "terrain_ufo_v0")
    overrides = replace_hydra_override(overrides, "terrain.terrain_type", terrain)
    overrides = replace_hydra_override(overrides, "terrain.seed", seed)
    if terrain == "course":
        overrides = replace_hydra_override(overrides, "terrain.num_rows", 1)
    if dense_terrain:
        # Evaluation-only presentation preset. The 30 m coverage invariant is
        # unchanged, but stairs span most of the patch instead of one small ring.
        overrides = replace_hydra_override(overrides, "terrain.stairs.num_steps", 20)
        overrides = replace_hydra_override(overrides, "terrain.stairs.step_depth", 0.30)
        overrides = replace_hydra_override(overrides, "terrain.stairs.platform_width", 1.5)
    return base_cfg.model_copy(update={"hydra_overrides": overrides, "seed": seed})


def _default_target_states(env) -> dict[str, torch.Tensor]:
    core = env._env
    init = core.config.robot.init_state
    root_pos = torch.as_tensor(init.pos, device=core.device, dtype=torch.float32).unsqueeze(0) + core.env_origins
    root_rot = torch.as_tensor(init.rot, device=core.device, dtype=torch.float32).unsqueeze(0)
    root_state = torch.cat((root_pos, root_rot, torch.zeros((1, 6), device=core.device)), dim=-1)
    dof_state = torch.zeros((1, core.num_dof, 2), device=core.device)
    dof_state[..., 0] = core.default_dof_pos
    return {"root_states": root_state, "dof_states": dof_state}


def _compute_goal_or_tracking_z(args, model, base_cfg):
    encoding_cfg = _terrain_env_cfg(base_cfg, "flat", args.seed, dense_terrain=args.dense_terrain)
    wrapped_env, _ = encoding_cfg.build(num_envs=1)
    env = wrapped_env._env
    try:
        env._motion_lib.load_all_motions()
        env.is_evaluating = True
        if args.prompt_type == "tracking":
            motion_id = int(args.motion_id)
            backward_obs, obs_dict = get_backward_observation(env, motion_id, use_root_height_obs=args.use_root_height_obs)
            z = _tracking_z(
                model,
                tree_map(lambda x: x[1:].to(args.device) if hasattr(x, "to") else x, backward_obs),
            )
            identifier = f"motion:{motion_id}"
        else:
            goal_path = _find_goal_json(
                args.goal_json,
                num_dof=env.num_dof,
                robot_name=args.robot_training.robot.name,
            )
            goals = load_and_validate_goal_json(goal_path, num_dof=env.num_dof)
            goal = goals[int(args.goal_index)]
            motion_id = int(goal["motion_id"])
            frame_idx = int(args.goal_frame if args.goal_frame is not None else goal["frames"][0])
            backward_obs, obs_dict = get_backward_observation(
                env,
                motion_id,
                use_root_height_obs=args.use_root_height_obs,
                velocity_multiplier=0,
            )
            goal_obs = {
                key: torch.as_tensor(value[frame_idx : frame_idx + 1], device=args.device, dtype=torch.float32)
                for key, value in backward_obs.items()
            }
            z = model.goal_inference(goal_obs)
            identifier = f"{goal.get('motion_name', motion_id)}:{frame_idx}"
        target_states = _target_states_from_obs(obs_dict, device=args.device, num_dof=env.num_dof)
        # Store the prompt pose in terrain-local coordinates. Each rollout adds
        # its own physical terrain origin, including the safe spawn height.
        target_states["root_states"] = target_states["root_states"].clone()
        target_states["root_states"][:, :3] -= env.env_origins[:1].to(args.device)
        target_states["root_states"][:, :2] = 0.0
        return z.detach(), identifier, target_states
    finally:
        wrapped_env.close()


def _compute_reward_z(args, model):
    from humanoidverse.mjlab_inference_utils import write_g1_mjlab_relabel_xml
    from humanoidverse.mjlab_reward_relabel import RewardWrapperHV
    from humanoidverse.reward_inference import _load_replay_buffer

    dataset, _ = _load_replay_buffer(args.model_folder, buffer_rank=args.buffer_rank, buffer_path=args.buffer_path)
    output_dir = args.output.parent / "terrain_transfer_relabel"
    relabel_xml = write_g1_mjlab_relabel_xml(Path(args.robot_training.robot.xml_path), output_dir)
    wrapper = RewardWrapperHV(
        model=model,
        inference_dataset=dataset,
        num_samples_per_inference=args.num_samples,
        inference_function="reward_wr_inference",
        max_workers=args.max_workers,
        process_executor=args.process_executor,
        env_model=str(relabel_xml),
    )
    return wrapper.reward_inference(task=args.reward_task).detach(), args.reward_task, None


def _prompt_value(model, observation, z: torch.Tensor) -> float:
    discriminator = getattr(model, "_discriminator", None)
    if discriminator is None:
        return float("nan")
    z_step = z if z.shape[0] == 1 else z[:1]
    value = discriminator.compute_reward(model._normalize(observation), z_step)
    return float(value.mean().item())


def _root_ground_clearance(env) -> float:
    sensor = env._env.mjlab_env.scene.sensors["terrain_height"]
    heights = sensor.data.heights
    if heights.ndim == 3:
        heights = heights[:, 0]
    reference_index = env._env._terrain_reference_index
    if reference_index is None:
        raise RuntimeError("terrain reference ray was not initialized")
    return float(heights[0, reference_index].item())


def _run_rollout(args, model, base_cfg, terrain: str, z: torch.Tensor, target_states) -> dict[str, Any]:
    env_cfg = _terrain_env_cfg(base_cfg, terrain, args.seed, dense_terrain=args.dense_terrain)
    wrapped_env, _ = env_cfg.build(num_envs=1)
    checksum = tensor_checksum(z)
    renderer = None
    try:
        if target_states is None:
            target_states = _default_target_states(wrapped_env)
        else:
            target_states = {key: value.clone() for key, value in target_states.items()}
            target_states["root_states"][:, :3] += wrapped_env._env.env_origins[:1].to(args.device)
        if terrain == "stairs" and args.stairs_start_step > 0:
            stairs_cfg = wrapped_env._env.config.terrain.stairs
            start_offset = _stairs_step_center_offset(
                args.stairs_start_step,
                platform_width=float(stairs_cfg.platform_width),
                step_depth=float(stairs_cfg.step_depth),
            )
            target_states["root_states"][:, 0] += start_offset
            print(
                f"[INFO] stairs start: step={args.stairs_start_step}, "
                f"local_xy=({start_offset:.3f}, 0.000)"
            )
        observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)
        initial_root = wrapped_env._env.robot_root_states[0, :3].clone()
        max_forward_displacement = 0.0
        max_planar_displacement = 0.0
        course_completion_radius = (
            _course_completion_radius(wrapped_env._env.config.terrain.course)
            if terrain == "course"
            else None
        )
        course_completed = False
        velocities: list[float] = []
        prompt_values: list[float] = []
        tracking_errors: list[float] = []
        frames = []
        if args.save_mp4:
            renderer = MujocoQposRenderer(
                None,
                render_size=args.render_size,
                scene_spec=wrapped_env._env.mjlab_env.scene.spec,
                source_xml_path=Path(args.robot_training.robot.xml_path),
                add_floor=False,
                camera_distance=args.camera_distance,
                camera_azimuth=args.camera_azimuth,
                camera_elevation=args.camera_elevation,
                expected_qpos_size=7 + wrapped_env._env.num_dof,
            )
        terminated_flag = False
        truncated_flag = False
        boundary_reset_flag = False
        steps = min(args.episode_length, int(z.shape[0])) if args.prompt_type == "tracking" else args.episode_length
        completed = 0
        for step in range(steps):
            z_step = z[step : step + 1] if z.shape[0] > 1 else z
            action = model.act(observation, z_step, mean=True)
            observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
            completed = step + 1
            forward_displacement = float(
                (wrapped_env._env.robot_root_states[0, 0] - initial_root[0]).item()
            )
            max_forward_displacement = max(max_forward_displacement, forward_displacement)
            planar_displacement = float(
                torch.linalg.vector_norm(wrapped_env._env.robot_root_states[0, :2] - initial_root[:2]).item()
            )
            max_planar_displacement = max(max_planar_displacement, planar_displacement)
            course_completed = bool(
                course_completion_radius is not None and planar_displacement >= course_completion_radius
            )
            velocities.append(float(torch.linalg.vector_norm(wrapped_env._env.robot_root_states[0, 7:9]).item()))
            prompt_values.append(_prompt_value(model, observation, z_step))
            if args.prompt_type == "tracking":
                tracking_errors.append(float(torch.linalg.vector_norm(wrapped_env._env.dif_global_body_pos, dim=-1).mean().item()))
            if renderer is not None:
                render_qpos = policy_qpos_from_env(wrapped_env, expected_qpos_size=renderer.input_nq)
                frames.append(renderer.render_qpos(render_qpos))
                if step == 0:
                    print(f"[INFO] terrain renderer state: {renderer.render_debug_state()}")
            terminated_flag = bool(torch.as_tensor(terminated).any().item())
            truncated_flag = bool(torch.as_tensor(truncated).any().item())
            boundary_reset_flag = bool(torch.as_tensor(_info.get("boundary_resets", False)).any().item())
            if course_completed or terminated_flag or truncated_flag:
                break
        final_root = wrapped_env._env.robot_root_states[0, :3].clone()
        final_ground_clearance = _root_ground_clearance(wrapped_env)
        final_goal_error = None
        if args.prompt_type == "goal":
            achieved_z = model.goal_inference(observation)
            final_goal_error = float(torch.linalg.vector_norm(achieved_z - z[:1], dim=-1).mean().item())
        fell = terminated_flag or final_ground_clearance < args.fall_clearance
        video_path = None
        if renderer is not None:
            video_path = args.output.with_suffix("").with_name(f"{args.output.stem}_{terrain}.mp4")
            if not frames:
                raise RuntimeError(f"No frames rendered for terrain={terrain}")
            media.write_video(str(video_path), frames, fps=args.fps)
            print(f"[INFO] wrote terrain video {video_path}")
        result = {
            "terrain_type": terrain,
            "seed": args.seed,
            "prompt_type": args.prompt_type,
            "prompt_identifier": args.prompt_identifier,
            "z_shape": list(z.shape),
            "z_checksum": checksum,
            "episode_length": completed,
            "terminated": terminated_flag,
            "truncated": truncated_flag,
            "boundary_reset": boundary_reset_flag,
            "fell": fell,
            "root_displacement": float(torch.linalg.vector_norm(final_root[:2] - initial_root[:2]).item()),
            "forward_displacement": float((final_root[0] - initial_root[0]).item()),
            "max_forward_displacement": max_forward_displacement,
            "max_planar_displacement": max_planar_displacement,
            "course_completion_radius": course_completion_radius,
            "course_completed": course_completed if terrain == "course" else None,
            "mean_root_velocity": sum(velocities) / max(len(velocities), 1),
            "final_root_height": float(final_root[2].item()),
            "final_ground_clearance": final_ground_clearance,
            "mean_prompt_value": sum(prompt_values) / max(len(prompt_values), 1),
            "final_goal_error": final_goal_error,
            "mean_tracking_error": sum(tracking_errors) / len(tracking_errors) if tracking_errors else None,
            "video_path": str(video_path) if video_path is not None else None,
        }
        return result
    finally:
        if renderer is not None:
            renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reuse one UFO z exactly across physical terrains.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--prompt-type", choices=["reward", "goal", "tracking"], required=True)
    parser.add_argument("--terrains", default="flat,slope,stairs,rough")
    parser.add_argument(
        "--dense-terrain",
        action="store_true",
        help="Evaluation-only terrain presentation preset with stairs spanning most of the 30 m patch.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_INFERENCE_DATA_PATH)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--fall-clearance", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=Path("terrain_transfer_results.json"))
    parser.add_argument("--reward-task", default="move-ego-0-0.7")
    parser.add_argument("--buffer-path", type=Path, default=None)
    parser.add_argument("--buffer-rank", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--process-executor", action="store_true")
    parser.add_argument("--goal-json", type=Path, default=None)
    parser.add_argument("--goal-index", type=int, default=0)
    parser.add_argument("--goal-frame", type=int, default=None)
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument(
        "--stairs-start-step",
        type=int,
        default=0,
        help="Start stairs rollout at the center of this one-indexed stair band; 0 keeps the center platform.",
    )
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_folder = args.model_folder.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.robot_config = resolve_inference_robot_config(args.robot_config, None)
    args.robot_training = load_robot_training_spec(args.robot_config)
    checkpoint_dir = args.model_folder / "checkpoint"
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(args.device))
    model.to(args.device).eval()
    base_cfg, args.use_root_height_obs = load_mjlab_env_cfg(
        args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,
        device=args.device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=max(10.0, args.episode_length / 50.0 + 1.0),
    )
    if args.prompt_type == "reward":
        z, identifier, target_states = _compute_reward_z(args, model)
    else:
        z, identifier, target_states = _compute_goal_or_tracking_z(args, model, base_cfg)
    args.prompt_identifier = identifier
    terrains = [value.strip() for value in args.terrains.split(",") if value.strip()]
    unknown = sorted(set(terrains) - set(SUPPORTED_TERRAINS))
    if unknown:
        raise ValueError(f"Unsupported terrains: {unknown}")
    same_z = clone_same_z_for_terrains(z, terrains)
    checksum = tensor_checksum(z)
    print(f"[INFO] prompt_type={args.prompt_type} prompt_source={identifier} z_shape={tuple(z.shape)} z_checksum={checksum}")
    results = []
    for terrain in terrains:
        assert tensor_checksum(same_z[terrain]) == checksum
        print(f"[INFO] terrain={terrain} z_checksum={checksum}")
        results.append(_run_rollout(args, model, base_cfg, terrain, same_z[terrain], target_states))
    if {row["z_checksum"] for row in results} != {checksum}:
        raise AssertionError("same-z checksum changed across terrain rollouts")
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
