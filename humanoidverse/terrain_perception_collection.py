"""Collect projected terrain-map supervision from a frozen GT-map policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.depth_terrain_evaluation import (
    TERRAIN_NAMES,
    build_depth_evaluation_env,
    synchronize_depth_and_gt,
)
from humanoidverse.mjlab_inference_utils import (
    checkpoint_load_device,
    load_mjlab_env_cfg,
    replace_hydra_override,
)
from humanoidverse.perception.depth_camera import DepthCameraConfig, depth_frame_from_raycast
from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter
from humanoidverse.perception.terrain_dataset import (
    TerrainPerceptionChunkWriter,
    TerrainPerceptionFrameBatch,
)
from humanoidverse.utils.torch_utils import calc_heading_quat, get_euler_xyz


def collect_terrain_perception(
    *,
    model_folder: Path,
    output_dir: Path,
    num_envs: int,
    num_steps: int,
    terrain: str,
    device: str,
    seed: int,
    chunk_steps: int,
    camera: DepthCameraConfig,
) -> dict[str, object]:
    """Roll out a frozen policy on GT maps while recording camera-map pairs."""
    if min(num_envs, num_steps, chunk_steps) <= 0:
        raise ValueError("num_envs, num_steps, and chunk_steps must be positive")
    if terrain not in {"mixed", *TERRAIN_NAMES}:
        raise ValueError(f"unknown terrain selection: {terrain!r}")

    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    model = load_model_from_checkpoint_dir(
        checkpoint_dir,
        device=checkpoint_load_device(device),
    )
    model.to(device)
    model.eval()

    env_config, _ = load_mjlab_env_cfg(
        model_folder,
        data_path=None,
        robot_config=None,
        device=device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=10_000.0,
    )
    updates: dict[str, object] = {"seed": seed}
    if terrain != "mixed":
        updates["hydra_overrides"] = replace_hydra_override(
            list(env_config.hydra_overrides),
            "terrain.terrain_type",
            terrain,
        )
    env_config = env_config.model_copy(update=updates)
    wrapped_env, _ = build_depth_evaluation_env(env_config, num_envs=num_envs, camera=camera)
    core = wrapped_env._env
    adapter = DepthTerrainAdapter(camera.intrinsics(), camera.height, camera.width).to(device)
    episode_id = torch.zeros(num_envs, device=device, dtype=torch.long)
    episode_time = torch.zeros(num_envs, device=device)
    env_id = torch.arange(num_envs, device=device, dtype=torch.long)
    latent = model.sample_z(num_envs, device=device)
    visible_sum = 0
    frame_count = 0

    metadata = {
        "model_folder": str(model_folder),
        "checkpoint_dir": str(checkpoint_dir),
        "terrain": terrain,
        "seed": seed,
        "control_dt_s": core.dt,
        "camera": asdict(camera),
        "policy_terrain_input": "GT terrain_actor",
        "stored_depth": False,
    }
    writer = TerrainPerceptionChunkWriter(
        output_dir,
        chunk_steps=chunk_steps,
        metadata=metadata,
    )
    try:
        observation, _ = wrapped_env.reset(to_numpy=False)
        for _step in range(num_steps):
            synchronize_depth_and_gt(core, camera.name)
            frame = depth_frame_from_raycast(core.mjlab_env.scene.sensors[camera.name], camera)
            heading_quat = calc_heading_quat(core.base_quat, w_last=True)
            partial_map, visible_mask = adapter(
                frame.depth_z,
                frame.camera_pos_w,
                frame.camera_optical_quat_w,
                core.robot_root_states[:, :3],
                heading_quat,
            )
            gt = core._terrain_actor_obs().clone()
            yaw = get_euler_xyz(core.base_quat, w_last=True)[2]
            writer.append(
                TerrainPerceptionFrameBatch(
                    partial_map=partial_map,
                    visible_mask=visible_mask,
                    pelvis_pos_w=core.robot_root_states[:, :3],
                    heading_yaw_w=yaw,
                    timestamp_s=episode_time,
                    proprio=observation["state"],
                    gt_terrain_actor=gt,
                    episode_id=episode_id,
                    env_id=env_id,
                    terrain_type=core._current_terrain_type_ids(),
                )
            )
            visible_sum += int(visible_mask.sum().item())
            frame_count += visible_mask.numel()

            with torch.no_grad():
                action = model.act(observation, latent, mean=True)
            observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
            reset = (
                torch.as_tensor(terminated, device=device).bool()
                | torch.as_tensor(
                    truncated,
                    device=device,
                ).bool()
            )
            episode_time += core.dt
            if torch.any(reset):
                episode_id[reset] += 1
                episode_time[reset] = 0.0
                latent[reset] = model.sample_z(int(reset.sum().item()), device=device)
    finally:
        writer.close()
        wrapped_env.close()

    summary = {
        **metadata,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "frames": num_envs * num_steps,
        "visible_fraction": visible_sum / frame_count if frame_count else 0.0,
        "history_target_s": 0.6,
    }
    (output_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--chunk-steps", type=int, default=128)
    parser.add_argument("--terrain", choices=("mixed", *TERRAIN_NAMES), default="mixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--horizontal-fov", type=float, default=89.0)
    parser.add_argument("--vertical-fov", type=float, default=58.0)
    parser.add_argument("--down-pitch", type=float, default=48.0)
    parser.add_argument("--min-range", type=float, default=0.10)
    parser.add_argument("--max-range", type=float, default=2.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = DepthCameraConfig(
        width=args.width,
        height=args.height,
        horizontal_fov_deg=args.horizontal_fov,
        vertical_fov_deg=args.vertical_fov,
        down_pitch_deg=args.down_pitch,
        min_range=args.min_range,
        max_range=args.max_range,
        include_geom_groups=(5,),
    )
    result = collect_terrain_perception(
        model_folder=args.model_folder,
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        chunk_steps=args.chunk_steps,
        terrain=args.terrain,
        device=args.device,
        seed=args.seed,
        camera=camera,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
