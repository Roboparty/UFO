"""Benchmark the frozen direct-depth branch against the map-only environment."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from humanoidverse.train import build_ufo_mjlab_config


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _make_cfg(*, agent: str, num_envs: int, device: str, seed: int):
    return build_ufo_mjlab_config(
        device=device,
        work_dir="/tmp/pbfm-direct-depth-benchmark",
        num_envs=num_envs,
        num_env_steps=2048,
        seed=seed,
        use_wandb=False,
        wandb_run_name=None,
        disable_eval_prioritization=True,
        smoke=True,
        agent=agent,
        terrain_mode="mixed",
        disable_dr=True,
        disable_obs_noise=True,
    )


@torch.no_grad()
def _measure_env(agent: str, *, num_envs: int, device: str, seed: int, warmup: int, steps: int):
    cfg = _make_cfg(agent=agent, num_envs=num_envs, device=device, seed=seed)
    env, _ = cfg.env.build(num_envs=num_envs)
    actions = torch.zeros(num_envs, env._env.num_dof, device=device)
    try:
        env.reset(to_numpy=False)
        for _ in range(warmup):
            env.step(actions, to_numpy=False)
        _sync(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        start = time.perf_counter()
        observation = None
        for _ in range(steps):
            observation, *_ = env.step(actions, to_numpy=False)
        _sync(device)
        elapsed = time.perf_counter() - start
        memory = torch.cuda.max_memory_allocated(torch.device(device)) if device.startswith("cuda") else 0
        ray_counts = {
            name: int(sensor.data.distances.shape[-1])
            for name, sensor in env._env.mjlab_env.scene.sensors.items()
            if hasattr(sensor.data, "distances")
        }
        return elapsed, memory, observation, env.single_observation_space, cfg.agent.model, ray_counts
    finally:
        env.close()


@torch.no_grad()
def _measure_actor(model_cfg, obs_space, observation, *, num_envs: int, device: str, warmup: int, steps: int):
    model = model_cfg.model_copy(update={"device": device}).build(obs_space, action_dim=29)
    model.eval()
    observation = {key: value.to(device) for key, value in observation.items() if key != "time"}
    z = model.sample_z(num_envs, device=device)
    for _ in range(warmup):
        model.act(observation, z)
    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        model.act(observation, z)
    _sync(device)
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    direct_elapsed, peak_memory, observation, obs_space, model_cfg, direct_ray_counts = _measure_env(
        "fb_depth",
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        steps=args.steps,
    )
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    map_elapsed, _map_memory, _map_obs, _map_space, _map_model, map_ray_counts = _measure_env(
        "fb_terrain",
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        steps=args.steps,
    )
    actor_elapsed = _measure_actor(
        model_cfg,
        obs_space,
        observation,
        num_envs=args.num_envs,
        device=args.device,
        warmup=args.warmup,
        steps=args.steps,
    )
    transitions = args.num_envs * args.steps
    direct_step_s = direct_elapsed / args.steps
    map_step_s = map_elapsed / args.steps
    actor_step_s = actor_elapsed / args.steps
    estimated_combined_step_s = direct_step_s + actor_step_s
    # Compact replay stores only the newest uint8 36x32 frame and rebuilds
    # the 8-frame temporal input at sample time.
    depth_replay_bytes = 5_120_000 * 36 * 32
    return {
        "num_envs": args.num_envs,
        "steps": args.steps,
        "warmup": args.warmup,
        "device": args.device,
        "direct_env_step_ms": direct_step_s * 1000.0,
        "map_env_step_ms": map_step_s * 1000.0,
        "depth_sensor_increment_ms": (direct_step_s - map_step_s) * 1000.0,
        "depth_sensor_share": (direct_step_s - map_step_s) / max(direct_step_s, 1.0e-12),
        "direct_sensor_ray_counts_per_env": direct_ray_counts,
        "map_sensor_ray_counts_per_env": map_ray_counts,
        "camera_frames_per_second": transitions / direct_elapsed,
        "actor_step_ms": actor_step_s * 1000.0,
        "estimated_policy_env_fps": args.num_envs / estimated_combined_step_s,
        "peak_allocated_memory_gib": peak_memory / 2**30,
        "depth_observation_shape": list(observation["depth_image"].shape),
        "depth_observation_dtype": str(observation["depth_image"].dtype),
        "projected_compact_uint8_depth_replay_gib_per_rank": depth_replay_bytes / 2**30,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
