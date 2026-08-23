"""Region-conditioned stairs evaluation for frozen PBFM checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

import torch

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import (
    DEFAULT_INFERENCE_DATA_PATH,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    resolve_inference_robot_config,
)
from humanoidverse.terrain_transfer import tensor_checksum
from humanoidverse.terrain_transfer_inference import (
    _compute_reward_z,
    _run_rollout,
    _save_prompt_latent,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec


TERRAINS = ("stairs_up", "stairs_down")
RATE_FIELDS = (
    "center_departed",
    "first_transition",
    "outer_ground_reached",
    "stalled_at_center",
    "center_looped",
    "impact_safe",
    "normal_final_clearance",
    "fell",
)
MEAN_FIELDS = (
    "consecutive_steps_completed",
    "max_stair_level_reached",
    "cumulative_planar_path",
    "mean_body_impact",
    "max_body_impact",
    "min_ground_clearance",
    "final_ground_clearance",
    "forward_displacement",
    "mean_root_velocity",
)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"episodes": len(rows)}
    for field in RATE_FIELDS:
        summary[f"{field}_rate"] = sum(bool(row[field]) for row in rows) / len(rows)
    for field in MEAN_FIELDS:
        values = [float(row[field]) for row in rows]
        summary[f"{field}_mean"] = statistics.fmean(values)
        summary[f"{field}_min"] = min(values)
        summary[f"{field}_max"] = max(values)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--buffer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_INFERENCE_DATA_PATH)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--reward-task", default="move-ego-0-0.7")
    parser.add_argument("--seeds", default="4728,4729,4730,4731,4732")
    parser.add_argument("--episode-length", type=int, default=1500)
    parser.add_argument("--fall-clearance", type=float, default=0.45)
    parser.add_argument("--min-descent-steps", type=int, default=3)
    parser.add_argument("--max-body-impact", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_folder = args.model_folder.expanduser().resolve()
    args.buffer_path = args.buffer_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.robot_config = resolve_inference_robot_config(args.robot_config, None)
    args.robot_training = load_robot_training_spec(args.robot_config)
    args.data_path = args.data_path.expanduser().resolve()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("At least one seed is required")

    checkpoint_dir = args.model_folder / "checkpoint"
    model = load_model_from_checkpoint_dir(
        checkpoint_dir, device=checkpoint_load_device(args.device)
    )
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

    args.prompt_type = "reward"
    args.buffer_rank = 0
    args.process_executor = False
    args.output = args.output_dir / "latent_probe.json"
    z, identifier, target_states = _compute_reward_z(args, model)
    args.prompt_identifier = identifier
    latent_path = args.output_dir / "forward_latent.pt"
    _save_prompt_latent(
        latent_path,
        z,
        prompt_type=args.prompt_type,
        identifier=identifier,
    )
    checksum = tensor_checksum(z)

    args.dense_terrain = False
    args.patch_size = None
    args.stairs_start_step = 0
    args.stairs_down_edge_margin = None
    args.camera_distance = 3.0
    args.camera_azimuth = 135.0
    args.camera_elevation = -18.0

    rows: list[dict[str, Any]] = []
    for terrain in TERRAINS:
        for seed_index, seed in enumerate(seeds):
            args.seed = seed
            args.stairs_reset_region = None
            args.save_mp4 = seed_index == 0
            args.output = args.output_dir / f"{terrain}.json"
            result = _run_rollout(args, model, base_cfg, terrain, z, target_states)
            result["evaluation_terrain"] = terrain
            rows.append(result)
            if args.save_mp4:
                generated = args.output.with_suffix("").with_name(
                    f"{args.output.stem}_{terrain}.mp4"
                )
                generated.replace(args.output_dir / f"{terrain}.mp4")

    metrics_path = args.output_dir / "metrics.csv"
    fieldnames = list(rows[0])
    with metrics_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_path = checkpoint_dir / "train_status.json"
    train_status = json.loads(status_path.read_text()) if status_path.exists() else None
    summary = {
        "checkpoint_global_time": (
            int(train_status["global_time"]) if train_status is not None else None
        ),
        "reward_task": args.reward_task,
        "z_checksum": checksum,
        "seeds": list(seeds),
        "episode_length": args.episode_length,
        "fps": args.fps,
        "fixed_thresholds": {
            "fall_clearance": args.fall_clearance,
            "max_body_impact": args.max_body_impact,
            "center_departure_radius": "platform_width / 2",
            "center_loop_min_path": "2 * platform_width",
        },
        "terrains": {
            terrain: _aggregate(
                [row for row in rows if row["evaluation_terrain"] == terrain]
            )
            for terrain in TERRAINS
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "raw_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] wrote stairs region evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
