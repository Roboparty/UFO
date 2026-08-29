"""Five-pose GPU sanity check for self-occluding terrain depth."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

from humanoidverse.depth_terrain_evaluation import build_depth_evaluation_env, synchronize_depth_and_gt
from humanoidverse.mjlab_inference_utils import load_mjlab_env_cfg, replace_hydra_override
from humanoidverse.perception.depth_augmentation import MetricDepthAugmentation, MetricDepthAugmentationConfig
from humanoidverse.perception.depth_camera import DepthCameraConfig, rotate_xyzw, rotation_matrix_to_xyzw
from humanoidverse.perception.depth_preprocessing import resize_depth_with_conservative_invalid_mask
from humanoidverse.perception.local_depth_terrain_adapter import LocalDepthTerrainAdapter
from humanoidverse.perception.realsense_depth_runtime import RealSenseCalibration
from humanoidverse.perception.self_occluding_depth import (
    SelfOcclusionDepthConfig,
    make_self_occlusion_camera_pair,
    self_occluding_depth_from_sensors,
)

POSE_LABELS = ("upright", "crouch", "prone", "supine", "get_up_transition")


def _candidate_states(core) -> dict[str, tuple[int, float, dict[str, torch.Tensor], dict[str, float]]]:
    core._motion_lib.load_motions_for_evaluation(start_idx=0)
    keys = list(core._motion_lib.curr_motion_keys)
    if not keys or not all("fallAndGetUp" in key for key in keys):
        raise RuntimeError(f"the first evaluation motions are not get-up clips: {keys}")

    states: list[dict[str, torch.Tensor]] = []
    motion_ids: list[torch.Tensor] = []
    times: list[torch.Tensor] = []
    for motion_id in range(core._motion_lib.num_motions()):
        length = float(core._motion_lib._motion_lengths[motion_id].item())
        sample_times = torch.linspace(0.0, max(0.0, length - 1.0e-4), 121, device=core.device)
        ids = torch.full((sample_times.numel(),), motion_id, device=core.device, dtype=torch.long)
        states.append(core._motion_lib.get_motion_state(ids, sample_times))
        motion_ids.append(ids)
        times.append(sample_times)

    merged = {
        key: torch.cat([state[key] for state in states], dim=0)
        for key in ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel")
    }
    all_ids = torch.cat(motion_ids)
    all_times = torch.cat(times)
    root_quat = merged["root_rot"]
    count = root_quat.shape[0]
    up_w = rotate_xyzw(root_quat, torch.tensor((0.0, 0.0, 1.0), device=core.device).expand(count, -1))
    forward_w = rotate_xyzw(root_quat, torch.tensor((1.0, 0.0, 0.0), device=core.device).expand(count, -1))
    height = merged["root_pos"][:, 2]
    upright_mask = up_w[:, 2] > 0.85
    lying_mask = up_w[:, 2].abs() < 0.55
    if torch.count_nonzero(upright_mask) < 2 or not torch.any(lying_mask & (forward_w[:, 2] < -0.35)):
        raise RuntimeError("get-up clips do not contain the required upright/prone pose coverage")
    if not torch.any(lying_mask & (forward_w[:, 2] > 0.35)):
        raise RuntimeError("get-up clips do not contain a supine pose")

    neg_inf = torch.full_like(height, -torch.inf)
    pos_inf = torch.full_like(height, torch.inf)
    choices = {
        "upright": torch.argmax(torch.where(upright_mask, height, neg_inf)),
        "crouch": torch.argmin(torch.where(upright_mask, height, pos_inf)),
        "prone": torch.argmax(torch.where(lying_mask & (forward_w[:, 2] < -0.35), -forward_w[:, 2], neg_inf)),
        "supine": torch.argmax(torch.where(lying_mask & (forward_w[:, 2] > 0.35), forward_w[:, 2], neg_inf)),
        "get_up_transition": torch.argmin((up_w[:, 2].abs() - 0.55).abs() + 0.25 * (height - 0.55).abs()),
    }
    result = {}
    for label, tensor_index in choices.items():
        index = int(tensor_index.item())
        selected = {key: value[index : index + 1] for key, value in merged.items()}
        result[label] = (
            int(all_ids[index].item()),
            float(all_times[index].item()),
            selected,
            {
                "root_height_m": float(height[index].item()),
                "root_up_world_z": float(up_w[index, 2].item()),
                "root_forward_world_z": float(forward_w[index, 2].item()),
            },
        )
    return result


def _target_states(core, candidates) -> dict[str, torch.Tensor]:
    roots = []
    dofs = []
    for env_index, label in enumerate(POSE_LABELS):
        _motion_id, _time, state, _diagnostics = candidates[label]
        root_pos = state["root_pos"].clone()
        root_pos[:, :2] = core.env_origins[env_index : env_index + 1, :2]
        roots.append(
            torch.cat(
                (root_pos, state["root_rot"], state["root_vel"], state["root_ang_vel"]),
                dim=-1,
            )
        )
        dofs.append(torch.stack((state["dof_pos"], state["dof_vel"]), dim=-1))
    return {"root_states": torch.cat(roots), "dof_states": torch.cat(dofs)}


def _save_diagnostic(path: Path, labels, frame, partial_map, visible_mask) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(labels), 5, figsize=(18, 14), constrained_layout=True)
    depth_entries = (
        (frame.scene_depth_z, "scene first-hit optical-Z"),
        (frame.terrain_depth_z, "terrain-only optical-Z"),
        (frame.self_mask.to(torch.float32), "raw self mask"),
        (frame.final_depth_z, "final valid optical-Z"),
    )
    for row, label in enumerate(labels):
        for column, (values, title) in enumerate(depth_entries):
            image = values[row].detach().cpu().numpy()
            cmap = "gray" if column == 2 else "viridis"
            plotted = axes[row, column].imshow(image, vmin=None if column == 2 else 0.1, vmax=None if column == 2 else 2.0, cmap=cmap)
            if row == 0:
                axes[row, column].set_title(title)
            figure.colorbar(plotted, ax=axes[row, column], fraction=0.035)
        partial = partial_map[row].reshape(21, 13).T.detach().cpu().numpy()
        plotted = axes[row, 4].imshow(partial, origin="lower", aspect="auto", cmap="viridis_r")
        axes[row, 4].contour(visible_mask[row].reshape(21, 13).T.detach().cpu().numpy(), levels=(0.5,), colors="white", linewidths=0.6)
        if row == 0:
            axes[row, 4].set_title("273D partial clearance")
        figure.colorbar(plotted, ax=axes[row, 4], fraction=0.035)
        axes[row, 0].set_ylabel(label)
    figure.savefig(path, dpi=170)
    plt.close(figure)


@torch.no_grad()
def run_sanity(args: argparse.Namespace) -> dict[str, object]:
    device = args.device
    base_camera = DepthCameraConfig(
        width=480,
        height=270,
        horizontal_fov_deg=89.0,
        vertical_fov_deg=58.0,
        min_range=0.1,
        max_range=2.0,
        down_pitch_deg=48.0,
        include_geom_groups=(5,),
    )
    semantic_config = SelfOcclusionDepthConfig()
    terrain_camera, scene_camera = make_self_occlusion_camera_pair(base_camera, semantic_config)
    env_config, _ = load_mjlab_env_cfg(
        args.model_folder,
        data_path=None,
        robot_config=None,
        device=device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=10_000.0,
    )
    env_config = env_config.model_copy(
        update={
            "seed": args.seed,
            "hydra_overrides": replace_hydra_override(
                list(env_config.hydra_overrides),
                "terrain.terrain_type",
                "flat",
            ),
        }
    )
    wrapped, _ = build_depth_evaluation_env(
        env_config,
        num_envs=len(POSE_LABELS),
        camera=terrain_camera,
        extra_cameras=(
            (
                scene_camera,
                False,
                semantic_config.camera_housing_geom_names,
                semantic_config.camera_housing_mesh_names,
                semantic_config.camera_housing_geom_group,
            ),
        ),
    )
    core = wrapped._env
    try:
        candidates = _candidate_states(core)
        wrapped.reset(to_numpy=False, target_states=_target_states(core, candidates))
        synchronize_depth_and_gt(core, (terrain_camera.name, scene_camera.name))
        augmentation_config = MetricDepthAugmentationConfig(
            max_depth_m=2.0,
            blur_probability=1.0,
            sigma_min_px=0.0,
            sigma_max_px=3.0,
        )
        frame = self_occluding_depth_from_sensors(
            core.mjlab_env.scene.sensors[terrain_camera.name],
            core.mjlab_env.scene.sensors[scene_camera.name],
            terrain_camera,
            semantic_config,
            MetricDepthAugmentation(augmentation_config, seed=args.seed + 17_003),
        )
        if torch.any(frame.ambiguous_mask):
            raise RuntimeError("five-pose sanity found inconsistent dual-ray hits")
        depth_z, resized_self = resize_depth_with_conservative_invalid_mask(
            frame.final_depth_z,
            frame.dilated_self_mask,
            target_height=36,
            target_width=64,
        )
        calibration = RealSenseCalibration(
            native_width=480,
            native_height=270,
            target_width=64,
            target_height=36,
            intrinsic_matrix=tuple(float(value) for value in terrain_camera.intrinsics().reshape(-1)),
            depth_scale_m=1.0,
        )
        camera_quat_torso = rotation_matrix_to_xyzw(terrain_camera.torso_from_optical()).to(torch.float32)
        adapter = LocalDepthTerrainAdapter(
            calibration.target_intrinsics(),
            36,
            64,
            camera_pos_torso=terrain_camera.mount_pos_torso,
            camera_optical_quat_torso_xyzw=tuple(float(value) for value in camera_quat_torso),
        ).to(device)
        waist_indices = torch.tensor(
            [core.dof_names.index(name) for name in core.config.robot.waist_dof_names],
            device=device,
        )
        partial_map, visible_mask = adapter(depth_z, core.projected_gravity, core.dof_pos[:, waist_indices])
        if torch.any(torch.isfinite(depth_z) & resized_self):
            raise RuntimeError("self occlusion survived conservative downsampling")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = args.output_dir / "five_pose_self_occlusion_sanity.png"
        _save_diagnostic(image_path, POSE_LABELS, frame, partial_map, visible_mask)
        np.savez_compressed(
            args.output_dir / "five_pose_self_occlusion_sanity.npz",
            labels=np.asarray(POSE_LABELS),
            scene_depth_z=frame.scene_depth_z.cpu().numpy(),
            terrain_depth_z=frame.terrain_depth_z.cpu().numpy(),
            self_mask=frame.self_mask.cpu().numpy(),
            dilated_self_mask=frame.dilated_self_mask.cpu().numpy(),
            final_depth_z=frame.final_depth_z.cpu().numpy(),
            partial_map=partial_map.cpu().numpy(),
            visible_mask=visible_mask.cpu().numpy(),
        )
        pose_summary = {
            label: {
                "motion_id": candidates[label][0],
                "motion_key": core._motion_lib.curr_motion_keys[candidates[label][0]],
                "motion_time_s": candidates[label][1],
                **candidates[label][3],
                "sigma_px": float(frame.sigma_px[index].item()),
                "dilation_radius_px": int(frame.dilation_radius_px[index].item()),
                "self_fraction": float(frame.self_mask[index].float().mean().item()),
                "dilated_self_fraction": float(frame.dilated_self_mask[index].float().mean().item()),
                "partial_visibility": float(visible_mask[index].float().mean().item()),
            }
            for index, label in enumerate(POSE_LABELS)
        }
        summary = {
            "status": "passed",
            "image": str(image_path),
            "camera": asdict(terrain_camera),
            "contract": asdict(semantic_config),
            "poses": pose_summary,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        wrapped.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_sanity(parse_args()), indent=2))


if __name__ == "__main__":
    main()
