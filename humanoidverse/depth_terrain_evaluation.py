"""Side-channel evaluation of camera-derived terrain against PBFM GT rays."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    RESET_REGION_NAMES,
    HumanoidVerseMjlabCore,
    HumanoidVerseMjlabVectorEnv,
    _compose_humanoidverse_config,
    make_mjlab_ufo_env_cfg,
)
from humanoidverse.mjlab_inference_utils import load_mjlab_env_cfg, replace_hydra_override
from humanoidverse.perception.depth_camera import (
    DepthCameraConfig,
    depth_frame_from_raycast,
    make_depth_camera_sensor_cfg,
)
from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter
from humanoidverse.utils.torch_utils import calc_heading_quat, xyzw_to_wxyz

TERRAIN_NAMES = ("flat", "slope", "stairs", "rough", "platforms")


def build_depth_evaluation_env(env_config, *, num_envs: int, camera: DepthCameraConfig):
    """Build the normal UFO env with one evaluation-only raycast camera."""
    from mjlab.envs import ManagerBasedRlEnv

    hv_config, unresolved = _compose_humanoidverse_config(
        num_envs=num_envs,
        relative_config_path=env_config.relative_config_path,
        hydra_overrides=list(env_config.hydra_overrides),
        lafan_tail_path=env_config.lafan_tail_path,
        data_mix_weights=env_config.data_mix_weights,
        disable_obs_noise=env_config.disable_obs_noise,
        disable_domain_randomization=env_config.disable_domain_randomization,
        max_episode_length_s=env_config.max_episode_length_s,
        root_height_obs=env_config.root_height_obs,
        robot_training=env_config.robot_training,
    )
    mjlab_cfg = make_mjlab_ufo_env_cfg(
        hv_config,
        num_envs=num_envs,
        seed=env_config.seed,
        mjcf_path=env_config.mjcf_path,
        auto_reset=env_config.auto_reset,
        robot_training=env_config.robot_training,
    )
    camera_sensor_cfg = make_depth_camera_sensor_cfg(
        camera,
        torso_body_name=camera.mount_body,
    )
    mjlab_cfg.scene.sensors = tuple(mjlab_cfg.scene.sensors) + (camera_sensor_cfg,)
    mjlab_env = ManagerBasedRlEnv(mjlab_cfg, device=env_config.device)
    core = HumanoidVerseMjlabCore(hv_config, mjlab_env, creation_config=env_config)
    wrapped = HumanoidVerseMjlabVectorEnv(
        core,
        include_last_action=env_config.include_last_action,
        context_length=env_config.context_length,
        include_history_actor=env_config.include_history_actor,
        include_history_noaction=env_config.include_history_noaction,
    )
    wrapped._creation_config = env_config
    return wrapped, {"unresolved_conf": unresolved, "mjlab_env_cfg": mjlab_cfg}


def synchronize_depth_and_gt(core, camera_name: str) -> None:
    """Force both side-channel and teacher rays to observe one simulator state."""
    core.mjlab_env.scene.sensors[camera_name].update(0.0)
    if "terrain_height" in core.mjlab_env.scene.sensors:
        core.mjlab_env.scene.sensors["terrain_height"].update(0.0)
    core.mjlab_env.sim.sense()
    core._latest_terrain_observations = None
    core._refresh_state()


def terrain_names_for_envs(core) -> list[str]:
    ids = core._current_terrain_type_ids().detach().cpu().tolist()
    names = tuple(core.terrain_component_names)
    return [names[index] if 0 <= index < len(names) else f"terrain_{index}" for index in ids]


def validate_geometry_sample(
    *,
    frame,
    predicted: torch.Tensor,
    visible: torch.Tensor,
    gt: torch.Tensor,
    root_state: torch.Tensor,
) -> None:
    """Fail fast on depth, pose, mask, and output contract violations."""
    if predicted.shape != gt.shape or predicted.shape[-1] != 273:
        raise RuntimeError(f"camera and GT terrain shapes differ: predicted={predicted.shape}, gt={gt.shape}")
    if visible.shape != predicted.shape or visible.dtype != torch.bool:
        raise RuntimeError("visible mask must be bool and match C_geo")
    if not torch.equal(visible, torch.isfinite(predicted)):
        raise RuntimeError("visible mask must equal isfinite(C_geo)")
    if not torch.equal(frame.valid, torch.isfinite(frame.depth_z)):
        raise RuntimeError("valid depth mask must equal isfinite(depth_z)")
    if torch.any(frame.depth_z[frame.valid] <= 0.0):
        raise RuntimeError("valid optical-Z depth must be positive")
    for name, value in (
        ("camera_pos_w", frame.camera_pos_w),
        ("camera_optical_quat_w", frame.camera_optical_quat_w),
        ("root_state", root_state),
    ):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains non-finite values")


def stair_edge_mask(gt: torch.Tensor, *, threshold: float = 0.05) -> torch.Tensor:
    """Mark GT cells adjacent to a terrain discontinuity."""
    grid = gt.reshape(-1, 21, 13)
    finite = torch.isfinite(grid)
    edge = torch.zeros_like(finite)
    dx = finite[:, 1:] & finite[:, :-1] & ((grid[:, 1:] - grid[:, :-1]).abs() > threshold)
    dy = finite[:, :, 1:] & finite[:, :, :-1] & ((grid[:, :, 1:] - grid[:, :, :-1]).abs() > threshold)
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    edge[:, :, 1:] |= dy
    edge[:, :, :-1] |= dy
    return edge.reshape(-1, 273)


class MetricAccumulator:
    def __init__(self, device: torch.device) -> None:
        self.samples = 0
        self.visible_count = torch.zeros(273, device=device, dtype=torch.float64)
        self.gt_valid_count = torch.zeros(273, device=device, dtype=torch.float64)
        self.abs_error_sum = torch.zeros(273, device=device, dtype=torch.float64)
        self.sq_error_sum = torch.zeros(273, device=device, dtype=torch.float64)
        self.visible_error_count = torch.zeros(273, device=device, dtype=torch.float64)

    def update(self, predicted: torch.Tensor, visible: torch.Tensor, gt: torch.Tensor) -> None:
        finite_gt = torch.isfinite(gt)
        valid = visible & finite_gt & torch.isfinite(predicted)
        error = torch.where(valid, predicted - gt, torch.zeros_like(predicted)).double()
        self.samples += predicted.shape[0]
        self.visible_count += visible.double().sum(dim=0)
        self.gt_valid_count += finite_gt.double().sum(dim=0)
        self.abs_error_sum += error.abs().sum(dim=0)
        self.sq_error_sum += error.square().sum(dim=0)
        self.visible_error_count += valid.double().sum(dim=0)

    def summary(self) -> dict[str, Any]:
        count = self.visible_error_count
        total = float(count.sum().item())
        mae = float(self.abs_error_sum.sum().item() / total) if total else None
        rmse = float((self.sq_error_sum.sum().item() / total) ** 0.5) if total else None
        center_count = float(count[DepthTerrainAdapter.CENTER_INDEX].item())
        return {
            "samples": self.samples,
            "visible_fraction": float(self.visible_count.sum().item() / max(self.samples * 273, 1)),
            "gt_valid_fraction": float(self.gt_valid_count.sum().item() / max(self.samples * 273, 1)),
            "visible_mae_m": mae,
            "visible_rmse_m": rmse,
            "center_visibility": float(self.visible_count[58].item() / max(self.samples, 1)),
            "center_mae_m": (
                float(self.abs_error_sum[58].item() / center_count) if center_count else None
            ),
        }

    def per_cell(self) -> dict[str, np.ndarray]:
        count = self.visible_error_count.cpu().numpy()
        return {
            "visibility_probability": self.visible_count.cpu().numpy() / max(self.samples, 1),
            "mae_m": np.divide(
                self.abs_error_sum.cpu().numpy(),
                count,
                out=np.full(273, np.nan),
                where=count > 0,
            ),
            "rmse_m": np.sqrt(
                np.divide(
                    self.sq_error_sum.cpu().numpy(),
                    count,
                    out=np.full(273, np.nan),
                    where=count > 0,
                )
            ),
            "count": count,
        }


def region_metrics(predicted: torch.Tensor, visible: torch.Tensor, gt: torch.Tensor) -> dict[str, dict[str, float | None]]:
    grid_offsets = DepthTerrainAdapter(
        torch.eye(3, device=predicted.device, dtype=predicted.dtype), 1, 1
    ).grid_offsets
    x, y = grid_offsets[:, 0], grid_offsets[:, 1]
    masks = {
        "rear": x < 0.0,
        "underfoot": (x.abs() <= 0.2001) & (y.abs() <= 0.2001),
        "forward": x > 0.20,
    }
    output: dict[str, dict[str, float | None]] = {}
    for name, cell_mask in masks.items():
        camera_visible = visible[:, cell_mask]
        gt_valid = torch.isfinite(gt[:, cell_mask])
        valid = camera_visible & gt_valid
        error = (predicted[:, cell_mask] - gt[:, cell_mask])[valid]
        output[name] = {
            "visible_fraction": float(camera_visible.float().mean().item()),
            "gt_valid_fraction": float(gt_valid.float().mean().item()),
            "mae_m": float(error.abs().mean().item()) if error.numel() else None,
        }
    return output


def save_snapshot(
    path: Path,
    *,
    depth_z: torch.Tensor,
    visible: torch.Tensor,
    predicted: torch.Tensor,
    gt: torch.Tensor,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    predicted_grid = predicted.reshape(21, 13).detach().cpu().numpy()
    gt_grid = gt.reshape(21, 13).detach().cpu().numpy()
    visible_grid = visible.reshape(21, 13).detach().cpu().numpy()
    error = np.abs(predicted_grid - gt_grid)
    figure, axes = plt.subplots(1, 5, figsize=(19, 3.6), constrained_layout=True)
    entries = (
        (depth_z.detach().cpu().numpy(), "optical-Z depth", "viridis"),
        (visible_grid.T, "273 visible", "gray"),
        (predicted_grid.T, "camera clearance", "viridis_r"),
        (gt_grid.T, "GT clearance", "viridis_r"),
        (error.T, "absolute error", "magma"),
    )
    for axis, (image, label, cmap) in zip(axes, entries):
        plotted = axis.imshow(image, origin="lower", aspect="auto", cmap=cmap)
        axis.set_title(label)
        figure.colorbar(plotted, ax=axis, fraction=0.046)
    for axis in axes[1:]:
        axis.set_xlabel("x: rear to forward")
        axis.set_ylabel("y: right to left")
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_heatmaps(path: Path, per_cell: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, key, label, cmap in (
        (axes[0], "visibility_probability", "visibility probability", "viridis"),
        (axes[1], "mae_m", "visible MAE [m]", "magma"),
    ):
        image = axis.imshow(per_cell[key].reshape(21, 13).T, origin="lower", aspect="auto", cmap=cmap)
        axis.set_xlabel("x: -0.4 m rear to +1.6 m forward")
        axis.set_ylabel("y: -0.6 m right to +0.6 m left")
        axis.set_title(label)
        figure.colorbar(image, ax=axis)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate_explicit_probe(
    core,
    adapter: DepthTerrainAdapter,
    camera: DepthCameraConfig,
    *,
    xy: tuple[float, float],
    yaw: float,
    output_path: Path,
    label: str,
) -> dict[str, float | None]:
    """Place env zero at a diagnostic XY and compare camera and GT rays."""
    env_ids = torch.zeros(1, device=core.device, dtype=torch.long)
    root_xyzw = core.robot_root_states[0:1].clone()
    root_xyzw[:, :3] = torch.tensor([[xy[0], xy[1], 0.80]], device=core.device)
    root_xyzw[:, 3:7] = torch.tensor(
        [[0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)]], device=core.device
    )
    root_xyzw[:, 7:13] = 0.0
    root_wxyz = torch.cat(
        (root_xyzw[:, :3], xyzw_to_wxyz(root_xyzw[:, 3:7]), root_xyzw[:, 7:13]), dim=-1
    )
    core.robot.write_root_state_to_sim(root_wxyz, env_ids=env_ids)
    core.robot.write_joint_state_to_sim(
        core.default_dof_pos[0:1],
        torch.zeros_like(core.default_dof_pos[0:1]),
        joint_ids=core._joint_ids,
        env_ids=env_ids,
    )
    core.mjlab_env.scene.write_data_to_sim()
    core.mjlab_env.sim.forward()
    synchronize_depth_and_gt(core, camera.name)
    frame = depth_frame_from_raycast(core.mjlab_env.scene.sensors[camera.name], camera)
    gt = core._terrain_actor_obs().clone()
    gt = torch.where(
        gt < float(core.config.terrain.terrain_priv.max_ray_distance) * 0.999,
        gt,
        torch.full_like(gt, float("nan")),
    )
    predicted, visible = adapter(
        frame.depth_z,
        frame.camera_pos_w,
        frame.camera_optical_quat_w,
        core.robot_root_states[:, :3],
        calc_heading_quat(core.base_quat, w_last=True),
    )
    save_snapshot(
        output_path,
        depth_z=frame.depth_z[0],
        visible=visible[0],
        predicted=predicted[0],
        gt=gt[0],
        title=label,
    )
    valid = visible[0] & torch.isfinite(gt[0])
    error = (predicted[0] - gt[0])[valid]
    return {
        "visible_fraction": float(visible[0].float().mean().item()),
        "gt_valid_fraction": float(torch.isfinite(gt[0]).float().mean().item()),
        "mae_m": float(error.abs().mean().item()) if error.numel() else None,
        "rmse_m": float(error.square().mean().sqrt().item()) if error.numel() else None,
    }


def evaluate_depth_terrain(
    *,
    model_folder: Path,
    output_dir: Path,
    num_envs: int,
    num_steps: int,
    terrain: str,
    device: str,
    seed: int,
    camera: DepthCameraConfig,
) -> dict[str, Any]:
    if terrain not in {"mixed", *TERRAIN_NAMES}:
        raise ValueError(f"terrain must be mixed or one of {TERRAIN_NAMES}, got {terrain!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
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
    env_updates: dict[str, Any] = {"seed": seed}
    if terrain != "mixed":
        env_updates["hydra_overrides"] = replace_hydra_override(
            list(env_config.hydra_overrides), "terrain.terrain_type", terrain
        )
    env_config = env_config.model_copy(update=env_updates)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        before_allocated = torch.cuda.memory_allocated(device)
    else:
        before_allocated = 0
    build_start = time.perf_counter()
    wrapped_env, _ = build_depth_evaluation_env(env_config, num_envs=num_envs, camera=camera)
    build_seconds = time.perf_counter() - build_start
    core = wrapped_env._env
    adapter = DepthTerrainAdapter(camera.intrinsics(), camera.height, camera.width).to(device)
    accumulators: dict[str, MetricAccumulator] = {"overall": MetricAccumulator(torch.device(device))}
    edge_abs_sum = edge_sq_sum = edge_count = 0.0
    nonedge_abs_sum = nonedge_sq_sum = nonedge_count = 0.0
    region_buffers: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
    snapshots_saved: set[str] = set()
    elapsed_step_seconds = 0.0
    partial_reset_exercised = False

    try:
        wrapped_env.reset(to_numpy=False)
        zero_actions = torch.zeros((num_envs, core.num_dof), device=device)
        for step in range(num_steps):
            if num_envs > 1 and num_steps > 1 and step == num_steps // 2:
                core.reset_idx(torch.zeros(1, device=device, dtype=torch.long))
                partial_reset_exercised = True
            synchronize_depth_and_gt(core, camera.name)
            frame = depth_frame_from_raycast(core.mjlab_env.scene.sensors[camera.name], camera)
            gt = core._terrain_actor_obs().clone()
            gt = torch.where(
                gt < float(core.config.terrain.terrain_priv.max_ray_distance) * 0.999,
                gt,
                torch.full_like(gt, float("nan")),
            )
            heading = calc_heading_quat(core.base_quat, w_last=True)
            predicted, visible = adapter(
                frame.depth_z,
                frame.camera_pos_w,
                frame.camera_optical_quat_w,
                core.robot_root_states[:, :3],
                heading,
            )
            validate_geometry_sample(
                frame=frame,
                predicted=predicted,
                visible=visible,
                gt=gt,
                root_state=core.robot_root_states,
            )
            names = terrain_names_for_envs(core)
            accumulators["overall"].update(predicted, visible, gt)
            for name in sorted(set(names)):
                mask = torch.tensor([item == name for item in names], device=device)
                accumulators.setdefault(name, MetricAccumulator(torch.device(device))).update(
                    predicted[mask], visible[mask], gt[mask]
                )
                region_buffers[name].append((predicted[mask], visible[mask], gt[mask]))

            stairs_mask = torch.tensor([item == "stairs" for item in names], device=device)
            if torch.any(stairs_mask):
                edges = stair_edge_mask(gt[stairs_mask])
                valid = visible[stairs_mask] & torch.isfinite(gt[stairs_mask])
                error = predicted[stairs_mask] - gt[stairs_mask]
                for edge_choice in (True, False):
                    selected = valid & (edges if edge_choice else ~edges)
                    values = error[selected]
                    if edge_choice:
                        edge_abs_sum += float(values.abs().sum().item())
                        edge_sq_sum += float(values.square().sum().item())
                        edge_count += values.numel()
                    else:
                        nonedge_abs_sum += float(values.abs().sum().item())
                        nonedge_sq_sum += float(values.square().sum().item())
                        nonedge_count += values.numel()

            for env_index, name in enumerate(names):
                label = name
                reset_region = RESET_REGION_NAMES[int(core._reset_region_ids[env_index].item())]
                if reset_region == "tile_seam":
                    label = "tile_seam"
                if name == "stairs":
                    grid = gt[env_index].reshape(21, 13)
                    center = grid[4, 6]
                    forward = grid[5:, 6]
                    if torch.any(forward < center - 0.05):
                        label = "stairs_ascent"
                    elif torch.any(forward > center + 0.05):
                        label = "stairs_descent"
                if label not in snapshots_saved:
                    save_snapshot(
                        output_dir / f"snapshot_{label}.png",
                        depth_z=frame.depth_z[env_index],
                        visible=visible[env_index],
                        predicted=predicted[env_index],
                        gt=gt[env_index],
                        title=f"{label}, step={step}, env={env_index}",
                    )
                    snapshots_saved.add(label)

            if step + 1 < num_steps:
                step_start = time.perf_counter()
                wrapped_env.step(zero_actions, to_numpy=False)
                elapsed_step_seconds += time.perf_counter() - step_start

        explicit_probes: dict[str, dict[str, float | None]] = {}
        if core._terrain_patch_size is not None:
            patch_x, patch_y = (float(value) for value in core._terrain_patch_size.tolist())
            core_half_x = 0.5 * patch_x * core._terrain_grid_rows
            explicit_probes["padding"] = evaluate_explicit_probe(
                core,
                adapter,
                camera,
                xy=(core_half_x + 0.5, 0.0),
                yaw=0.0,
                output_path=output_dir / "snapshot_padding.png",
                label="global group-5 padding",
            )
            snapshots_saved.add("padding")
            if core._terrain_grid_cols > 1:
                first_seam_y = -0.5 * patch_y * core._terrain_grid_cols + patch_y
                explicit_probes["tile_seam"] = evaluate_explicit_probe(
                    core,
                    adapter,
                    camera,
                    xy=(0.5, first_seam_y - 0.5),
                    yaw=np.pi / 2.0,
                    output_path=output_dir / "snapshot_tile_seam.png",
                    label="internal connected-tile seam",
                )
                snapshots_saved.add("tile_seam")

        overall_cells = accumulators["overall"].per_cell()
        save_heatmaps(output_dir / "terrain_camera_heatmaps.png", overall_cells)
        with (output_dir / "per_cell_metrics.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("index", "x_m", "y_m", "visibility_probability", "mae_m", "rmse_m", "visible_count"))
            offsets = adapter.grid_offsets.detach().cpu().numpy()
            for index in range(273):
                writer.writerow(
                    (
                        index,
                        offsets[index, 0],
                        offsets[index, 1],
                        overall_cells["visibility_probability"][index],
                        overall_cells["mae_m"][index],
                        overall_cells["rmse_m"][index],
                        overall_cells["count"][index],
                    )
                )

        summaries = {name: accumulator.summary() for name, accumulator in accumulators.items()}
        region_summary = {}
        for name, batches in region_buffers.items():
            predicted = torch.cat([batch[0] for batch in batches], dim=0)
            visible = torch.cat([batch[1] for batch in batches], dim=0)
            gt = torch.cat([batch[2] for batch in batches], dim=0)
            region_summary[name] = region_metrics(predicted, visible, gt)
        report = {
            "model_folder": str(model_folder.resolve()),
            "terrain": terrain,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "seed": seed,
            "camera": asdict(camera),
            "intrinsic_matrix": camera.intrinsics().tolist(),
            "metrics": summaries,
            "regions": region_summary,
            "stairs": {
                "edge_mae_m": edge_abs_sum / edge_count if edge_count else None,
                "edge_rmse_m": (edge_sq_sum / edge_count) ** 0.5 if edge_count else None,
                "edge_count": edge_count,
                "nonedge_mae_m": nonedge_abs_sum / nonedge_count if nonedge_count else None,
                "nonedge_rmse_m": (nonedge_sq_sum / nonedge_count) ** 0.5 if nonedge_count else None,
                "nonedge_count": nonedge_count,
            },
            "explicit_probes": explicit_probes,
            "performance": {
                "build_seconds": build_seconds,
                "policy_steps_per_second": (
                    (num_steps - 1) / elapsed_step_seconds if elapsed_step_seconds > 0.0 else None
                ),
                "environment_transitions_per_second": (
                    num_envs * (num_steps - 1) / elapsed_step_seconds
                    if elapsed_step_seconds > 0.0
                    else None
                ),
                "allocated_vram_delta_bytes": (
                    torch.cuda.memory_allocated(device) - before_allocated if device.startswith("cuda") else None
                ),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
            },
            "snapshots": sorted(snapshots_saved),
            "partial_reset_exercised": partial_reset_exercised,
        }
        (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        with (output_dir / "metrics.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("terrain", "samples", "visible_fraction", "gt_valid_fraction", "visible_mae_m", "visible_rmse_m", "center_visibility", "center_mae_m"))
            for name, summary in summaries.items():
                writer.writerow((name, *(summary[key] for key in ("samples", "visible_fraction", "gt_valid_fraction", "visible_mae_m", "visible_rmse_m", "center_visibility", "center_mae_m"))))
        return report
    finally:
        wrapped_env.close()


def benchmark_baseline_environment(
    *,
    model_folder: Path,
    num_envs: int,
    num_steps: int,
    terrain: str,
    device: str,
    seed: int,
) -> dict[str, float | int | None]:
    """Benchmark the unchanged environment without the side-channel camera."""
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
    updates: dict[str, Any] = {"seed": seed}
    if terrain != "mixed":
        updates["hydra_overrides"] = replace_hydra_override(
            list(env_config.hydra_overrides), "terrain.terrain_type", terrain
        )
    env_config = env_config.model_copy(update=updates)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
    else:
        before = 0
    build_start = time.perf_counter()
    wrapped_env, _ = env_config.build(num_envs=num_envs)
    build_seconds = time.perf_counter() - build_start
    core = wrapped_env._env
    try:
        wrapped_env.reset(to_numpy=False)
        actions = torch.zeros((num_envs, core.num_dof), device=device)
        for _ in range(min(2, num_steps)):
            wrapped_env.step(actions, to_numpy=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(num_steps):
            wrapped_env.step(actions, to_numpy=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        return {
            "num_envs": num_envs,
            "num_steps": num_steps,
            "build_seconds": build_seconds,
            "policy_steps_per_second": num_steps / elapsed,
            "environment_transitions_per_second": num_envs * num_steps / elapsed,
            "allocated_vram_delta_bytes": (
                torch.cuda.memory_allocated(device) - before if device.startswith("cuda") else None
            ),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
        }
    finally:
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--terrain", choices=("mixed", *TERRAIN_NAMES), default="mixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--horizontal-fov", type=float, default=89.0)
    parser.add_argument("--vertical-fov", type=float, default=58.0)
    parser.add_argument("--intrinsic-matrix", type=float, nargs=9, default=None)
    parser.add_argument("--mount-body", default="torso_link")
    parser.add_argument(
        "--mount-pos",
        type=float,
        nargs=3,
        default=(0.0487988662332928, 0.01, 0.4378029937970051),
    )
    parser.add_argument("--down-pitch", type=float, default=48.0)
    parser.add_argument("--min-range", type=float, default=0.10)
    parser.add_argument("--max-range", type=float, default=2.50)
    parser.add_argument("--geom-groups", type=int, nargs="+", default=(5,))
    parser.add_argument(
        "--benchmark-baseline",
        action="store_true",
        help="Also build the unchanged no-camera env and report camera slowdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = DepthCameraConfig(
        width=args.width,
        height=args.height,
        horizontal_fov_deg=args.horizontal_fov,
        vertical_fov_deg=args.vertical_fov,
        intrinsic_matrix=tuple(args.intrinsic_matrix) if args.intrinsic_matrix is not None else None,
        mount_body=args.mount_body,
        mount_pos_torso=tuple(args.mount_pos),
        down_pitch_deg=args.down_pitch,
        min_range=args.min_range,
        max_range=args.max_range,
        include_geom_groups=tuple(args.geom_groups),
    )
    baseline = None
    if args.benchmark_baseline:
        baseline = benchmark_baseline_environment(
            model_folder=args.model_folder,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            terrain=args.terrain,
            device=args.device,
            seed=args.seed,
        )
    report = evaluate_depth_terrain(
        model_folder=args.model_folder,
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        terrain=args.terrain,
        device=args.device,
        seed=args.seed,
        camera=camera,
    )
    if baseline is not None:
        camera_rate = report["performance"]["policy_steps_per_second"]
        baseline_rate = baseline["policy_steps_per_second"]
        report["performance"]["baseline"] = baseline
        report["performance"]["camera_slowdown_fraction"] = (
            1.0 - camera_rate / baseline_rate if camera_rate is not None else None
        )
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
