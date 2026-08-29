"""Render the RP1-compatible direct-depth contract on five G1 poses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

from humanoidverse.perception.instinct_direct_depth import (
    RP1DirectDepthConfig,
    preprocess_rp1_depth,
    raycast_ranges_to_image_plane,
)
from humanoidverse.self_occlusion_sanity import _candidate_states
from humanoidverse.train import build_ufo_mjlab_config

POSE_LABELS = ("upright", "crouch", "arm_crossing", "prone", "get_up_transition")


def _select_candidates(core):
    candidates = _candidate_states(core)
    # The first get-up clips include arm motion. Pick the upright sample whose
    # wrists are closest together as the reproducible arm-crossing fixture.
    wrist_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    wrist_ids = [core.body_names.index(name) for name in wrist_names]
    best = None
    for motion_id in range(core._motion_lib.num_motions()):
        length = float(core._motion_lib._motion_lengths[motion_id].item())
        times = torch.linspace(0.0, max(0.0, length - 1.0e-4), 241, device=core.device)
        ids = torch.full((times.numel(),), motion_id, device=core.device, dtype=torch.long)
        states = core._motion_lib.get_motion_state(ids, times)
        wrists = states["rg_pos_t"][:, wrist_ids]
        separation = torch.linalg.vector_norm(wrists[:, 0] - wrists[:, 1], dim=-1)
        index = int(torch.argmin(separation).item())
        score = float(separation[index].item())
        if best is None or score < best[0]:
            selected = {
                key: states[key][index : index + 1]
                for key in ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel")
            }
            best = (score, motion_id, float(times[index].item()), selected)
    assert best is not None
    candidates["arm_crossing"] = (
        best[1],
        best[2],
        best[3],
        {"wrist_separation_m": best[0]},
    )
    return candidates


def _target_states(core, candidates):
    roots, dofs = [], []
    for env_id, label in enumerate(POSE_LABELS):
        state = candidates[label][2]
        root_pos = state["root_pos"].clone()
        root_pos[:, :2] = core.env_origins[env_id : env_id + 1, :2]
        roots.append(torch.cat((root_pos, state["root_rot"], state["root_vel"], state["root_ang_vel"]), dim=-1))
        dofs.append(torch.stack((state["dof_pos"], state["dof_vel"]), dim=-1))
    return {"root_states": torch.cat(roots), "dof_states": torch.cat(dofs)}


def _save_figure(path: Path, raw, crop, final, self_mask):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(POSE_LABELS), 4, figsize=(13, 12), constrained_layout=True)
    columns = (
        (raw, "raw 64x36 optical depth"),
        (crop, "flip + crop 36x32"),
        (final, "blur + uint8/255"),
        (self_mask, "robot first-hit mask"),
    )
    for row, label in enumerate(POSE_LABELS):
        for col, (values, title) in enumerate(columns):
            image = values[row].detach().cpu().numpy()
            axes[row, col].imshow(image, cmap="gray" if col == 3 else "viridis", vmin=0.0, vmax=1.0 if col >= 2 else 2.5)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title)
        axes[row, 0].set_ylabel(label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def run(args):
    cfg = build_ufo_mjlab_config(
        device=args.device,
        work_dir=str(args.output_dir),
        num_envs=len(POSE_LABELS),
        num_env_steps=2048,
        seed=args.seed,
        use_wandb=False,
        wandb_run_name=None,
        disable_eval_prioritization=True,
        smoke=True,
        agent="fb_depth",
        terrain_mode="flat",
        disable_dr=True,
        disable_obs_noise=True,
    )
    cfg.env.hydra_overrides.append("terrain.direct_depth.debug_terrain_reference=true")
    env, _ = cfg.env.build(num_envs=len(POSE_LABELS))
    core = env._env
    direct_cfg = RP1DirectDepthConfig()
    try:
        candidates = _select_candidates(core)
        env.reset(to_numpy=False, target_states=_target_states(core, candidates))
        for name in ("g1_direct_depth", "g1_direct_depth_terrain_reference"):
            core.mjlab_env.scene.sensors[name].update(0.0)
        core.mjlab_env.sim.sense()
        scene = raycast_ranges_to_image_plane(
            core.mjlab_env.scene.sensors["g1_direct_depth"].data.distances,
            direct_cfg.camera_config(),
        )
        terrain = raycast_ranges_to_image_plane(
            core.mjlab_env.scene.sensors["g1_direct_depth_terrain_reference"].data.distances,
            direct_cfg.camera_config(),
        )
        self_mask = scene + 1.0e-3 < terrain
        crop = scene.flip(1)[:, :, 16:48]
        final = preprocess_rp1_depth(scene, direct_cfg, enable_noise=False)
        if not torch.isfinite(scene).all() or not torch.isfinite(final).all():
            raise RuntimeError("direct-depth fixture contains non-finite pixels")
        if not torch.any(self_mask):
            raise RuntimeError("group (2,5) camera did not observe any robot first-hit pixels")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = args.output_dir / "direct_depth_five_pose.png"
        _save_figure(image_path, scene, crop, final.float() / 255.0, self_mask.float())
        np.savez_compressed(
            args.output_dir / "direct_depth_five_pose.npz",
            labels=np.asarray(POSE_LABELS),
            raw_depth=scene.cpu().numpy(),
            terrain_only_depth=terrain.cpu().numpy(),
            crop_depth=crop.cpu().numpy(),
            final_depth=final.cpu().numpy(),
            self_mask=self_mask.cpu().numpy(),
        )
        poses = {}
        for index, label in enumerate(POSE_LABELS):
            poses[label] = {
                "motion_id": candidates[label][0],
                "motion_key": core._motion_lib.curr_motion_keys[candidates[label][0]],
                "motion_time_s": candidates[label][1],
                "self_occlusion_fraction": float(self_mask[index].float().mean().item()),
                **candidates[label][3],
            }
        summary = {
            "status": "passed",
            "image": str(image_path),
            "raw_shape": list(scene.shape),
            "crop_shape": list(crop.shape),
            "final_shape": list(final.shape),
            "poses": poses,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260827)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
