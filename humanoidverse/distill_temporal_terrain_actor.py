"""On-policy privileged-map distillation for a temporal-terrain Actor.

The student closes the loop with a frozen temporal terrain completion model.
At every student-visited state, a frozen copy of the original Actor receives
the simulator GT terrain map and supplies the deterministic target action.
Only the student Actor is optimized; the PBFM representation and perception
model remain frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.depth_terrain_evaluation import build_depth_evaluation_env, synchronize_depth_and_gt
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg, replace_hydra_override
from humanoidverse.perception.depth_camera import DepthCameraConfig, depth_frame_from_raycast
from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter
from humanoidverse.perception.temporal_terrain import TemporalTerrainCompletion, TerrainHistoryBuffer
from humanoidverse.terrain_perception_closed_loop import _load_latent, _load_perception
from humanoidverse.terrain_transfer import tensor_checksum
from humanoidverse.utils.torch_utils import calc_heading_quat, get_euler_xyz


@dataclass(frozen=True)
class DistillationConfig:
    high_stairs_envs: int = 384
    mixed_envs: int = 256
    training_steps: int = 5000
    learning_rate: float = 3.0e-5
    anchor_weight: float = 1.0
    max_grad_norm: float = 1.0
    high_stairs_min_difficulty: float = 5.0 / 9.0
    checkpoint_every: int = 500
    milestone_steps: tuple[int, ...] = (500, 1000, 2000, 5000)
    log_every: int = 25
    seed: int = 6840
    max_episode_length_s: float = 20.0

    def validate(self) -> None:
        if min(self.high_stairs_envs, self.mixed_envs, self.training_steps) <= 0:
            raise ValueError("environment counts and training_steps must be positive")
        if self.learning_rate <= 0.0 or self.anchor_weight < 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("invalid optimization hyperparameters")
        if not 0.0 <= self.high_stairs_min_difficulty <= 1.0:
            raise ValueError("high_stairs_min_difficulty must be in [0, 1]")
        if min(self.checkpoint_every, self.log_every) <= 0:
            raise ValueError("checkpoint_every and log_every must be positive")
        if (
            not self.milestone_steps
            or any(step <= 0 for step in self.milestone_steps)
            or tuple(sorted(set(self.milestone_steps))) != self.milestone_steps
        ):
            raise ValueError("milestone_steps must be unique, positive, and increasing")
        if self.max_episode_length_s <= 0.0:
            raise ValueError("max_episode_length_s must be positive")


def module_checksum(module: nn.Module) -> str:
    """Return a deterministic SHA256 over module state without retaining copies."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def configure_actor_only_training(model) -> nn.Module:
    """Freeze a loaded PBFM model and return its frozen teacher Actor copy."""
    model.eval().requires_grad_(False)
    teacher_actor = copy.deepcopy(model._actor).eval().requires_grad_(False)
    model._actor.train().requires_grad_(True)
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actor_parameters = {id(parameter) for parameter in model._actor.parameters()}
    if trainable != actor_parameters:
        raise AssertionError("only student Actor parameters may require gradients")
    return teacher_actor


def actor_mean(actor: nn.Module, normalized_obs: dict[str, torch.Tensor], z: torch.Tensor, std: float) -> torch.Tensor:
    return actor(normalized_obs, z, std).mean.float()


def actor_distillation_loss(
    student_temporal: torch.Tensor,
    teacher_gt: torch.Tensor,
    student_gt: torch.Tensor,
    *,
    anchor_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if student_temporal.shape != teacher_gt.shape or student_gt.shape != teacher_gt.shape:
        raise ValueError("student and teacher action tensors must have identical shapes")
    deploy = F.mse_loss(student_temporal, teacher_gt)
    anchor = F.mse_loss(student_gt, teacher_gt)
    return deploy + float(anchor_weight) * anchor, deploy, anchor


def _concat_observations(observations: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("at least one observation dictionary is required")
    keys = tuple(observations[0])
    if any(tuple(observation) != keys for observation in observations[1:]):
        raise ValueError("distillation streams have different observation schemas")
    return {key: torch.cat([observation[key] for observation in observations], dim=0) for key in keys}


def _frozen_modules(model, perception: nn.Module, teacher_actor: nn.Module) -> dict[str, nn.Module]:
    modules = {"teacher_actor": teacher_actor, "perception": perception, "normalizer": model._obs_normalizer}
    for name, module in model.named_children():
        if name not in {"_actor", "_obs_normalizer"}:
            modules[f"pbfm{name}"] = module
    return modules


def _verify_frozen_checksums(modules: dict[str, nn.Module], expected: dict[str, str]) -> None:
    actual = {name: module_checksum(module) for name, module in modules.items()}
    changed = {name: (expected[name], checksum) for name, checksum in actual.items() if checksum != expected[name]}
    if changed:
        raise AssertionError(f"frozen module state changed: {changed}")


class _DistillationStream:
    def __init__(
        self,
        *,
        name: str,
        env_config,
        num_envs: int,
        camera: DepthCameraConfig,
        perception_config: dict[str, Any],
        device: str,
        fixed_latent: torch.Tensor | None,
        sample_z,
    ) -> None:
        self.name = name
        self.num_envs = num_envs
        self.device = device
        self.wrapped_env, _ = build_depth_evaluation_env(env_config, num_envs=num_envs, camera=camera)
        self.core = self.wrapped_env._env
        self.adapter = DepthTerrainAdapter(camera.intrinsics(), camera.height, camera.width).to(device)
        self.camera = camera
        self.history = TerrainHistoryBuffer(
            batch_size=num_envs,
            time_steps=int(perception_config["sequence_steps"]),
            proprio_dim=int(perception_config["proprio_dim"]),
            device=device,
        )
        self.history_seconds = float(perception_config["history_seconds"])
        self.episode_time = torch.zeros(num_envs, device=device)
        self.pending_reset = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.fixed_latent = fixed_latent
        self.sample_z = sample_z
        self.z = fixed_latent.expand(num_envs, -1).clone() if fixed_latent is not None else sample_z(num_envs)
        self.observation, _ = self.wrapped_env.reset(to_numpy=False)
        self.reset_count = 0

    @torch.no_grad()
    def observe(self, perception: TemporalTerrainCompletion) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        synchronize_depth_and_gt(self.core, self.camera.name)
        frame = depth_frame_from_raycast(self.core.mjlab_env.scene.sensors[self.camera.name], self.camera)
        heading = calc_heading_quat(self.core.base_quat, w_last=True)
        partial, visible = self.adapter(
            frame.depth_z,
            frame.camera_pos_w,
            frame.camera_optical_quat_w,
            self.core.robot_root_states[:, :3],
            heading,
        )
        gt = self.core._terrain_actor_obs().clone()
        yaw = get_euler_xyz(self.core.base_quat, w_last=True)[2]
        self.history.reset(self.pending_reset)
        self.history.append(
            partial_map=partial,
            visible_mask=visible,
            pelvis_pos_w=self.core.robot_root_states[:, :3],
            heading_yaw_w=yaw,
            timestamp_s=self.episode_time,
            proprio=self.observation["state"],
        )
        warped = self.history.warp(history_seconds=self.history_seconds, interpolation="bilinear")
        predicted = perception(warped, proprio=self.history.proprio).completed_clearance
        if not torch.isfinite(predicted).all() or not torch.isfinite(gt).all():
            raise RuntimeError(f"{self.name} produced non-finite temporal or GT terrain map")
        temporal_observation = dict(self.observation)
        temporal_observation["terrain_actor"] = predicted
        gt_observation = dict(self.observation)
        gt_observation["terrain_actor"] = gt
        return temporal_observation, gt_observation

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> dict[str, float]:
        self.observation, _reward, terminated, truncated, info = self.wrapped_env.step(action, to_numpy=False)
        reset = torch.as_tensor(terminated, device=self.device).bool() | torch.as_tensor(
            truncated, device=self.device
        ).bool()
        reset_count = int(reset.sum().item())
        self.reset_count += reset_count
        self.episode_time += self.core.dt
        self.episode_time[reset] = 0.0
        self.pending_reset = reset
        if self.fixed_latent is None and reset_count:
            self.z[reset] = self.sample_z(reset_count)
        impact = info.get("aux_rewards", {}).get("penalty_body_impact")
        return {
            "resets": float(reset_count),
            "body_impact_mean": float(torch.as_tensor(impact).float().mean()) if impact is not None else 0.0,
        }

    def close(self) -> None:
        self.wrapped_env.close()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_training_state(
    *,
    output_dir: Path,
    step: int,
    model,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    immutable_milestone: bool = False,
) -> None:
    actor_state = {name: value.detach().cpu() for name, value in model._actor.state_dict().items()}
    _atomic_torch_save(
        {"step": step, "actor": actor_state, "optimizer": optimizer.state_dict(), "metadata": metadata},
        output_dir / "actor_latest.pt",
    )
    if immutable_milestone:
        milestone_path = output_dir / "milestones" / f"actor_step_{step:06d}.pt"
        if milestone_path.exists():
            raise FileExistsError(f"refusing to overwrite immutable milestone: {milestone_path}")
        _atomic_torch_save(
            {"step": step, "actor": actor_state, "metadata": metadata},
            milestone_path,
        )
    (output_dir / "status.json").write_text(
        json.dumps({**metadata, "completed_training_steps": step}, indent=2, sort_keys=True) + "\n"
    )


def _materialize_model(*, source_model_folder: Path, output_dir: Path, model) -> Path:
    deploy_dir = output_dir / "distilled_model"
    checkpoint_dir = deploy_dir / "checkpoint"
    model_dir = checkpoint_dir / "model"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model_folder / "config.json", deploy_dir / "config.json")
    source_status = source_model_folder / "checkpoint" / "train_status.json"
    if source_status.exists():
        shutil.copy2(source_status, checkpoint_dir / "train_status.json")
    model.eval()
    model.save(str(model_dir))
    return deploy_dir


def distill_actor(
    *,
    model_folder: Path,
    perception_checkpoint: Path,
    latent_path: Path,
    output_dir: Path,
    config: DistillationConfig,
    camera: DepthCameraConfig,
    device: str,
    materialize_model: bool,
    resume: bool,
) -> dict[str, Any]:
    config.validate()
    model_folder = model_folder.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=checkpoint_load_device(device))
    model.to(device)
    teacher_actor = configure_actor_only_training(model)
    perception, perception_payload = _load_perception(perception_checkpoint, device)
    perception.eval().requires_grad_(False)
    forward_latent, latent_payload = _load_latent(latent_path, device)

    optimizer = torch.optim.AdamW(model._actor.parameters(), lr=config.learning_rate, weight_decay=0.0)
    start_step = 0
    latest_path = output_dir / "actor_latest.pt"
    if resume and latest_path.exists():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        model._actor.load_state_dict(saved["actor"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])

    env_config, _ = load_mjlab_env_cfg(
        model_folder,
        data_path=None,
        robot_config=None,
        device=device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=config.max_episode_length_s,
    )
    high_overrides = replace_hydra_override(list(env_config.hydra_overrides), "terrain.terrain_type", "stairs_up")
    high_overrides = replace_hydra_override(
        high_overrides,
        "terrain.difficulty_range",
        f"[{config.high_stairs_min_difficulty},1.0]",
    )
    high_config = env_config.model_copy(update={"seed": config.seed, "hydra_overrides": high_overrides})
    mixed_config = env_config.model_copy(
        update={
            "seed": config.seed + 1,
            "hydra_overrides": replace_hydra_override(
                list(env_config.hydra_overrides), "terrain.terrain_type", "mixed"
            ),
        }
    )
    sample_z = lambda count: model.sample_z(count, device=device)
    high_stream = _DistillationStream(
        name="high_stairs",
        env_config=high_config,
        num_envs=config.high_stairs_envs,
        camera=camera,
        perception_config=perception_payload["config"],
        device=device,
        fixed_latent=forward_latent,
        sample_z=sample_z,
    )
    mixed_stream = _DistillationStream(
        name="mixed",
        env_config=mixed_config,
        num_envs=config.mixed_envs,
        camera=camera,
        perception_config=perception_payload["config"],
        device=device,
        fixed_latent=None,
        sample_z=sample_z,
    )
    streams = (high_stream, mixed_stream)
    frozen_modules = _frozen_modules(model, perception, teacher_actor)
    frozen_before = {name: module_checksum(module) for name, module in frozen_modules.items()}
    actor_before = module_checksum(model._actor)
    metadata = {
        "source_model_folder": str(model_folder),
        "perception_checkpoint": str(perception_checkpoint.expanduser().resolve()),
        "perception_epoch": int(perception_payload["epoch"]),
        "forward_latent": str(latent_path.expanduser().resolve()),
        "z_checksum": latent_payload["z_checksum"],
        "distillation": asdict(config),
        "camera": asdict(camera),
        "teacher": "frozen source Actor with GT terrain_actor",
        "student": "source Actor initialized, temporal predicted terrain_actor, deterministic mean action",
        "loss": "MSE(student_temporal, teacher_gt) + anchor_weight * MSE(student_gt, teacher_gt)",
        "trainable_module": "student Actor only",
    }
    loss_history: list[dict[str, float]] = []
    high_count = config.high_stairs_envs
    try:
        for step in range(start_step + 1, config.training_steps + 1):
            temporal_raw: list[dict[str, torch.Tensor]] = []
            gt_raw: list[dict[str, torch.Tensor]] = []
            for stream in streams:
                temporal_observation, gt_observation = stream.observe(perception)
                temporal_raw.append(temporal_observation)
                gt_raw.append(gt_observation)
            normalized_temporal = model._normalize(_concat_observations(temporal_raw))
            normalized_gt = model._normalize(_concat_observations(gt_raw))
            z = torch.cat([stream.z for stream in streams], dim=0)
            amp_enabled = bool(model.cfg.amp) and torch.device(device).type == "cuda"
            with torch.no_grad(), torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                teacher_gt = actor_mean(teacher_actor, normalized_gt, z, model.cfg.actor_std)
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                student_temporal = actor_mean(model._actor, normalized_temporal, z, model.cfg.actor_std)
                student_gt = actor_mean(model._actor, normalized_gt, z, model.cfg.actor_std)
                loss, deploy_loss, anchor_loss = actor_distillation_loss(
                    student_temporal,
                    teacher_gt,
                    student_gt,
                    anchor_weight=config.anchor_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite distillation loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model._actor.parameters(), config.max_grad_norm)
            optimizer.step()
            actions = student_temporal.detach()
            high_stats = high_stream.step(actions[:high_count])
            mixed_stats = mixed_stream.step(actions[high_count:])
            row = {
                "step": float(step),
                "loss": float(loss.detach()),
                "deploy_loss": float(deploy_loss.detach()),
                "anchor_loss": float(anchor_loss.detach()),
                "high_deploy_loss": float(F.mse_loss(student_temporal[:high_count], teacher_gt[:high_count]).detach()),
                "mixed_deploy_loss": float(F.mse_loss(student_temporal[high_count:], teacher_gt[high_count:]).detach()),
                "grad_norm": float(grad_norm),
                "high_resets": high_stats["resets"],
                "mixed_resets": mixed_stats["resets"],
                "high_body_impact": high_stats["body_impact_mean"],
                "mixed_body_impact": mixed_stats["body_impact_mean"],
            }
            loss_history.append(row)
            if step % config.log_every == 0 or step == start_step + 1:
                print(json.dumps(row, sort_keys=True), flush=True)
            is_milestone = step in config.milestone_steps
            if step % config.checkpoint_every == 0 or is_milestone:
                _save_training_state(
                    output_dir=output_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    metadata=metadata,
                    immutable_milestone=is_milestone,
                )
        actor_after = module_checksum(model._actor)
        if config.training_steps > start_step and actor_after == actor_before:
            raise AssertionError("student Actor did not change during distillation")
        _verify_frozen_checksums(frozen_modules, frozen_before)
        if tensor_checksum(forward_latent) != latent_payload["z_checksum"]:
            raise AssertionError("fixed forward latent changed during distillation")
        metadata["student_actor_checksum_before"] = actor_before
        metadata["student_actor_checksum_after"] = actor_after
        metadata["frozen_module_checksums"] = frozen_before
        _save_training_state(
            output_dir=output_dir,
            step=config.training_steps,
            model=model,
            optimizer=optimizer,
            metadata=metadata,
        )
        (output_dir / "metrics.json").write_text(json.dumps(loss_history, indent=2, sort_keys=True) + "\n")
        if materialize_model:
            metadata["distilled_model_folder"] = str(
                _materialize_model(source_model_folder=model_folder, output_dir=output_dir, model=model)
            )
        (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    finally:
        for stream in streams:
            stream.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--perception-checkpoint", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--high-stairs-envs", type=int, default=384)
    parser.add_argument("--mixed-envs", type=int, default=256)
    parser.add_argument("--training-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--high-stairs-min-difficulty", type=float, default=5.0 / 9.0)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--milestone-steps", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=6840)
    parser.add_argument("--max-episode-length-s", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--materialize-model", action=argparse.BooleanOptionalAction, default=True)
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
    config = DistillationConfig(
        high_stairs_envs=args.high_stairs_envs,
        mixed_envs=args.mixed_envs,
        training_steps=args.training_steps,
        learning_rate=args.learning_rate,
        anchor_weight=args.anchor_weight,
        max_grad_norm=args.max_grad_norm,
        high_stairs_min_difficulty=args.high_stairs_min_difficulty,
        checkpoint_every=args.checkpoint_every,
        milestone_steps=tuple(args.milestone_steps),
        log_every=args.log_every,
        seed=args.seed,
        max_episode_length_s=args.max_episode_length_s,
    )
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
    summary = distill_actor(
        model_folder=args.model_folder,
        perception_checkpoint=args.perception_checkpoint,
        latent_path=args.latent,
        output_dir=args.output_dir,
        config=config,
        camera=camera,
        device=args.device,
        materialize_model=args.materialize_model,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
