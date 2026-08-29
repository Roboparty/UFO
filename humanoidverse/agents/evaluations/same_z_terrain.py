"""Deterministic same-latent evaluation across the seven RP1 terrain families."""

import collections
import math
import numbers
from typing import Any, Literal, Mapping

import numpy as np
import pydantic
import torch

from humanoidverse.terrain_transfer import clone_same_z_for_terrains, tensor_checksum
from humanoidverse.terrains.rp1_simple import RP1_TERRAIN_COMPONENT_NAMES, rp1_center_reset_profile

from ..envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
from ..nn_models import eval_mode
from .base import BaseEvalConfig, extract_model
from .humanoidverse_mjlab import _encode_motion_contexts, _expert_motion_slice

DEFAULT_LOCOMOTION_MOTIONS = (
    # Forward locomotion spanning roughly 0.15--0.73 m/s net speed.
    "walk3_subject3_clip11",
    "walk3_subject3_clip6",
    "walk3_subject3_clip4",
    "walk1_subject1_clip0",
    "walk1_subject1_clip5",
    "sprint1_subject2_clip11",
    "walk3_subject1_clip23",
    "walk3_subject4_clip1",
    # Diagonal and lateral locomotion in both directions.
    "walk3_subject5_clip12",
    "sprint1_subject4_clip20",
    "walk2_subject3_clip17",
    "sprint1_subject4_clip8",
    "run1_subject5_clip4",
    "sprint1_subject4_clip21",
    # Left- and right-turning locomotion with substantial net travel.
    "walk2_subject4_clip16",
    "sprint1_subject2_clip4",
    "walk2_subject4_clip11",
    "run2_subject1_clip16",
    "walk1_subject1_clip20",
    "walk1_subject1_clip3",
)


def make_same_z_terrain_eval_config(env_cfg: HumanoidVerseMjlabConfig, *, seed: int) -> HumanoidVerseMjlabConfig:
    """Make terrain the only varying factor in the periodic benchmark."""

    overrides = list(env_cfg.hydra_overrides)
    terrain_index = next(
        (index for index, value in enumerate(overrides) if value.startswith("terrain.terrain_type=")),
        None,
    )
    if terrain_index is None:
        raise ValueError("same-z terrain evaluation requires a configured terrain.terrain_type override")
    overrides[terrain_index] = "terrain.terrain_type=rp1_simple"
    seed_index = next(
        (index for index, value in enumerate(overrides) if value.startswith("terrain.seed=")),
        None,
    )
    if seed_index is None:
        overrides.append(f"terrain.seed={int(seed)}")
    else:
        overrides[seed_index] = f"terrain.seed={int(seed)}"
    return env_cfg.model_copy(
        update={
            "hydra_overrides": overrides,
            "seed": int(seed),
            # The benchmark horizon is the latent sequence itself. Keep the
            # simulator's administrative timeout beyond every 10 s LaFAN
            # clip so a successful final frame is not mislabeled as a fall.
            "max_episode_length_s": 30.0,
            "disable_obs_noise": True,
            "disable_domain_randomization": True,
            "evaluation_fast_path": False,
            "fixed_direct_depth_delay_frames": 0,
        }
    )


def _yaw_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def _wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _quat_mul_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for broadcast-compatible xyzw quaternions."""

    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def _rotate_reference_to_course(
    reference: dict[str, torch.Tensor],
    *,
    target_heading: float,
    minimum_distance: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    """Rotate one motion around world Z so its net travel follows the column.

    RP1 families occupy columns along world X.  This keeps every compared
    terrain on the same yaw while preventing an arbitrary LaFAN world heading
    from immediately crossing into a neighbouring terrain family.
    """

    root_pos = reference["root_pos"]
    delta_xy = root_pos[-1, :2] - root_pos[0, :2]
    distance = torch.linalg.vector_norm(delta_xy)
    if float(distance.item()) < float(minimum_distance):
        raise ValueError(
            "Same-z locomotion motion has insufficient horizontal travel: "
            f"distance={float(distance.item()):.3f}m < minimum={float(minimum_distance):.3f}m"
        )
    source_heading = torch.atan2(delta_xy[1], delta_xy[0])
    yaw = torch.as_tensor(target_heading, device=root_pos.device, dtype=root_pos.dtype) - source_heading
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotation_xy = torch.stack(
        (
            torch.stack((cos_yaw, -sin_yaw)),
            torch.stack((sin_yaw, cos_yaw)),
        )
    )

    rotated = dict(reference)
    rotated_root_pos = root_pos.clone()
    relative_xy = root_pos[:, :2] - root_pos[0, :2]
    rotated_root_pos[:, :2] = relative_xy @ rotation_xy.T + root_pos[0, :2]
    rotated["root_pos"] = rotated_root_pos

    for key in ("root_vel", "root_ang_vel"):
        value = reference[key].clone()
        value[:, :2] = value[:, :2] @ rotation_xy.T
        rotated[key] = value

    half_yaw = yaw * 0.5
    yaw_quaternion = torch.stack(
        (
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
            torch.cos(half_yaw),
        )
    )
    rotated["root_rot"] = _quat_mul_xyzw(
        yaw_quaternion.expand_as(reference["root_rot"]),
        reference["root_rot"],
    )
    return rotated


class SameZTerrainEvaluationConfig(BaseEvalConfig):
    name: Literal["SameZTerrainEvaluationConfig"] = "SameZTerrainEvaluationConfig"
    name_in_logs: str = "same_z_terrain_eval"
    affects_motion_priority: bool = False
    terrain_families: tuple[str, ...] = RP1_TERRAIN_COMPONENT_NAMES
    locomotion_motion_names: tuple[str, ...] = DEFAULT_LOCOMOTION_MOTIONS
    difficulty_rows: tuple[int, ...] = (0, 2, 4, 7, 9)
    motions_per_batch: int = 8
    context_batch_size: int = 65536
    seed: int = 4728
    minimum_reference_distance_m: float = 1.25
    minimum_progress_ratio: float = 0.5
    semantic_progress_ratio: float = 0.1
    semantic_heading_error_rad: float = math.pi / 2.0

    @pydantic.model_validator(mode="after")
    def _validate_contract(self):
        if tuple(self.terrain_families) != tuple(RP1_TERRAIN_COMPONENT_NAMES):
            raise ValueError(
                "SameZTerrainEvaluation must cover the exact seven RP1 families in their canonical order"
            )
        if not self.locomotion_motion_names or len(set(self.locomotion_motion_names)) != len(
            self.locomotion_motion_names
        ):
            raise ValueError("locomotion_motion_names must be non-empty and unique")
        if not self.difficulty_rows or any(row < 0 for row in self.difficulty_rows):
            raise ValueError("difficulty_rows must be non-empty and non-negative")
        if self.motions_per_batch < 1 or self.context_batch_size < 1:
            raise ValueError("same-z batch sizes must be positive")
        if self.minimum_reference_distance_m <= 0.0:
            raise ValueError("minimum_reference_distance_m must be positive")
        return self

    def build(self):
        return SameZTerrainEvaluation(self)


class SameZTerrainEvaluation:
    """Evaluate exact motion latents on terrain without creating training gradients."""

    def __init__(self, config: SameZTerrainEvaluationConfig) -> None:
        self.cfg = config

    def _motion_ids(self, expert_buffer) -> list[int]:
        if not hasattr(expert_buffer, "file_names"):
            raise ValueError("Same-z evaluation requires expert-buffer motion file names")
        by_name = {str(name): int(motion_id) for name, motion_id in zip(expert_buffer.file_names, expert_buffer.motion_ids)}
        missing = [name for name in self.cfg.locomotion_motion_names if name not in by_name]
        if missing:
            raise ValueError(f"Same-z locomotion subset is missing from the expert buffer: {missing[:8]}")
        return [by_name[name] for name in self.cfg.locomotion_motion_names]

    def _encode_once(self, model, expert_buffer, motion_ids: list[int], *, device: str):
        observations = [_expert_motion_slice(expert_buffer, motion_id) for motion_id in motion_ids]
        lengths = [int(observation["state"].shape[0]) for observation in observations]
        contexts = _encode_motion_contexts(
            model,
            observations,
            lengths,
            device=device,
            batch_size=self.cfg.context_batch_size,
        )
        encoded = {}
        for motion_id, context in zip(motion_ids, contexts):
            canonical = context.detach().to(device="cpu", dtype=torch.float32).contiguous()
            clones = clone_same_z_for_terrains(canonical, list(self.cfg.terrain_families))
            checksum = tensor_checksum(canonical)
            checksums = {family: tensor_checksum(value) for family, value in clones.items()}
            if set(checksums.values()) != {checksum}:
                raise AssertionError(f"same-z hash mismatch for motion_id={motion_id}: {checksums}")
            encoded[motion_id] = {
                "context": canonical,
                "clones": clones,
                "hash": checksum,
                "shape": tuple(canonical.shape),
                "dtype": str(canonical.dtype),
            }
        return encoded

    @staticmethod
    def _assign_rp1_tiles(core, family_ids: torch.Tensor, difficulty_row: int) -> None:
        terrain = core.mjlab_env.scene["terrain"]
        if tuple(core.terrain_component_names) != tuple(RP1_TERRAIN_COMPONENT_NAMES):
            raise RuntimeError(
                f"same-z evaluator expected RP1 family columns, got {core.terrain_component_names}"
            )
        if not 0 <= difficulty_row < terrain.terrain_origins.shape[0]:
            raise ValueError(
                f"difficulty row {difficulty_row} is outside [0, {terrain.terrain_origins.shape[0]})"
            )
        levels = torch.full_like(family_ids, int(difficulty_row))
        terrain.terrain_levels.copy_(levels)
        terrain.terrain_types.copy_(family_ids)
        origins = terrain.terrain_origins[levels, family_ids]
        core.env_origins.copy_(origins)

    @staticmethod
    def _motion_reference(core, motion_id: int, length: int) -> dict[str, torch.Tensor]:
        times = torch.arange(length + 1, device=core.device, dtype=torch.float32) * core.dt
        ids = torch.full((length + 1,), int(motion_id), device=core.device, dtype=torch.long)
        state = core._motion_lib.get_motion_state(ids, times)
        return {
            "root_pos": state["root_pos"].float(),
            "root_rot": state["root_rot"].float(),
            "root_vel": state["root_vel"].float(),
            "root_ang_vel": state["root_ang_vel"].float(),
            "dof_pos": state["dof_pos"].float(),
            "dof_vel": state["dof_vel"].float(),
        }

    @torch.no_grad()
    def _run_batch(
        self,
        *,
        env,
        agent,
        motion_ids: list[int],
        encoded: dict[int, dict[str, Any]],
        difficulty_row: int,
    ) -> dict[str, dict[str, Any]]:
        core = env._env
        families = tuple(self.cfg.terrain_families)
        real_motion_count = len(motion_ids)
        capacity = core.num_envs // len(families)
        if real_motion_count > capacity:
            raise ValueError("same-z motion chunk exceeds evaluator capacity")
        padded = motion_ids + [motion_ids[-1]] * (capacity - real_motion_count)
        env_motion_ids = [motion_id for motion_id in padded for _ in families]
        family_ids = torch.arange(len(families), device=core.device).repeat(capacity)
        self._assign_rp1_tiles(core, family_ids, difficulty_row)

        references = {
            motion_id: _rotate_reference_to_course(
                self._motion_reference(core, motion_id, encoded[motion_id]["context"].shape[0]),
                # Start every difficulty toward the interior of its family
                # column.  All seven terrain variants still receive exactly
                # the same yaw for a given (motion, difficulty) benchmark.
                target_heading=(
                    0.0
                    if difficulty_row < core.mjlab_env.scene["terrain"].terrain_origins.shape[0] / 2
                    else math.pi
                ),
                minimum_distance=self.cfg.minimum_reference_distance_m,
            )
            for motion_id in set(padded)
        }
        initial_dof = torch.stack([references[motion_id]["dof_pos"][0] for motion_id in env_motion_ids])
        initial_dof_vel = torch.stack([references[motion_id]["dof_vel"][0] for motion_id in env_motion_ids])
        initial_root_rot = torch.stack([references[motion_id]["root_rot"][0] for motion_id in env_motion_ids])
        initial_root_vel = torch.stack([references[motion_id]["root_vel"][0] for motion_id in env_motion_ids])
        initial_root_ang_vel = torch.stack(
            [references[motion_id]["root_ang_vel"][0] for motion_id in env_motion_ids]
        )
        reference_clearance = torch.stack(
            [references[motion_id]["root_pos"][0, 2] for motion_id in env_motion_ids]
        )
        root_pos = core.env_origins.clone()
        root_pos[:, 2] += reference_clearance
        target_states = {
            "dof_states": torch.stack((initial_dof, initial_dof_vel), dim=-1),
            "root_states": torch.cat(
                (root_pos, initial_root_rot, initial_root_vel, initial_root_ang_vel),
                dim=-1,
            ),
        }
        observation, _ = env.reset(target_states=target_states, to_numpy=False)
        grouped_dof_pos = core.dof_pos.reshape(capacity, len(families), -1)
        grouped_dof_vel = core.dof_vel.reshape(capacity, len(families), -1)
        grouped_root = core.robot_root_states.reshape(capacity, len(families), -1)
        grouped_origin = core.env_origins.reshape(capacity, len(families), -1)
        initial_invariants = {
            "q0": grouped_dof_pos,
            "qdot0": grouped_dof_vel,
            "root_xy_local": grouped_root[..., :2] - grouped_origin[..., :2],
            "root_quaternion": grouped_root[..., 3:7],
            "root_velocity": grouped_root[..., 7:10],
            "root_angular_velocity": grouped_root[..., 10:13],
        }
        for name, value in initial_invariants.items():
            reference_value = value[:, :1].expand_as(value)
            if not torch.allclose(value, reference_value, atol=1.0e-6, rtol=0.0):
                max_error = float((value - reference_value).abs().amax().item())
                raise AssertionError(
                    f"Same-z terrain reset changed {name} across terrain families: max_error={max_error:.3e}"
                )
        if core._direct_depth_runtime is None:
            raise RuntimeError("Same-z terrain evaluation requires direct depth")
        if torch.count_nonzero(core._direct_depth_runtime.delay_frames).item() != 0:
            raise AssertionError("Same-z evaluator must run with depth delay fixed to zero")
        if torch.count_nonzero(core._ctrl_delay_steps).item() != 0:
            raise AssertionError("Same-z deterministic benchmark must run with motor delay fixed to zero")

        initial_policy_root = core.robot_root_states[:, :7].clone()
        last_valid_root = initial_policy_root.clone()
        invalid = torch.zeros(core.num_envs, device=core.device, dtype=torch.bool)
        context_lengths = torch.tensor(
            [encoded[motion_id]["context"].shape[0] for motion_id in env_motion_ids],
            device=core.device,
            dtype=torch.long,
        )
        rollout_contexts = [
            encoded[motion_id]["clones"][families[family_id]].to(core.device).contiguous()
            for motion_id, family_id in zip(env_motion_ids, family_ids.tolist())
        ]
        for motion_id, rollout_context in zip(env_motion_ids, rollout_contexts):
            if tensor_checksum(rollout_context) != encoded[motion_id]["hash"]:
                raise AssertionError(
                    f"same-z device transfer changed bytes for motion_id={motion_id}"
                )
        max_length = int(context_lengths.max().item())
        for step in range(max_length):
            z_step = torch.stack(
                [
                    context[min(step, context.shape[0] - 1)]
                    for context in rollout_contexts
                ]
            )
            action = agent.act(observation, z_step, mean=True)
            observation, _, terminated, truncated, _ = env.step(action, to_numpy=False)
            active = step < context_lengths
            done = terminated.reshape(-1).bool() | truncated.reshape(-1).bool()
            finite = torch.isfinite(core.robot_root_states).all(dim=-1)
            valid_step = active & ~invalid & ~done & finite
            last_valid_root[valid_step] = core.robot_root_states[valid_step, :7]
            invalid |= active & (done | ~finite)

        rows: dict[str, dict[str, Any]] = {}
        for motion_index, motion_id in enumerate(motion_ids):
            reference = references[motion_id]
            reference_delta = reference["root_pos"][-1, :2] - reference["root_pos"][0, :2]
            reference_distance = torch.linalg.vector_norm(reference_delta).clamp_min(1.0e-6)
            direction = reference_delta / reference_distance
            reference_heading_delta = _wrap_angle(
                _yaw_xyzw(reference["root_rot"][-1]) - _yaw_xyzw(reference["root_rot"][0])
            )
            reference_travel_relative_to_initial_heading = _wrap_angle(
                torch.atan2(reference_delta[1], reference_delta[0])
                - _yaw_xyzw(reference["root_rot"][0])
            )
            for family_id, family in enumerate(families):
                evaluation_family, initial_vertical_direction = rp1_center_reset_profile(family)
                env_index = motion_index * len(families) + family_id
                policy_delta = last_valid_root[env_index, :2] - initial_policy_root[env_index, :2]
                signed_progress = torch.dot(policy_delta, direction)
                progress_ratio = signed_progress / reference_distance
                policy_heading_delta = _wrap_angle(
                    _yaw_xyzw(last_valid_root[env_index, 3:7])
                    - _yaw_xyzw(initial_policy_root[env_index, 3:7])
                )
                relative_heading_error = _wrap_angle(policy_heading_delta - reference_heading_delta)
                survived = not bool(invalid[env_index].item())
                traversal = survived and float(progress_ratio.item()) >= self.cfg.minimum_progress_ratio
                semantic_drift = (
                    not survived
                    or float(progress_ratio.item()) < self.cfg.semantic_progress_ratio
                    or abs(float(relative_heading_error.item())) > self.cfg.semantic_heading_error_rad
                )
                z_meta = encoded[motion_id]
                rollout_context = rollout_contexts[env_index]
                if tensor_checksum(rollout_context) != z_meta["hash"]:
                    raise AssertionError(f"same-z tensor changed during rollout for motion={motion_id} family={family}")
                key = f"motion={motion_id}/terrain={evaluation_family}/difficulty={difficulty_row}"
                rows[key] = {
                    "motion_id": int(motion_id),
                    "terrain_family": evaluation_family,
                    "terrain_asset_family": family,
                    "initial_vertical_direction": initial_vertical_direction,
                    "reset_region": "tile_center",
                    "difficulty_row": int(difficulty_row),
                    "z_hash": z_meta["hash"],
                    "z_shape": str(z_meta["shape"]),
                    "z_dtype": z_meta["dtype"],
                    "depth_delay_frames": 0,
                    "survival": survived,
                    "traversal": traversal,
                    "signed_progress": float(signed_progress.item()),
                    "reference_progress": float(reference_distance.item()),
                    "reference_heading_change_rad": float(reference_heading_delta.item()),
                    "reference_travel_relative_to_initial_heading_rad": float(
                        reference_travel_relative_to_initial_heading.item()
                    ),
                    "progress_ratio": float(progress_ratio.item()),
                    "relative_heading_error_rad": float(relative_heading_error.item()),
                    "semantic_drift": semantic_drift,
                }
        return rows

    def run(
        self,
        *,
        timestep,
        agent_or_model,
        logger,
        base_env_config: HumanoidVerseMjlabConfig | None = None,
        motion_lib=None,
        expert_buffer=None,
        write_outputs: bool = True,
        **kwargs,
    ):
        if base_env_config is None or motion_lib is None or expert_buffer is None:
            raise ValueError("Same-z evaluation requires base_env_config, motion_lib, and expert_buffer")
        model = extract_model(agent_or_model)
        motion_ids = self._motion_ids(expert_buffer)
        load_all = getattr(motion_lib, "load_all_motions", None)
        env = None
        metrics: dict[str, dict[str, Any]] = {}
        try:
            if callable(load_all) and not bool(getattr(motion_lib, "all_motions_loaded", False)):
                load_all()
            encoded = self._encode_once(model, expert_buffer, motion_ids, device=str(model.device))
            eval_cfg = make_same_z_terrain_eval_config(base_env_config, seed=self.cfg.seed)
            capacity = min(self.cfg.motions_per_batch, len(motion_ids))
            env, _ = eval_cfg.build(
                num_envs=capacity * len(self.cfg.terrain_families),
                motion_lib=motion_lib,
            )
            env._env.is_evaluating = True
            with eval_mode(model):
                for start in range(0, len(motion_ids), capacity):
                    chunk = motion_ids[start : start + capacity]
                    for difficulty_row in self.cfg.difficulty_rows:
                        metrics.update(
                            self._run_batch(
                                env=env,
                                agent=agent_or_model,
                                motion_ids=chunk,
                                encoded=encoded,
                                difficulty_row=difficulty_row,
                            )
                        )
        finally:
            if env is not None:
                env.close()
            load_training = getattr(motion_lib, "load_motions_for_training", None)
            if callable(load_training):
                load_training()
        if write_outputs:
            self.record_results(metrics, timestep=timestep, logger=logger)
        return metrics, self.summarize(metrics)

    @staticmethod
    def summarize(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        aggregates: dict[str, list[float]] = collections.defaultdict(list)
        by_family: dict[str, dict[str, list[float]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        for metric in metrics.values():
            family = str(metric["terrain_family"])
            for key, value in metric.items():
                if isinstance(value, numbers.Number) and np.isfinite(value):
                    aggregates[key].append(float(value))
                    by_family[family][key].append(float(value))
        output = {key: float(np.mean(values)) for key, values in aggregates.items() if values}
        for family, family_metrics in by_family.items():
            for key, values in family_metrics.items():
                if values:
                    output[f"{family}/{key}"] = float(np.mean(values))
        return output

    @staticmethod
    def record_results(metrics: Mapping[str, Mapping[str, Any]], *, timestep: int, logger) -> None:
        if logger is None:
            return
        rows = []
        for case, metric in metrics.items():
            row = dict(metric)
            row["case"] = case
            row["timestep"] = int(timestep)
            rows.append(row)
        logger.log_many(rows)
