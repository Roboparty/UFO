"""Offline diagnostics for direct-depth Actor branch interference and visual use.

The command is intentionally read-only with respect to a training run.  It
loads a frozen model plus memory-mapped replay files, measures the gradients
that the FB, auxiliary, and canonical-plane discriminator-value branches send
to the shared Actor, and performs paired depth counterfactuals at identical
RP1 stair states.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.buffers.trajectory import (
    TrajectoryDictBuffer,
    find_start_stop_traj,
    get_idxs,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.train import build_ufo_mjlab_config
from humanoidverse.training.workspace import make_canonical_plane_training_config
from humanoidverse.utils.torch_utils import quat_rotate

DEPTH_HISTORY_OFFSETS = (35, 30, 25, 20, 15, 10, 5, 0)
OBSERVATION_PREFIX = "observation-"
STAIRS_COLUMNS = (2, 3)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


class MemoryMappedTrajectoryReplay:
    """Read only the sampled points from an uncompressed checkpoint HDF5 file."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.config = _load_json(self.path / "config.json")
        self.h5_path = self.path / "buffer.hdf5"
        self.capacity = int(self.config["capacity"])
        self.cursor = int(self.config["_idx"])
        self.is_full = bool(self.config["_is_full"])
        self.end_key = str(self.config["end_key"])
        self.depth_history_offsets = tuple(int(value) for value in self.config.get("depth_history_offsets", DEPTH_HISTORY_OFFSETS))
        if self.depth_history_offsets != DEPTH_HISTORY_OFFSETS:
            raise ValueError(f"Expected the frozen RP1 depth offsets {DEPTH_HISTORY_OFFSETS}, got {self.depth_history_offsets}")

        self.arrays: dict[str, np.memmap] = {}
        with h5py.File(self.h5_path, "r") as h5:
            for name, dataset in h5.items():
                if dataset.compression is not None or dataset.chunks is not None:
                    raise ValueError(f"Memory-mapped replay requires contiguous uncompressed data: {name}")
                offset = int(dataset.id.get_offset())
                if offset < 0:
                    raise ValueError(f"HDF5 dataset {name} has no file offset")
                self.arrays[name] = np.memmap(
                    self.h5_path,
                    dtype=dataset.dtype,
                    mode="r",
                    offset=offset,
                    shape=dataset.shape,
                    order="C",
                )
        if self.end_key not in self.arrays:
            raise KeyError(f"Replay is missing end key {self.end_key!r}")
        self.storage_length = int(self.arrays[self.end_key].shape[0])
        if self.is_full and self.storage_length != self.capacity:
            raise ValueError("A full replay must serialize exactly capacity time slots")
        self.observation_keys = tuple(name.removeprefix(OBSERVATION_PREFIX) for name in self.arrays if name.startswith(OBSERVATION_PREFIX))
        self.aux_reward_keys = tuple(name.removeprefix("aux_rewards-") for name in self.arrays if name.startswith("aux_rewards-"))
        self._prepare_trajectory_index()

    def _prepare_trajectory_index(self) -> None:
        done = torch.from_numpy(np.asarray(self.arrays[self.end_key]).copy()).bool()
        done = done.reshape(*done.shape[:2], -1).any(dim=-1)
        self.start_idx, _stop_idx, self.lengths = find_start_stop_traj(
            done,
            at_capacity=self.is_full,
            cursor=self.cursor - 1 if self.is_full else None,
        )

    def sample_indices(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        valid = self.lengths >= 2
        if not torch.any(valid):
            raise RuntimeError("Replay has no trajectories with a valid successor state")
        indices = get_idxs(
            seq_length=1,
            num_slices=int(batch_size),
            lengths=self.lengths[valid],
            start_idx=self.start_idx[valid],
            storage_length=self.capacity,
            priorities=None,
        ).to(torch.long)
        return (
            indices[:, 0].cpu().numpy(),
            indices[:, 1].cpu().numpy(),
        )

    def _take(self, key: str, time_idx: np.ndarray, env_idx: np.ndarray) -> torch.Tensor:
        if key not in self.arrays:
            raise KeyError(f"Replay field {key!r} is unavailable in {self.path}")
        return torch.from_numpy(np.asarray(self.arrays[key][time_idx, env_idx]).copy())

    def available_depth_history(self, time_idx: np.ndarray, env_idx: np.ndarray) -> np.ndarray:
        max_lookback = max(self.depth_history_offsets)
        logical_start = self.cursor if self.is_full else 0
        available = np.remainder(time_idx - logical_start, self.capacity)
        available = np.minimum(available, max_lookback)
        lookbacks = np.arange(max_lookback + 1, dtype=np.int64)
        lookback_time = np.remainder(time_idx[:, None] - lookbacks[None, :], self.capacity)
        lookback_env = np.broadcast_to(env_idx[:, None], lookback_time.shape)
        boundary = np.asarray(self.arrays[self.end_key][lookback_time, lookback_env])
        boundary = boundary.reshape(boundary.shape[0], boundary.shape[1], -1).any(axis=-1)
        if "terminated" in self.arrays:
            terminated = np.asarray(self.arrays["terminated"][lookback_time, lookback_env])
            boundary |= terminated.reshape(terminated.shape[0], terminated.shape[1], -1).any(axis=-1)
        sentinel = np.full_like(lookback_time, max_lookback + 1)
        nearest = np.where(boundary, lookbacks[None, :], sentinel).min(axis=1)
        return np.minimum(available, np.minimum(nearest, max_lookback))

    def depth_history(self, time_idx: np.ndarray, env_idx: np.ndarray) -> torch.Tensor:
        available = self.available_depth_history(time_idx, env_idx)
        offsets = np.asarray(self.depth_history_offsets, dtype=np.int64)
        effective = np.minimum(offsets[None, :], available[:, None])
        source_time = np.remainder(time_idx[:, None] - effective, self.capacity)
        source_env = np.broadcast_to(env_idx[:, None], source_time.shape)
        values = np.asarray(self.arrays[f"{OBSERVATION_PREFIX}depth_image"][source_time, source_env]).copy()
        return torch.from_numpy(values)

    def observation(self, time_idx: np.ndarray, env_idx: np.ndarray) -> dict[str, torch.Tensor]:
        output = {key: self._take(f"{OBSERVATION_PREFIX}{key}", time_idx, env_idx) for key in self.observation_keys if key != "depth_image"}
        output["depth_image"] = self.depth_history(time_idx, env_idx)
        return output

    def sample(self, batch_size: int) -> dict[str, Any]:
        time_idx, env_idx = self.sample_indices(batch_size)
        next_time = np.remainder(time_idx + 1, self.capacity)
        next_fields: dict[str, Any] = {"observation": self.observation(next_time, env_idx)}
        for flag in ("terminated", "truncated"):
            if flag in self.arrays:
                next_fields[flag] = self._take(flag, next_time, env_idx)
        return {
            "observation": self.observation(time_idx, env_idx),
            "z": self._take("z", time_idx, env_idx),
            "action": self._take("action", time_idx, env_idx),
            "aux_rewards": {
                key: self._take(f"aux_rewards-{key}", time_idx, env_idx)
                for key in self.aux_reward_keys
            },
            "next": next_fields,
            "sample_time_idx": torch.from_numpy(time_idx.copy()),
            "sample_env_idx": torch.from_numpy(env_idx.copy()),
        }


def _find_expert_cache(cache_root: Path) -> Path:
    candidates = sorted(cache_root.expanduser().resolve().glob("*/expert_buffer.pt"))
    if not candidates:
        raise FileNotFoundError(f"No expert_buffer.pt found below {cache_root}")
    if len(candidates) > 1:
        raise RuntimeError("Multiple expert caches exist; pass --expert-cache explicitly: " + ", ".join(str(path) for path in candidates))
    return candidates[0]


def _load_expert_buffer(path: Path) -> TrajectoryDictBuffer:
    payload = torch.load(
        path.expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = payload.get("buffer", payload)
    if not isinstance(state, dict) or "storage" not in state:
        raise ValueError(f"Invalid expert replay cache: {path}")
    return TrajectoryDictBuffer.from_cache_state_dict(state, device="cpu")


def _to_device(tree: Any, device: str) -> Any:
    return tree_map(lambda value: value.to(device, non_blocking=True), tree)


def _pessimistic_value(predictions: torch.Tensor, penalty: float) -> torch.Tensor:
    mean = predictions.mean(dim=0)
    left = predictions.unsqueeze(0)
    right = predictions.unsqueeze(1)
    scale = predictions.shape[0] ** 2 - predictions.shape[0]
    uncertainty = (left - right).abs().sum(dim=(0, 1)) / scale
    return mean - float(penalty) * uncertainty


@torch.no_grad()
def _encode_expert(model, expert_next_obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    encoded = model._backward_map(expert_next_obs)
    return model.project_z(encoded)


@torch.no_grad()
def _sample_mixed_z(
    model,
    train_next_obs: Mapping[str, torch.Tensor],
    expert_z: torch.Tensor,
    *,
    p_goal: float,
    p_expert: float,
) -> torch.Tensor:
    batch_size = expert_z.shape[0]
    z = model.sample_z(batch_size, device=model.device)
    probabilities = torch.tensor(
        [p_goal, p_expert, 1.0 - p_goal - p_expert],
        device=z.device,
        dtype=torch.float32,
    )
    choice = torch.multinomial(probabilities, batch_size, replacement=True).reshape(-1, 1)
    permutation = torch.randperm(batch_size, device=z.device)
    goals = model.project_z(model._backward_map(tree_map(lambda value: value[permutation], train_next_obs)))
    z = torch.where(choice == 0, goals, z)
    permutation = torch.randperm(batch_size, device=z.device)
    return torch.where(choice == 1, expert_z[permutation], z)


def _actor_parameter_groups(actor: torch.nn.Module) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    required = ("depth_encoder", "embed_s", "embed_z", "fusion", "policy")
    missing = [name for name in required if not hasattr(actor, name)]
    if missing:
        raise TypeError(f"Expected a DirectDepthActor; missing modules: {missing}")
    return {
        "depth_encoder": tuple(actor.depth_encoder.parameters()),
        "state_latent_encoders": tuple((*actor.embed_s.parameters(), *actor.embed_z.parameters())),
        "fusion": tuple(actor.fusion.parameters()),
        "policy": tuple(actor.policy.parameters()),
        "all": tuple(actor.parameters()),
    }


def _gradient_vector(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach() for parameter, gradient in zip(parameters, gradients)
    )


def _gradient_pair_metrics(
    first: Sequence[torch.Tensor],
    second: Sequence[torch.Tensor],
    indices: Sequence[int],
) -> dict[str, float]:
    first_sq = torch.zeros((), device=first[0].device, dtype=torch.float64)
    second_sq = torch.zeros_like(first_sq)
    dot = torch.zeros_like(first_sq)
    for index in indices:
        first_value = first[index].double()
        second_value = second[index].double()
        first_sq += first_value.square().sum()
        second_sq += second_value.square().sum()
        dot += (first_value * second_value).sum()
    first_norm = first_sq.sqrt()
    second_norm = second_sq.sqrt()
    denominator = (first_norm * second_norm).clamp_min(1.0e-30)
    return {
        "first_norm": float(first_norm.item()),
        "second_norm": float(second_norm.item()),
        "dot": float(dot.item()),
        "cosine": float((dot / denominator).item()),
        "first_projection_on_second": float((dot / second_sq.clamp_min(1.0e-30)).item()),
    }


def _add_gradients(*branches: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(sum(values) for values in zip(*branches))


def _scale_gradients(branch: Sequence[torch.Tensor], scale: float) -> tuple[torch.Tensor, ...]:
    return tuple(value * float(scale) for value in branch)


def _branch_gradient_report(
    *,
    actor: torch.nn.Module,
    branch_gradients: Mapping[str, Sequence[torch.Tensor]],
) -> dict[str, Any]:
    parameters = tuple(actor.parameters())
    parameter_to_index = {id(parameter): index for index, parameter in enumerate(parameters)}
    groups = _actor_parameter_groups(actor)
    pairs = {
        "D_vs_FB": ("D", "FB"),
        "Aux_vs_FB": ("Aux", "FB"),
        "D_vs_main": ("D", "main"),
        "D_vs_total": ("D", "total"),
    }
    output: dict[str, Any] = {}
    for group_name, group_parameters in groups.items():
        indices = [parameter_to_index[id(parameter)] for parameter in group_parameters]
        group_output: dict[str, Any] = {}
        for branch_name, gradient in branch_gradients.items():
            sq_norm = sum(gradient[index].double().square().sum() for index in indices)
            group_output[f"norm/{branch_name}"] = float(torch.sqrt(sq_norm).item())
        for label, (first, second) in pairs.items():
            metrics = _gradient_pair_metrics(
                branch_gradients[first],
                branch_gradients[second],
                indices,
            )
            group_output[f"cosine/{label}"] = metrics["cosine"]
            group_output[f"projection/{first}_on_{second}"] = metrics["first_projection_on_second"]
        fb_norm = group_output["norm/FB"]
        group_output["norm_ratio/D_to_FB"] = group_output["norm/D"] / max(fb_norm, 1.0e-30)
        group_output["norm_ratio/Aux_to_FB"] = group_output["norm/Aux"] / max(fb_norm, 1.0e-30)
        output[group_name] = group_output
    return output


def _summarize_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("Cannot summarize an empty report list")

    def recurse(values: Sequence[Any]) -> Any:
        first = values[0]
        if isinstance(first, Mapping):
            return {key: recurse([value[key] for value in values]) for key in first}
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            numeric = [float(value) for value in values]
            return {
                "mean": statistics.fmean(numeric),
                "std": statistics.pstdev(numeric),
                "min": min(numeric),
                "max": max(numeric),
            }
        return first

    return recurse(reports)


def run_gradient_diagnostic(
    *,
    model,
    agent_config: Mapping[str, Any],
    main_replay: MemoryMappedTrajectoryReplay,
    prior_replay: MemoryMappedTrajectoryReplay | None,
    expert_buffer: TrajectoryDictBuffer,
    batch_size: int,
    batches: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    train = agent_config["train"]
    actor = model._actor
    actor_was_training = actor.training
    # cuDNN intentionally disables GRU backward in inference mode even when
    # its parameters require gradients.  The training Actor has no dropout,
    # so enabling training mode exactly restores the optimizer-time graph.
    actor.train(True)
    actor.requires_grad_(True)
    parameters = tuple(actor.parameters())
    reports = []
    for batch_index in range(batches):
        torch.manual_seed(seed + batch_index)
        np.random.seed(seed + batch_index)
        main_batch = main_replay.sample(batch_size)
        dedicated_prior = prior_replay is not None
        prior_batch = prior_replay.sample(batch_size) if dedicated_prior else main_batch
        expert_batch = expert_buffer.sample(batch_size)

        main_obs = _to_device(main_batch["observation"], device)
        main_next_obs = _to_device(main_batch["next"]["observation"], device)
        prior_obs = _to_device(prior_batch["observation"], device)
        prior_next_obs = _to_device(prior_batch["next"]["observation"], device)
        expert_next_obs = _to_device(expert_batch["next"]["observation"], device)
        with torch.no_grad():
            main_obs = model._normalize(main_obs)
            main_next_obs = model._normalize(main_next_obs)
            prior_obs = model._normalize(prior_obs)
            prior_next_obs = model._normalize(prior_next_obs)
            expert_next_obs = model._normalize(expert_next_obs)
            expert_z = _encode_expert(model, expert_next_obs)
            main_sampled_z = _sample_mixed_z(
                model,
                main_next_obs,
                expert_z,
                p_goal=float(train["train_goal_ratio"]),
                p_expert=float(train["expert_asm_ratio"]),
            )
            prior_sampled_z = (
                _sample_mixed_z(
                    model,
                    prior_next_obs,
                    expert_z,
                    p_goal=float(train["train_goal_ratio"]),
                    p_expert=float(train["expert_asm_ratio"]),
                )
                if dedicated_prior
                else main_sampled_z
            )
            relabel_ratio = train.get("relabel_ratio")
            if relabel_ratio is None:
                main_z = main_batch["z"].to(device)
                prior_z = prior_batch["z"].to(device) if dedicated_prior else main_z
            else:
                main_mask = torch.rand((batch_size, 1), device=device) <= float(relabel_ratio)
                prior_mask = (
                    torch.rand((batch_size, 1), device=device) <= float(relabel_ratio)
                    if dedicated_prior
                    else main_mask
                )
                main_z = torch.where(main_mask, main_sampled_z, main_batch["z"].to(device))
                prior_z = (
                    torch.where(prior_mask, prior_sampled_z, prior_batch["z"].to(device))
                    if dedicated_prior
                    else main_z
                )

        if dedicated_prior:
            combined_obs = tree_map(
                lambda main_value, prior_value: torch.cat((main_value, prior_value), dim=0),
                main_obs,
                prior_obs,
            )
            combined_z = torch.cat((main_z, prior_z), dim=0)
        else:
            combined_obs = main_obs
            combined_z = main_z
        distribution = actor(combined_obs, combined_z, model.cfg.actor_std)
        combined_action = distribution.sample(clip=float(train["stddev_clip"]))
        main_action = combined_action[:batch_size]
        prior_action = combined_action[batch_size:] if dedicated_prior else main_action

        q_discriminator = _pessimistic_value(
            model._critic(prior_obs, prior_z, prior_action),
            float(train["actor_pessimism_penalty"]),
        )
        q_aux = _pessimistic_value(
            model._aux_critic(main_obs, main_z, main_action),
            float(train["actor_pessimism_penalty"]),
        )
        forward = model._forward_map(main_obs, main_z, main_action)
        q_fb = _pessimistic_value(
            (forward * main_z).sum(dim=-1),
            float(train["actor_pessimism_penalty"]),
        )
        weight_tensor = q_fb.abs().mean().detach() if bool(train["scale_reg"]) else q_fb.new_ones(())
        d_scale = float(train["reg_coeff"]) * float(weight_tensor.item())
        aux_scale = float(train["reg_coeff_aux"]) * float(weight_tensor.item())
        losses = {
            "FB": -q_fb.mean(),
            "Aux": -q_aux.mean() * aux_scale,
            "D": -q_discriminator.mean() * d_scale,
        }
        gradients = {
            "FB": _gradient_vector(losses["FB"], parameters, retain_graph=True),
            "Aux": _gradient_vector(losses["Aux"], parameters, retain_graph=True),
            "D": _gradient_vector(losses["D"], parameters, retain_graph=False),
        }
        gradients["main"] = _add_gradients(gradients["FB"], gradients["Aux"])
        gradients["total"] = _add_gradients(gradients["main"], gradients["D"])
        raw_d = _scale_gradients(gradients["D"], 1.0 / max(d_scale, 1.0e-30))
        raw_aux = _scale_gradients(gradients["Aux"], 1.0 / max(aux_scale, 1.0e-30))
        report = {
            "batch_index": batch_index,
            "q": {
                "FB": float(q_fb.mean().item()),
                "Aux": float(q_aux.mean().item()),
                "D": float(q_discriminator.mean().item()),
                "weight": float(weight_tensor.item()),
                "D_effective_scale": d_scale,
                "Aux_effective_scale": aux_scale,
            },
            "effective_gradients": _branch_gradient_report(actor=actor, branch_gradients=gradients),
            "raw_gradients": _branch_gradient_report(
                actor=actor,
                branch_gradients={
                    "FB": gradients["FB"],
                    "Aux": raw_aux,
                    "D": raw_d,
                    "main": _add_gradients(gradients["FB"], raw_aux),
                    "total": _add_gradients(gradients["FB"], raw_aux, raw_d),
                },
            ),
        }
        reports.append(report)
        del combined_action, distribution, gradients, losses
        torch.cuda.empty_cache()
    actor.requires_grad_(False)
    actor.train(actor_was_training)
    return {
        "batch_size": batch_size,
        "num_batches": batches,
        "seed": seed,
        "per_batch": reports,
        "summary": _summarize_reports(reports),
    }


def _stairs_column(root_y: np.ndarray) -> np.ndarray:
    # Frozen rp1_simple layout: seven 5 m columns centered around world y=0.
    return np.floor((root_y + 17.5) / 5.0).astype(np.int64)


def _upright_score_wxyz(quaternion: np.ndarray) -> np.ndarray:
    _w, x, y, _z = np.moveaxis(quaternion, -1, 0)
    return 1.0 - 2.0 * (x * x + y * y)


def _yaw_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def select_stairs_replay_indices(
    replay: MemoryMappedTrajectoryReplay,
    *,
    count: int,
    seed: int,
    search_size: int = 250_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if "qpos" not in replay.arrays or "qvel" not in replay.arrays:
        raise KeyError("Main replay must store qpos and qvel for state-matched depth rendering")
    rng = np.random.default_rng(seed)
    time_idx = rng.integers(0, replay.storage_length, size=search_size, dtype=np.int64)
    env_count = int(replay.arrays["qpos"].shape[1])
    env_idx = rng.integers(0, env_count, size=search_size, dtype=np.int64)
    qpos = np.asarray(replay.arrays["qpos"][time_idx, env_idx, :7])
    qvel_xy = np.asarray(replay.arrays["qvel"][time_idx, env_idx, :2])
    columns = _stairs_column(qpos[:, 1])
    yaw = _yaw_wxyz(qpos[:, 3:7])
    forward_speed = np.cos(yaw) * qvel_xy[:, 0] + np.sin(yaw) * qvel_xy[:, 1]
    upright = _upright_score_wxyz(qpos[:, 3:7])
    available = replay.available_depth_history(time_idx, env_idx)
    valid = np.isin(columns, STAIRS_COLUMNS) & (upright >= 0.65) & (available >= max(DEPTH_HISTORY_OFFSETS))
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < count:
        raise RuntimeError(f"Only found {valid_indices.size} full-history upright stairs states, need {count}")
    # Cover both stair directions while preferring locomoting states.  A small
    # random jitter prevents repeatedly selecting a single fast trajectory.
    score = np.abs(forward_speed[valid_indices] - 0.7) + rng.uniform(0.0, 0.05, valid_indices.size)
    ordered = valid_indices[np.argsort(score)]
    selected: list[int] = []
    per_column = max(1, count // 2)
    for column in STAIRS_COLUMNS:
        candidates = ordered[columns[ordered] == column]
        selected.extend(candidates[:per_column].tolist())
    if len(selected) < count:
        already = set(selected)
        selected.extend(index for index in ordered if index not in already)
    selected_array = np.asarray(selected[:count], dtype=np.int64)
    return (
        time_idx[selected_array],
        env_idx[selected_array],
        {
            "terrain_column": columns[selected_array],
            "forward_speed": forward_speed[selected_array],
            "upright_score": upright[selected_array],
        },
    )


def _write_qpos_and_sense(core, qpos: torch.Tensor) -> None:
    qpos = qpos.to(core.device, dtype=torch.float32)
    if qpos.ndim != 2 or qpos.shape[0] != core.num_envs or qpos.shape[1] != 7 + core.num_dof:
        raise ValueError(f"Expected qpos [{core.num_envs}, {7 + core.num_dof}], got {tuple(qpos.shape)}")
    root_state = torch.cat(
        (qpos[:, :7], torch.zeros((core.num_envs, 6), device=core.device)),
        dim=-1,
    )
    env_ids = torch.arange(core.num_envs, device=core.device, dtype=torch.long)
    core.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    core.robot.write_joint_state_to_sim(
        qpos[:, 7:],
        torch.zeros_like(qpos[:, 7:]),
        joint_ids=core._joint_ids,
        env_ids=env_ids,
    )
    core.mjlab_env.scene.write_data_to_sim()
    core.mjlab_env.sim.forward()
    for name in ("terrain_height", "g1_direct_depth"):
        sensor = core.mjlab_env.scene.sensors.get(name)
        if sensor is not None:
            sensor.update(0.0)
    core.mjlab_env.sim.sense()


@dataclass
class PairedDepthRenderer:
    stairs_env: Any
    plane_env: Any

    @classmethod
    def build(cls, *, device: str, output_dir: Path, seed: int) -> "PairedDepthRenderer":
        config = build_ufo_mjlab_config(
            device=device,
            work_dir=str(output_dir / "renderer_env"),
            num_envs=1,
            num_env_steps=2048,
            seed=seed,
            use_wandb=False,
            wandb_run_name=None,
            disable_eval_prioritization=True,
            smoke=True,
            agent="fb_depth",
            terrain_mode="rp1_simple",
            prior_plane_envs=0,
            disable_dr=True,
            disable_obs_noise=True,
        )
        stairs_cfg = config.env.model_copy(update={"fixed_direct_depth_delay_frames": 0})
        plane_cfg = make_canonical_plane_training_config(stairs_cfg).model_copy(
            update={
                "disable_domain_randomization": True,
                "disable_obs_noise": True,
                "fixed_direct_depth_delay_frames": 0,
            }
        )
        stairs_env, _ = stairs_cfg.build(num_envs=1)
        try:
            plane_env, _ = plane_cfg.build(num_envs=1)
        except Exception:
            stairs_env.close()
            raise
        stairs_env.reset(to_numpy=False)
        plane_env.reset(to_numpy=False)
        return cls(stairs_env=stairs_env, plane_env=plane_env)

    def close(self) -> None:
        self.stairs_env.close()
        self.plane_env.close()

    @torch.no_grad()
    def render_pair(self, qpos_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        if qpos_wxyz.shape != (1, 36):
            raise ValueError(f"Expected one G1 qpos [1, 36], got {tuple(qpos_wxyz.shape)}")
        stairs_core = self.stairs_env._env
        plane_core = self.plane_env._env
        _write_qpos_and_sense(stairs_core, qpos_wxyz)
        clearances = stairs_core._terrain_sensor_clearances()
        if stairs_core._terrain_reference_index is None:
            raise RuntimeError("Terrain reference ray was not initialized")
        clearance = clearances[:, stairs_core._terrain_reference_index]
        stairs_frame = stairs_core._direct_depth_runtime.current_frame(stairs_core.mjlab_env.scene.sensors["g1_direct_depth"])

        plane_qpos = qpos_wxyz.clone()
        plane_qpos[:, 2] = clearance + plane_core.env_origins[:, 2]
        _write_qpos_and_sense(plane_core, plane_qpos)
        plane_frame = plane_core._direct_depth_runtime.current_frame(plane_core.mjlab_env.scene.sensors["g1_direct_depth"])
        return stairs_frame.cpu(), plane_frame.cpu(), float(clearance.item())


def counterfactual_action_metrics(
    stairs_action: torch.Tensor,
    flat_action: torch.Tensor,
    *,
    actor_std: float,
) -> dict[str, Any]:
    difference = stairs_action.float() - flat_action.float()
    per_sample_l2 = torch.linalg.vector_norm(difference, dim=-1)
    per_sample_rms = difference.square().mean(dim=-1).sqrt()
    per_sample_mae = difference.abs().mean(dim=-1)
    joint_groups = {
        "legs": slice(0, 12),
        "waist": slice(12, 15),
        "arms": slice(15, 29),
    }
    return {
        "samples": int(difference.shape[0]),
        "action_dim": int(difference.shape[1]),
        "l2_mean": float(per_sample_l2.mean().item()),
        "l2_median": float(per_sample_l2.median().item()),
        "l2_max": float(per_sample_l2.max().item()),
        "rms_mean": float(per_sample_rms.mean().item()),
        "mae_mean": float(per_sample_mae.mean().item()),
        "rms_in_actor_std": float((per_sample_rms.mean() / float(actor_std)).item()),
        "fraction_rms_above_actor_std": float((per_sample_rms > float(actor_std)).float().mean().item()),
        "joint_group_rms": {name: float(difference[:, group].square().mean().sqrt().item()) for name, group in joint_groups.items()},
        "per_sample_l2": per_sample_l2.cpu().tolist(),
    }


@torch.no_grad()
def run_depth_counterfactual(
    *,
    model,
    main_replay: MemoryMappedTrajectoryReplay,
    samples: int,
    selection_seed: int,
    terrain_seed: int,
    device: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    render_count = max(samples * 2, samples + 8)
    time_idx, env_idx, selection = select_stairs_replay_indices(
        main_replay,
        count=render_count,
        seed=selection_seed,
    )
    available = main_replay.available_depth_history(time_idx, env_idx)
    if np.any(available < max(DEPTH_HISTORY_OFFSETS)):
        raise AssertionError("Counterfactual states do not have the full RP1 history")
    offsets = np.asarray(DEPTH_HISTORY_OFFSETS, dtype=np.int64)
    history_times = np.remainder(time_idx[:, None] - offsets[None, :], main_replay.capacity)
    history_envs = np.broadcast_to(env_idx[:, None], history_times.shape)
    history_qpos = np.asarray(main_replay.arrays["qpos"][history_times, history_envs]).copy()

    renderer = PairedDepthRenderer.build(
        device=device,
        output_dir=output_dir,
        seed=terrain_seed,
    )
    stairs_frames = torch.empty((render_count, len(offsets), 36, 32), dtype=torch.uint8)
    flat_frames = torch.empty_like(stairs_frames)
    clearances = torch.empty((render_count, len(offsets)), dtype=torch.float32)
    try:
        for sample_index in range(render_count):
            for frame_index in range(len(offsets)):
                stairs, flat, clearance = renderer.render_pair(
                    torch.from_numpy(history_qpos[sample_index, frame_index : frame_index + 1]).to(device)
                )
                stairs_frames[sample_index, frame_index] = stairs[0]
                flat_frames[sample_index, frame_index] = flat[0]
                clearances[sample_index, frame_index] = clearance
    finally:
        renderer.close()

    depth_mae = (stairs_frames.float() - flat_frames.float()).abs().mean(dim=(1, 2, 3)) / 255.0
    keep = torch.argsort(depth_mae, descending=True)[:samples].cpu().numpy()
    selected_time = time_idx[keep]
    selected_env = env_idx[keep]
    obs = main_replay.observation(selected_time, selected_env)
    z = main_replay._take("z", selected_time, selected_env).to(device)
    obs_stairs = _to_device(obs, device)
    obs_flat = {key: value.clone() for key, value in obs_stairs.items()}
    obs_stairs["depth_image"] = stairs_frames[keep].to(device)
    obs_flat["depth_image"] = flat_frames[keep].to(device)
    stairs_action = model.act(obs_stairs, z, mean=True)
    flat_action = model.act(obs_flat, z, mean=True)
    metrics = counterfactual_action_metrics(
        stairs_action,
        flat_action,
        actor_std=float(model.cfg.actor_std),
    )
    metrics.update(
        {
            "depth_mae_normalized_mean": float(depth_mae[keep].mean().item()),
            "depth_mae_normalized_min": float(depth_mae[keep].min().item()),
            "depth_mae_normalized_max": float(depth_mae[keep].max().item()),
            "stairs_up_samples": int(np.count_nonzero(selection["terrain_column"][keep] == 2)),
            "stairs_down_samples": int(np.count_nonzero(selection["terrain_column"][keep] == 3)),
            "forward_speed_mean": float(np.mean(selection["forward_speed"][keep])),
            "time_indices": selected_time.tolist(),
            "env_indices": selected_env.tolist(),
        }
    )
    fixture = {
        "stairs_depth": stairs_frames[keep],
        "flat_depth": flat_frames[keep],
        "stairs_action": stairs_action.cpu(),
        "flat_action": flat_action.cpu(),
        "z": z.cpu(),
        "qpos": torch.from_numpy(history_qpos[keep]),
        "clearance": clearances[keep],
        "time_idx": torch.from_numpy(selected_time.copy()),
        "env_idx": torch.from_numpy(selected_env.copy()),
    }
    return metrics, fixture


def _root_state_from_qpos_qvel(qpos: torch.Tensor, qvel: torch.Tensor) -> dict[str, torch.Tensor]:
    qpos = qpos.float()
    qvel = qvel.float()
    quaternion_wxyz = qpos[:, 3:7]
    quaternion_xyzw = quaternion_wxyz[:, (1, 2, 3, 0)]
    angular_body = qvel[:, 3:6]
    angular_world = quat_rotate(quaternion_xyzw, angular_body, w_last=True)
    root = torch.cat((qpos[:, :3], quaternion_xyzw, qvel[:, :3], angular_world), dim=-1)
    dof = torch.stack((qpos[:, 7:], qvel[:, 6:]), dim=-1)
    return {"root_states": root, "dof_states": dof}


class FlatDepthHistoryOverride:
    """Render the physical robot pose over a plane and mirror RP1 timing."""

    def __init__(self, plane_env) -> None:
        self.env = plane_env
        self.core = plane_env._env

    @torch.no_grad()
    def _sync_pose(self, physical_env) -> None:
        qpos, _qvel = physical_env._get_qpos_qvel(to_numpy=False)
        physical_core = physical_env._env
        clearances = physical_core._terrain_sensor_clearances()
        if physical_core._terrain_reference_index is None:
            raise RuntimeError("Physical terrain reference ray is unavailable")
        clearance = clearances[:, physical_core._terrain_reference_index]
        plane_qpos = qpos.clone()
        plane_qpos[:, 2] = clearance + self.core.env_origins[:, 2]
        _write_qpos_and_sense(self.core, plane_qpos)

    @torch.no_grad()
    def reset(self, physical_env) -> torch.Tensor:
        self._sync_pose(physical_env)
        ids = torch.arange(self.core.num_envs, device=self.core.device)
        sensor = self.core.mjlab_env.scene.sensors["g1_direct_depth"]
        self.core._direct_depth_runtime.reset_from_sensor(sensor, ids)
        return self.core._direct_depth_runtime.observation().clone()

    @torch.no_grad()
    def append(self, physical_env) -> torch.Tensor:
        self._sync_pose(physical_env)
        sensor = self.core.mjlab_env.scene.sensors["g1_direct_depth"]
        self.core._direct_depth_runtime.append_from_sensor(sensor)
        return self.core._direct_depth_runtime.observation().clone()


@torch.no_grad()
def _closed_loop_rollout(
    *,
    model,
    env,
    target_states: Mapping[str, torch.Tensor],
    z: torch.Tensor,
    episode_steps: int,
    flat_override: FlatDepthHistoryOverride | None,
) -> dict[str, Any]:
    observation, _ = env.reset(to_numpy=False, target_states=target_states)
    if flat_override is not None:
        observation["depth_image"] = flat_override.reset(env)
    core = env._env
    initial_pos = core.robot_root_states[:, :3].clone()
    initial_quaternion = core.robot_root_states[:, 3:7].clone()
    x, y, zq, w = initial_quaternion.unbind(dim=-1)
    initial_yaw = torch.atan2(2.0 * (w * zq + x * y), 1.0 - 2.0 * (y.square() + zq.square()))
    heading = torch.stack((torch.cos(initial_yaw), torch.sin(initial_yaw)), dim=-1)
    initial_clearance = core._terrain_sensor_clearances()[:, core._terrain_reference_index].clone()
    initial_ground = initial_pos[:, 2] - initial_clearance
    max_ground_gain = 0.0
    action_norms: list[float] = []
    completed = 0
    terminated_flag = False
    truncated_flag = False
    for step in range(episode_steps):
        action = model.act(observation, z, mean=True)
        action_norms.append(float(torch.linalg.vector_norm(action, dim=-1).mean().item()))
        observation, _reward, terminated, truncated, _info = env.step(action, to_numpy=False)
        completed = step + 1
        terminated_flag = bool(torch.as_tensor(terminated).reshape(-1)[0].item())
        truncated_flag = bool(torch.as_tensor(truncated).reshape(-1)[0].item())
        clearance = core._terrain_sensor_clearances()[:, core._terrain_reference_index]
        ground = core.robot_root_states[:, 2] - clearance
        max_ground_gain = max(max_ground_gain, float((ground - initial_ground).max().item()))
        if terminated_flag or truncated_flag:
            break
        if flat_override is not None:
            observation["depth_image"] = flat_override.append(env)
    displacement = core.robot_root_states[:, :2] - initial_pos[:, :2]
    forward = (displacement * heading).sum(dim=-1)
    planar = torch.linalg.vector_norm(displacement, dim=-1)
    current_quaternion = core.robot_root_states[:, 3:7]
    x, y, zq, w = current_quaternion.unbind(dim=-1)
    current_yaw = torch.atan2(2.0 * (w * zq + x * y), 1.0 - 2.0 * (y.square() + zq.square()))
    yaw_change = torch.atan2(torch.sin(current_yaw - initial_yaw), torch.cos(current_yaw - initial_yaw))
    return {
        "steps": completed,
        "survived": not terminated_flag,
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "forward_displacement": float(forward.item()),
        "planar_displacement": float(planar.item()),
        "yaw_change_rad": float(yaw_change.item()),
        "max_ground_height_gain": max_ground_gain,
        "mean_action_l2": statistics.fmean(action_norms) if action_norms else 0.0,
    }


@torch.no_grad()
def run_closed_loop_intervention(
    *,
    model,
    fixture: Mapping[str, torch.Tensor],
    device: str,
    output_dir: Path,
    terrain_seed: int,
    episode_steps: int,
) -> dict[str, Any]:
    # Use the replay state with the largest instantaneous action response.
    action_delta = torch.linalg.vector_norm(
        fixture["stairs_action"] - fixture["flat_action"],
        dim=-1,
    )
    selected = int(torch.argmax(action_delta).item())
    qpos = fixture["qpos"][selected, -1:].to(device)
    # Replay qvel is not included in the compact fixture.  A zero-velocity
    # reset makes both interventions exactly paired and avoids reconstructing
    # stale simulator actuator state.
    qvel = torch.zeros((1, 35), device=device)
    target_states = _root_state_from_qpos_qvel(qpos, qvel)
    z = fixture["z"][selected : selected + 1].to(device)

    config = build_ufo_mjlab_config(
        device=device,
        work_dir=str(output_dir / "closed_loop_env"),
        num_envs=1,
        num_env_steps=max(2048, episode_steps + 128),
        seed=terrain_seed,
        use_wandb=False,
        wandb_run_name=None,
        disable_eval_prioritization=True,
        smoke=True,
        agent="fb_depth",
        terrain_mode="rp1_simple",
        prior_plane_envs=0,
        disable_dr=True,
        disable_obs_noise=True,
    )
    physical_cfg = config.env.model_copy(
        update={
            "auto_reset": False,
            "max_episode_length_s": max(30.0, episode_steps / 50.0 + 1.0),
            "fixed_direct_depth_delay_frames": 0,
        }
    )
    plane_cfg = make_canonical_plane_training_config(physical_cfg)
    physical_env, _ = physical_cfg.build(num_envs=1)
    try:
        plane_env, _ = plane_cfg.build(num_envs=1)
    except Exception:
        physical_env.close()
        raise
    try:
        plane_env.reset(to_numpy=False)
        baseline = _closed_loop_rollout(
            model=model,
            env=physical_env,
            target_states=target_states,
            z=z,
            episode_steps=episode_steps,
            flat_override=None,
        )
        override = FlatDepthHistoryOverride(plane_env)
        flat_depth = _closed_loop_rollout(
            model=model,
            env=physical_env,
            target_states=target_states,
            z=z,
            episode_steps=episode_steps,
            flat_override=override,
        )
    finally:
        physical_env.close()
        plane_env.close()
    return {
        "selected_counterfactual_sample": selected,
        "initial_open_loop_action_l2": float(action_delta[selected].item()),
        "episode_steps_requested": episode_steps,
        "actual_stairs_depth": baseline,
        "flat_depth_on_stairs_physics": flat_depth,
        "difference_flat_minus_actual": {
            key: float(flat_depth[key]) - float(baseline[key])
            for key in (
                "forward_displacement",
                "planar_displacement",
                "yaw_change_rad",
                "max_ground_height_gain",
            )
        },
    }


def _checkpoint_step(checkpoint_dir: Path) -> int | None:
    status = checkpoint_dir / "train_status.json"
    if not status.exists():
        return None
    return int(_load_json(status).get("global_time", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--buffer-dir", type=Path, default=None)
    parser.add_argument("--buffer-rank", type=int, default=0)
    parser.add_argument("--expert-cache", type=Path, default=None)
    parser.add_argument(
        "--expert-cache-root",
        type=Path,
        default=Path("/data/xue/UFO/cache/expert_buffers"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gradient-batches", type=int, default=3)
    parser.add_argument("--counterfactual-samples", type=int, default=32)
    parser.add_argument("--closed-loop-steps", type=int, default=750)
    parser.add_argument("--skip-gradient", action="store_true")
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument("--skip-closed-loop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir is not None else args.run_dir / "checkpoint"
    buffer_dir = args.buffer_dir.expanduser().resolve() if args.buffer_dir is not None else checkpoint_dir / "buffers"
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expert_cache = args.expert_cache.expanduser().resolve() if args.expert_cache is not None else _find_expert_cache(args.expert_cache_root)
    load_device = checkpoint_load_device(args.device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=load_device)
    model.to(args.device).eval()
    agent_config = _load_json(checkpoint_dir / "config.json")
    main_replay = MemoryMappedTrajectoryReplay(buffer_dir / f"train_rank_{args.buffer_rank}")
    prior_buffer_dir = buffer_dir / f"prior_rank_{args.buffer_rank}"
    prior_replay = (
        MemoryMappedTrajectoryReplay(prior_buffer_dir)
        if prior_buffer_dir.exists()
        else None
    )
    report: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_global_step": _checkpoint_step(checkpoint_dir),
        "buffer_rank": args.buffer_rank,
        "expert_cache": str(expert_cache),
        "behavior_prior_source": "plane" if prior_replay is not None else "main",
        "seed": args.seed,
    }
    if not args.skip_gradient:
        expert_buffer = _load_expert_buffer(expert_cache)
        report["actor_branch_gradients"] = run_gradient_diagnostic(
            model=model,
            agent_config=agent_config,
            main_replay=main_replay,
            prior_replay=prior_replay,
            expert_buffer=expert_buffer,
            batch_size=args.batch_size,
            batches=args.gradient_batches,
            seed=args.seed,
            device=args.device,
        )
        del expert_buffer
    fixture = None
    if not args.skip_counterfactual:
        counterfactual, fixture = run_depth_counterfactual(
            model=model,
            main_replay=main_replay,
            samples=args.counterfactual_samples,
            selection_seed=args.seed + 10_000,
            terrain_seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
        )
        report["depth_counterfactual"] = counterfactual
        torch.save(fixture, args.output_dir / "depth_counterfactual_fixture.pt")
    if not args.skip_closed_loop:
        if fixture is None:
            fixture_path = args.output_dir / "depth_counterfactual_fixture.pt"
            if not fixture_path.exists():
                raise FileNotFoundError(
                    "Closed-loop intervention needs the depth counterfactual fixture; run without --skip-counterfactual first"
                )
            fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
        report["closed_loop_depth_intervention"] = run_closed_loop_intervention(
            model=model,
            fixture=fixture,
            device=args.device,
            output_dir=args.output_dir,
            terrain_seed=args.seed,
            episode_steps=args.closed_loop_steps,
        )
    output = args.output_dir / "diagnostics.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[INFO] wrote {output}")


if __name__ == "__main__":
    main()
