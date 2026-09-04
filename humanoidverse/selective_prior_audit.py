"""Frozen-replay feasibility audit for the selective online behavior prior.

This command never mutates a training run.  It reconstructs exact-tracking
policy windows directly from a checkpointed ring replay, evaluates the
matched temporal semantic gate with the checkpoint's frozen B/normalizer
snapshot, and can fit a fresh discriminator on the resulting E/V/B split.

The audit deliberately stores and reports replay provenance rather than the
collection-time latent.  Expert and policy latents are re-derived with one
frozen coordinate contract, and train/holdout separation is by motion id so
no context from a held-out behavior leaks into discriminator fitting.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd

from humanoidverse.agents.behavior_context import HEADING_SOURCE_EXACT_TRACKING
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.nn_models import eval_mode, weight_init
from humanoidverse.agents.selective_prior import (
    BehaviorFamily,
    approximate_pairwise_auc,
    behavior_family_from_name,
    future_window_means,
    masked_temporal_mean,
)
from humanoidverse.direct_depth_actor_diagnostics import (
    MemoryMappedTrajectoryReplay,
    _load_expert_buffer,
)
from humanoidverse.mjlab_inference_utils import checkpoint_load_device

OBSERVATION_KEYS = ("state", "privileged_state")
PATHOLOGY_THRESHOLDS = {
    "limits_dof_pos": 0.35,
    "penalty_body_impact": 20.0,
    "penalty_slippage": 3.0,
    "penalty_ankle_roll": 1.0,
    "penalty_action_rate": 150.0,
    "feet_stumble": 0.75,
    "feet_at_plane": 0.75,
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().reshape(-1).cpu()
    if value.numel() == 0:
        return {name: float("nan") for name in ("mean", "std", "p01", "p10", "p50", "p90", "p99")}
    points = torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99])
    q = torch.quantile(value, points)
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "p01": float(q[0].item()),
        "p10": float(q[1].item()),
        "p50": float(q[2].item()),
        "p90": float(q[3].item()),
        "p99": float(q[4].item()),
    }


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    count = int(denominator.to(torch.bool).sum().item())
    if count == 0:
        return float("nan")
    return float((numerator.to(torch.bool) & denominator.to(torch.bool)).sum().item() / count)


@dataclass(frozen=True)
class GateThresholds:
    good_cosine_mean: float = 0.75
    good_cosine_min: float = 0.35
    bad_cosine_mean: float = 0.0
    bad_sustain_fraction: float = 0.5
    good_heading_cost_mean_max: float = 0.30
    bad_heading_cost_mean_min: float = 1.0


@dataclass
class FrozenWindowIndices:
    time: np.ndarray
    env: np.ndarray

    def __post_init__(self) -> None:
        if self.time.ndim != 2 or self.env.shape != self.time.shape:
            raise ValueError("Frozen window indices must be identically shaped [window,time]")

    @property
    def count(self) -> int:
        return int(self.time.shape[0])

    @property
    def length(self) -> int:
        return int(self.time.shape[1])


@dataclass
class GateEvaluation:
    cosine: torch.Tensor
    semantic_mean: torch.Tensor
    semantic_min: torch.Tensor
    heading_cost: torch.Tensor
    pathology: torch.Tensor
    good_window: torch.Tensor
    bad_local: torch.Tensor
    getup: torch.Tensor
    getup_success: torch.Tensor
    transition_done: torch.Tensor


@dataclass
class OfflinePriorDataset:
    expert_obs: dict[str, torch.Tensor]
    validated_obs: dict[str, torch.Tensor]
    bad_obs: dict[str, torch.Tensor]
    expert_z: torch.Tensor
    validated_z: torch.Tensor
    bad_z: torch.Tensor
    validated_confidence: torch.Tensor
    expert_motion_id: torch.Tensor
    validated_motion_id: torch.Tensor
    bad_motion_id: torch.Tensor


def validate_exact_tracking_windows(
    *,
    source: np.ndarray,
    context_id: np.ndarray,
    motion_id: np.ndarray,
    reference_index: np.ndarray,
    transition_done: np.ndarray,
    exact_source: int = HEADING_SOURCE_EXACT_TRACKING,
) -> np.ndarray:
    """Return windows whose semantic provenance is continuous and explicit."""

    arrays = (source, context_id, motion_id, reference_index, transition_done)
    if any(value.ndim != 2 or value.shape != source.shape for value in arrays):
        raise ValueError("All exact-tracking validation inputs must share [batch,time]")
    expected_reference = reference_index[:, :1] + np.arange(reference_index.shape[1], dtype=np.int64)[None, :]
    return (
        np.all(source == int(exact_source), axis=1)
        & np.all(context_id == context_id[:, :1], axis=1)
        & np.all(motion_id == motion_id[:, :1], axis=1)
        & np.all(reference_index == expected_reference, axis=1)
        & np.all(motion_id >= 0, axis=1)
        & np.all(reference_index >= 0, axis=1)
        & ~np.any(transition_done[:, :-1], axis=1)
    )


def classify_gate_windows(
    *,
    cosine: torch.Tensor,
    heading_cost: torch.Tensor,
    pathology: torch.Tensor,
    thresholds: GateThresholds,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the online GOOD/BAD rules to already matched gate measurements."""

    if cosine.ndim != 2:
        raise ValueError("cosine must be [batch,gate_window]")
    if pathology.ndim != 2 or heading_cost.ndim != 2:
        raise ValueError("heading_cost and pathology must be [batch,total_time]")
    window = cosine.shape[1]
    if heading_cost.shape[0] != cosine.shape[0] or pathology.shape != heading_cost.shape:
        raise ValueError("Gate tensors have incompatible batch dimensions")
    if heading_cost.shape[1] < window:
        raise ValueError("Gate outcomes do not cover the semantic window")

    finite = torch.isfinite(cosine).all(dim=1)
    semantic_mean = torch.nanmean(cosine, dim=1)
    semantic_min = torch.nan_to_num(cosine, nan=float("inf")).amin(dim=1)
    semantic_min = torch.where(finite, semantic_min, torch.full_like(semantic_min, -1.0))
    good = (
        finite
        & (semantic_mean >= float(thresholds.good_cosine_mean))
        & (semantic_min >= float(thresholds.good_cosine_min))
        & (heading_cost.mean(dim=1) <= float(thresholds.good_heading_cost_mean_max))
        & ~pathology.any(dim=1)
    )
    semantic_bad = cosine <= float(thresholds.bad_cosine_mean)
    heading_bad = heading_cost[:, :window] >= float(thresholds.bad_heading_cost_mean_min)
    sustained_bad = (semantic_bad | heading_bad).float().mean(dim=1) >= float(thresholds.bad_sustain_fraction)
    bad_local = pathology[:, :window] | (sustained_bad[:, None] & (semantic_bad | heading_bad))
    # A full-window GOOD decision and any local BAD evidence conflict. The
    # online state machine resolves such evidence to UNKNOWN, never by enum
    # ordering, so exclude the whole window from positive admission here.
    good &= ~bad_local.any(dim=1)
    return good, bad_local, semantic_mean, semantic_min


class FrozenSelectiveReplay(MemoryMappedTrajectoryReplay):
    """Read-only ring replay with provenance-aware contiguous window sampling."""

    REQUIRED_FIELDS = (
        "heading_source_type",
        "heading_context_id",
        "prior_motion_id",
        "prior_reference_index",
        "transition_terminated",
        "transition_truncated",
        "observation-heading",
        "observation-state",
        "observation-privileged_state",
    )

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        missing = [key for key in self.REQUIRED_FIELDS if key not in self.arrays]
        if missing:
            raise RuntimeError(f"Frozen replay lacks selective-prior audit fields: {missing}")

    def sample_exact_tracking_windows(
        self,
        count: int,
        *,
        length: int,
        seed: int,
        maximum_draws: int | None = None,
    ) -> FrozenWindowIndices:
        if count <= 0 or length <= 1:
            raise ValueError("count must be positive and length must exceed one")
        maximum_draws = int(maximum_draws or max(100_000, count * 256))
        rng = np.random.default_rng(seed)
        logical_start = self.cursor if self.is_full else 0
        logical_length = self.storage_length if self.is_full else self.cursor
        if logical_length < length:
            raise RuntimeError("Frozen replay is shorter than the requested gate window")

        accepted: list[np.ndarray] = []
        accepted_env: list[np.ndarray] = []
        seen: set[tuple[int, int]] = set()
        draws = 0
        chunk = max(1024, min(65_536, count * 8))
        offsets = np.arange(length, dtype=np.int64)[None, :]
        while sum(value.shape[0] for value in accepted) < count and draws < maximum_draws:
            n = min(chunk, maximum_draws - draws)
            start_position = rng.integers(0, logical_length - length + 1, size=n, dtype=np.int64)
            start_time = np.remainder(logical_start + start_position, self.capacity)
            env = rng.integers(0, self.arrays["heading_context_id"].shape[1], size=n, dtype=np.int64)
            time = np.remainder(start_time[:, None] + offsets, self.capacity)
            env_grid = np.broadcast_to(env[:, None], time.shape)
            source = np.asarray(self.arrays["heading_source_type"][time, env_grid]).reshape(n, length, -1)[..., 0]
            context = np.asarray(self.arrays["heading_context_id"][time, env_grid]).reshape(n, length, -1)[..., 0]
            motion = np.asarray(self.arrays["prior_motion_id"][time, env_grid]).reshape(n, length, -1)[..., 0]
            reference = np.asarray(self.arrays["prior_reference_index"][time, env_grid]).reshape(n, length, -1)[..., 0]
            terminated = np.asarray(self.arrays["transition_terminated"][time, env_grid]).reshape(n, length, -1).any(axis=-1)
            truncated = np.asarray(self.arrays["transition_truncated"][time, env_grid]).reshape(n, length, -1).any(axis=-1)
            valid = validate_exact_tracking_windows(
                source=source,
                context_id=context,
                motion_id=motion,
                reference_index=reference,
                transition_done=terminated | truncated,
            )
            for row in np.nonzero(valid)[0]:
                key = (int(start_time[row]), int(env[row]))
                if key in seen:
                    continue
                seen.add(key)
                accepted.append(time[row : row + 1].copy())
                accepted_env.append(env_grid[row : row + 1].copy())
                if len(accepted) >= count:
                    break
            draws += n
        if not accepted:
            raise RuntimeError("No continuous exact-tracking windows were found in the frozen replay")
        if len(accepted) < count:
            raise RuntimeError(
                f"Only found {len(accepted)} unique exact-tracking windows after {draws} draws; requested {count}"
            )
        return FrozenWindowIndices(
            time=np.concatenate(accepted[:count], axis=0),
            env=np.concatenate(accepted_env[:count], axis=0),
        )

    def sequence_field(self, name: str, indices: FrozenWindowIndices) -> torch.Tensor:
        if name not in self.arrays:
            raise KeyError(f"Frozen replay field {name!r} is unavailable")
        return torch.from_numpy(np.asarray(self.arrays[name][indices.time, indices.env]).copy())

    def sequence_observation(
        self,
        indices: FrozenWindowIndices,
        *,
        successor: bool,
    ) -> dict[str, torch.Tensor]:
        time = np.remainder(indices.time + int(successor), self.capacity)
        return {
            key: torch.from_numpy(np.asarray(self.arrays[f"observation-{key}"][time, indices.env]).copy())
            for key in OBSERVATION_KEYS
        }

    def sequence_pathology(self, indices: FrozenWindowIndices) -> torch.Tensor:
        severe = torch.zeros(indices.time.shape, dtype=torch.bool)
        for name, threshold in PATHOLOGY_THRESHOLDS.items():
            key = f"aux_rewards-{name}"
            if key in self.arrays:
                value = self.sequence_field(key, indices).reshape(*indices.time.shape, -1)[..., 0].float()
                severe |= value >= float(threshold)
        if "action" in self.arrays:
            action = self.sequence_field("action", indices)
            severe |= ~torch.isfinite(action).all(dim=-1)
        return severe


def _snapshot_modules(model: torch.nn.Module, training_state: Mapping[str, Any], device: str):
    if "gate_backward_teacher" not in training_state or "gate_obs_normalizer" not in training_state:
        raise RuntimeError("Checkpoint has no frozen gate B/normalizer snapshot")
    encoder = copy.deepcopy(model._target_backward_map)
    normalizer = copy.deepcopy(model._obs_normalizer)
    encoder.load_state_dict(training_state["gate_backward_teacher"])
    normalizer.load_state_dict(training_state["gate_obs_normalizer"])
    encoder.to(device).eval().requires_grad_(False)
    normalizer.to(device).eval().requires_grad_(False)
    return encoder, normalizer


def _flatten_observation(obs: Mapping[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.flatten(0, 1).to(device, non_blocking=True) for key, value in obs.items()}


@torch.no_grad()
def encode_policy_gate_windows(
    *,
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    normalizer: torch.nn.Module,
    next_observation: Mapping[str, torch.Tensor],
    output_window: int,
    sequence_length: int,
    device: str,
) -> torch.Tensor:
    batch, total = next(iter(next_observation.values())).shape[:2]
    if total < output_window + sequence_length - 1:
        raise ValueError("Policy window does not contain enough future observations")
    flat = _flatten_observation(next_observation, device)
    with eval_mode(normalizer):
        normalized = normalizer(flat)
    embeddings = encoder(normalized).reshape(batch, total, -1)
    horizons = torch.full((batch, output_window), sequence_length, device=device, dtype=torch.long)
    return model.project_z(future_window_means(embeddings, horizons)).float()


@torch.no_grad()
def encode_reference_provenance(
    *,
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    normalizer: torch.nn.Module,
    expert_buffer: TrajectoryDictBuffer,
    motion_id: torch.Tensor,
    reference_index: torch.Tensor,
    horizon: torch.Tensor,
    sequence_length: int,
    device: str,
) -> torch.Tensor:
    motion_id = motion_id.reshape(-1).to(torch.long)
    reference_index = reference_index.reshape(-1).to(torch.long)
    horizon = horizon.reshape(-1).to(torch.long)
    if bool((horizon <= 0).any()) or bool((horizon > sequence_length).any()):
        raise ValueError("Reference horizons must lie inside the temporal contract")
    offsets = torch.arange(sequence_length, dtype=torch.long)
    indices = reference_index[:, None] + 1 + offsets[None, :]
    if bool((indices < 0).any()) or bool((indices >= len(expert_buffer)).any()):
        raise ValueError("Reference provenance points outside the expert replay")
    expected = motion_id[:, None].expand_as(indices)
    observed = expert_buffer.storage["motion_id"][indices].reshape_as(indices)
    valid = offsets[None, :] < horizon[:, None]
    if not bool((expected[valid] == observed[valid]).all()):
        raise ValueError("Reference provenance crosses an expert motion boundary")
    raw = {
        key: expert_buffer.storage["observation"][key][indices]
        for key in OBSERVATION_KEYS
    }
    flat = _flatten_observation(raw, device)
    with eval_mode(normalizer):
        normalized = normalizer(flat)
    embeddings = encoder(normalized).reshape(indices.shape[0], sequence_length, -1)
    return model.project_z(masked_temporal_mean(embeddings, horizon.to(device))).float()


@torch.no_grad()
def evaluate_frozen_gate(
    *,
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    normalizer: torch.nn.Module,
    replay: FrozenSelectiveReplay,
    expert_buffer: TrajectoryDictBuffer,
    indices: FrozenWindowIndices,
    gate_window: int,
    sequence_length: int,
    thresholds: GateThresholds,
    device: str,
) -> GateEvaluation:
    next_obs = replay.sequence_observation(indices, successor=True)
    policy_z = encode_policy_gate_windows(
        model=model,
        encoder=encoder,
        normalizer=normalizer,
        next_observation=next_obs,
        output_window=gate_window,
        sequence_length=sequence_length,
        device=device,
    )
    motion = replay.sequence_field("prior_motion_id", indices)[:, :gate_window, 0].to(torch.long)
    reference = replay.sequence_field("prior_reference_index", indices)[:, :gate_window, 0].to(torch.long)
    horizon = torch.full_like(reference, sequence_length)
    expert_z = encode_reference_provenance(
        model=model,
        encoder=encoder,
        normalizer=normalizer,
        expert_buffer=expert_buffer,
        motion_id=motion,
        reference_index=reference,
        horizon=horizon,
        sequence_length=sequence_length,
        device=device,
    ).reshape(indices.count, gate_window, -1)
    cosine = F.cosine_similarity(policy_z, expert_z, dim=-1).cpu()
    heading = replay.sequence_field("observation-heading", indices)[..., 0].float()
    pathology = replay.sequence_pathology(indices)
    good, bad_local, semantic_mean, semantic_min = classify_gate_windows(
        cosine=cosine,
        heading_cost=heading,
        pathology=pathology,
        thresholds=thresholds,
    )
    motion_first = motion[:, 0]
    names = getattr(expert_buffer, "file_names", ())
    getup_ids = {
        int(motion_id)
        for motion_id, name in zip(getattr(expert_buffer, "motion_ids", range(len(names))), names)
        if behavior_family_from_name(str(name)) is BehaviorFamily.GETUP_RECOVERY
    }
    getup = torch.zeros(indices.count, dtype=torch.bool)
    for motion_id in getup_ids:
        getup |= motion_first.eq(motion_id)
    state = replay.sequence_field("observation-state", indices).float()
    dof_count = int(model.action_dim)
    upright = -state[..., 2 * dof_count + 2] >= 0.80
    privileged = replay.sequence_field("observation-privileged_state", indices).float()
    upright &= privileged[..., 0] >= 0.65
    tail = min(4, max(1, indices.length - gate_window))
    getup_success = getup & upright[:, -tail:].all(dim=1)
    terminated = replay.sequence_field("transition_terminated", indices).reshape(indices.count, indices.length, -1).any(dim=-1)
    truncated = replay.sequence_field("transition_truncated", indices).reshape(indices.count, indices.length, -1).any(dim=-1)
    # Successful get-up admission is outcome based. Ordinary failed recovery
    # remains UNKNOWN; only local pathology can create BAD frames.
    good = torch.where(getup, good & getup_success, good)
    return GateEvaluation(
        cosine=cosine,
        semantic_mean=semantic_mean,
        semantic_min=semantic_min,
        heading_cost=heading,
        pathology=pathology,
        good_window=good,
        bad_local=bad_local,
        getup=getup,
        getup_success=getup_success,
        transition_done=terminated | truncated,
    )


def gate_report(
    evaluation: GateEvaluation,
    *,
    motion_id: torch.Tensor,
    context_id: torch.Tensor,
    reference_index: torch.Tensor,
    env_id: torch.Tensor,
    expert_buffer: TrajectoryDictBuffer,
    holdout_modulus: int = 5,
) -> dict[str, Any]:
    window = evaluation.cosine.shape[1]
    clean_future = ~evaluation.pathology.any(dim=1) & ~evaluation.transition_done[:, :-1].any(dim=1)
    proxy_good = clean_future & (~evaluation.getup | evaluation.getup_success)
    proxy_bad = evaluation.pathology[:, :window].any(dim=1)
    good = evaluation.good_window
    bad = evaluation.bad_local.any(dim=1)
    report: dict[str, Any] = {
        "windows": int(evaluation.cosine.shape[0]),
        "semantic_mean": _quantiles(evaluation.semantic_mean),
        "semantic_min": _quantiles(evaluation.semantic_min),
        "good_windows": int(good.sum().item()),
        "bad_windows": int(bad.sum().item()),
        "unknown_windows": int((~good & ~bad).sum().item()),
        # These are explicitly proxy precision values. They are independent of
        # B cosine but are not a substitute for outcome/video human review.
        "proxy_good_precision": _ratio(proxy_good, good),
        "proxy_bad_precision": _ratio(proxy_bad, bad),
        "proxy_good_recall": _ratio(good, proxy_good),
        "proxy_bad_recall": _ratio(bad, proxy_bad),
        "getup_windows": int(evaluation.getup.sum().item()),
        "getup_success_windows": int(evaluation.getup_success.sum().item()),
        "validated_getup_windows": int((good & evaluation.getup).sum().item()),
    }

    def independent_counts(mask: torch.Tensor) -> dict[str, int]:
        mask = mask.reshape(-1).to(torch.bool)
        if not bool(mask.any()):
            return {"windows": 0, "contexts": 0, "motions": 0, "heldout_windows": 0}
        selected_motion = motion_id.reshape(-1).to(torch.long)[mask]
        selected_context = context_id.reshape(-1).to(torch.long)[mask]
        selected_reference_bin = torch.div(
            reference_index.reshape(-1).to(torch.long)[mask],
            window,
            rounding_mode="floor",
        )
        selected_env = env_id.reshape(-1).to(torch.long)[mask]
        window_key = torch.stack(
            (selected_env, selected_context, selected_motion, selected_reference_bin), dim=1
        )
        context_key = torch.stack((selected_env, selected_context, selected_motion), dim=1)
        split_key = selected_motion * 73_856_093 + selected_context * 19_349_663 + selected_env * 2_654_435_761
        heldout = torch.remainder(split_key, int(holdout_modulus)).eq(0)
        return {
            "windows": int(torch.unique(window_key, dim=0).shape[0]),
            "contexts": int(torch.unique(context_key, dim=0).shape[0]),
            "motions": int(torch.unique(selected_motion).numel()),
            "heldout_windows": int(torch.unique(window_key[heldout], dim=0).shape[0]),
        }

    report["independent_support"] = {
        "validated": independent_counts(good),
        "bad": independent_counts(bad),
    }
    names = getattr(expert_buffer, "file_names", ())
    motion_id = motion_id.reshape(-1).to(torch.long)
    by_family: dict[str, Any] = {}
    for family in BehaviorFamily:
        ids = [
            int(identifier)
            for identifier, name in zip(getattr(expert_buffer, "motion_ids", range(len(names))), names)
            if behavior_family_from_name(str(name)) is family
        ]
        family_mask = torch.zeros_like(motion_id, dtype=torch.bool)
        for identifier in ids:
            family_mask |= motion_id.eq(identifier)
        if not bool(family_mask.any()):
            continue
        by_family[family.name.lower()] = {
            "windows": int(family_mask.sum().item()),
            "good": int((family_mask & good).sum().item()),
            "bad": int((family_mask & bad).sum().item()),
            "semantic_mean": _quantiles(evaluation.semantic_mean[family_mask]),
        }
    report["by_behavior_family"] = by_family
    return report


def _take_policy_frames(
    replay: FrozenSelectiveReplay,
    indices: FrozenWindowIndices,
    selection: torch.Tensor,
    *,
    gate_window: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    selection = selection.to(torch.bool)
    if selection.shape != (indices.count, gate_window):
        raise ValueError("Policy frame selector has the wrong shape")
    time = torch.from_numpy(indices.time[:, :gate_window].copy())[selection].numpy()
    env = torch.from_numpy(indices.env[:, :gate_window].copy())[selection].numpy()
    obs = {
        key: torch.from_numpy(np.asarray(replay.arrays[f"observation-{key}"][time, env]).copy())
        for key in OBSERVATION_KEYS
    }
    motion = torch.from_numpy(np.asarray(replay.arrays["prior_motion_id"][time, env]).copy()).reshape(-1).to(torch.long)
    reference = torch.from_numpy(np.asarray(replay.arrays["prior_reference_index"][time, env]).copy()).reshape(-1).to(torch.long)
    return obs, motion, reference


@torch.no_grad()
def build_offline_prior_dataset(
    *,
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    normalizer: torch.nn.Module,
    replay: FrozenSelectiveReplay,
    expert_buffer: TrajectoryDictBuffer,
    indices: FrozenWindowIndices,
    evaluation: GateEvaluation,
    gate_window: int,
    sequence_length: int,
    device: str,
) -> OfflinePriorDataset:
    validated_selector = evaluation.good_window[:, None].expand(-1, gate_window)
    bad_selector = evaluation.bad_local[:, :gate_window]
    validated_obs, validated_motion, validated_reference = _take_policy_frames(
        replay, indices, validated_selector, gate_window=gate_window
    )
    bad_obs, bad_motion, bad_reference = _take_policy_frames(
        replay, indices, bad_selector, gate_window=gate_window
    )
    if validated_motion.numel() == 0 or bad_motion.numel() == 0:
        raise RuntimeError("Frozen gate produced no usable validated or bad frames for offline D")
    validated_horizon = torch.full_like(validated_reference, sequence_length)
    bad_horizon = torch.full_like(bad_reference, sequence_length)
    validated_z = encode_reference_provenance(
        model=model,
        encoder=encoder,
        normalizer=normalizer,
        expert_buffer=expert_buffer,
        motion_id=validated_motion,
        reference_index=validated_reference,
        horizon=validated_horizon,
        sequence_length=sequence_length,
        device=device,
    ).cpu()
    bad_z = encode_reference_provenance(
        model=model,
        encoder=encoder,
        normalizer=normalizer,
        expert_buffer=expert_buffer,
        motion_id=bad_motion,
        reference_index=bad_reference,
        horizon=bad_horizon,
        sequence_length=sequence_length,
        device=device,
    ).cpu()

    # Match expert positives to the policy provenance distribution instead of
    # allowing a behavior-mixture shortcut. One expert frame is paired with
    # the exact same p -> z coordinate used by each V/B policy frame.
    expert_motion = torch.cat((validated_motion, bad_motion))
    expert_reference = torch.cat((validated_reference, bad_reference))
    expert_z = torch.cat((validated_z, bad_z))
    expert_obs = {
        key: expert_buffer.storage["observation"][key][expert_reference].clone()
        for key in OBSERVATION_KEYS
    }
    confidence = evaluation.semantic_mean[:, None].expand(-1, gate_window)[validated_selector].abs().clamp(0.0, 1.0)
    return OfflinePriorDataset(
        expert_obs=expert_obs,
        validated_obs=validated_obs,
        bad_obs=bad_obs,
        expert_z=expert_z,
        validated_z=validated_z,
        bad_z=bad_z,
        validated_confidence=confidence,
        expert_motion_id=expert_motion,
        validated_motion_id=validated_motion,
        bad_motion_id=bad_motion,
    )


def _normalize_pool(
    obs: Mapping[str, torch.Tensor],
    *,
    normalizer: torch.nn.Module,
    device: str,
    chunk_size: int = 16_384,
) -> dict[str, torch.Tensor]:
    output = {key: [] for key in obs}
    count = next(iter(obs.values())).shape[0]
    with torch.no_grad(), eval_mode(normalizer):
        for start in range(0, count, chunk_size):
            chunk = {key: value[start : start + chunk_size].to(device) for key, value in obs.items()}
            normalized = normalizer(chunk)
            for key, value in normalized.items():
                if key in output:
                    output[key].append(value.detach().cpu())
    return {key: torch.cat(values) for key, values in output.items()}


def _sample_pool(
    obs: Mapping[str, torch.Tensor],
    z: torch.Tensor,
    indices: torch.Tensor,
    count: int,
    *,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if indices.numel() == 0:
        raise RuntimeError("Cannot sample an empty offline prior pool")
    selected = indices[torch.randint(indices.numel(), (count,))]
    return (
        {key: value[selected].to(device, non_blocking=True) for key, value in obs.items()},
        z[selected].to(device, non_blocking=True),
        selected,
    )


def _gradient_penalty(
    discriminator: torch.nn.Module,
    expert_obs: Mapping[str, torch.Tensor],
    expert_z: torch.Tensor,
    bad_obs: Mapping[str, torch.Tensor],
    bad_z: torch.Tensor,
) -> torch.Tensor:
    count = expert_z.shape[0]
    alpha = torch.rand(count, 1, device=expert_z.device)
    interpolated_obs: dict[str, torch.Tensor] = {}
    differentiable: list[torch.Tensor] = []
    for key in expert_obs:
        value = (alpha * expert_obs[key] + (1.0 - alpha) * bad_obs[key]).requires_grad_(True)
        interpolated_obs[key] = value
        differentiable.append(value)
    interpolated_z = (alpha * expert_z + (1.0 - alpha) * bad_z).requires_grad_(True)
    logits = discriminator.compute_logits(interpolated_obs, interpolated_z)
    gradients = autograd.grad(
        outputs=logits,
        inputs=[*differentiable, interpolated_z],
        grad_outputs=torch.ones_like(logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )
    flat = torch.cat([value for value in gradients if value is not None], dim=1)
    return ((flat.norm(2, dim=1) - 1.0) ** 2).mean()


@torch.no_grad()
def _evaluate_discriminator_split(
    discriminator: torch.nn.Module,
    dataset: OfflinePriorDataset,
    split: Mapping[str, torch.Tensor],
    *,
    device: str,
    batch_size: int = 4096,
) -> dict[str, Any]:
    logits: dict[str, list[torch.Tensor]] = {"expert": [], "validated": [], "bad": []}
    sources = {
        "expert": (dataset.expert_obs, dataset.expert_z),
        "validated": (dataset.validated_obs, dataset.validated_z),
        "bad": (dataset.bad_obs, dataset.bad_z),
    }
    for name, (obs, z) in sources.items():
        selected = split[name]
        for start in range(0, selected.numel(), batch_size):
            index = selected[start : start + batch_size]
            batch_obs = {key: value[index].to(device) for key, value in obs.items()}
            batch_z = z[index].to(device)
            logits[name].append(discriminator.compute_logits(batch_obs, batch_z).float().cpu())
    merged = {
        key: torch.cat(value).reshape(-1) if value else torch.empty(0)
        for key, value in logits.items()
    }
    reward = {
        key: torch.clamp(1.0 - 0.25 * (value - 1.0).square(), min=0.0)
        for key, value in merged.items()
    }
    # The shared helper computes exact pairwise comparisons and therefore has
    # quadratic memory. A fixed, deterministic cap keeps large frozen banks
    # from turning a calibration diagnostic into an accidental OOM.
    auc_cap = 4096
    auc_values = {
        key: value[:auc_cap]
        for key, value in merged.items()
    }
    return {
        "count": {key: int(value.numel()) for key, value in merged.items()},
        "logits": {key: _quantiles(value) for key, value in merged.items()},
        "reward": {key: _quantiles(value) for key, value in reward.items()},
        "reward_zero_fraction": {
            key: float(value.eq(0).float().mean().item()) if value.numel() else float("nan")
            for key, value in reward.items()
        },
        "auc_expert_validated": float(
            approximate_pairwise_auc(auc_values["expert"], auc_values["validated"]).item()
        ),
        "auc_validated_bad": float(
            approximate_pairwise_auc(auc_values["validated"], auc_values["bad"]).item()
        ),
        "auc_sample_cap": auc_cap,
    }


def fit_offline_discriminator(
    *,
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    dataset: OfflinePriorDataset,
    steps: int,
    batch_size: int,
    learning_rate: float,
    grad_penalty: float,
    validated_weight: float,
    holdout_modulus: int,
    holdout_remainder: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("Offline discriminator steps must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset.expert_obs = _normalize_pool(dataset.expert_obs, normalizer=normalizer, device=device)
    dataset.validated_obs = _normalize_pool(dataset.validated_obs, normalizer=normalizer, device=device)
    dataset.bad_obs = _normalize_pool(dataset.bad_obs, normalizer=normalizer, device=device)
    discriminator = copy.deepcopy(model._discriminator).to(device)
    discriminator.apply(weight_init)
    discriminator.train().requires_grad_(True)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    motion = {
        "expert": dataset.expert_motion_id,
        "validated": dataset.validated_motion_id,
        "bad": dataset.bad_motion_id,
    }
    splits: dict[str, dict[str, torch.Tensor]] = {"train": {}, "holdout": {}}
    for source, ids in motion.items():
        holdout = torch.remainder(ids, holdout_modulus).eq(holdout_remainder)
        splits["train"][source] = torch.nonzero(~holdout, as_tuple=False).reshape(-1)
        splits["holdout"][source] = torch.nonzero(holdout, as_tuple=False).reshape(-1)
        if splits["train"][source].numel() == 0 or splits["holdout"][source].numel() == 0:
            raise RuntimeError(f"Motion-isolated split has an empty {source} pool")

    expert_count = max(1, round(batch_size * 0.50))
    validated_count = max(1, round(batch_size * 0.33))
    bad_count = max(1, batch_size - expert_count - validated_count)
    history: list[dict[str, float]] = []
    for step in range(steps):
        expert_obs, expert_z, _ = _sample_pool(
            dataset.expert_obs, dataset.expert_z, splits["train"]["expert"], expert_count, device=device
        )
        validated_obs, validated_z, validated_index = _sample_pool(
            dataset.validated_obs,
            dataset.validated_z,
            splits["train"]["validated"],
            validated_count,
            device=device,
        )
        bad_obs, bad_z, _ = _sample_pool(
            dataset.bad_obs, dataset.bad_z, splits["train"]["bad"], bad_count, device=device
        )
        expert_logits = discriminator.compute_logits(expert_obs, expert_z)
        validated_logits = discriminator.compute_logits(validated_obs, validated_z)
        bad_logits = discriminator.compute_logits(bad_obs, bad_z)
        expert_loss = 0.50 * 0.5 * (expert_logits - 1.0).square().mean()
        confidence = dataset.validated_confidence[validated_index].to(device).reshape_as(validated_logits)
        validated_error = 0.5 * (validated_logits - 1.0).square()
        validated_loss = 0.33 * float(validated_weight) * (
            (validated_error * confidence).sum() / confidence.sum().clamp_min(1.0)
        )
        bad_loss = 0.17 * 0.5 * (bad_logits + 1.0).square().mean()
        data_loss = expert_loss + validated_loss + bad_loss
        gp = torch.zeros((), device=device)
        if grad_penalty > 0.0:
            count = min(expert_count, bad_count)
            gp = _gradient_penalty(
                discriminator,
                {key: value[:count] for key, value in expert_obs.items()},
                expert_z[:count],
                {key: value[:count] for key, value in bad_obs.items()},
                bad_z[:count],
            )
        loss = data_loss + float(grad_penalty) * gp
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 20) == 0 or step + 1 == steps:
            history.append(
                {
                    "step": int(step + 1),
                    "loss": float(loss.detach().item()),
                    "data_loss": float(data_loss.detach().item()),
                    "gp": float(gp.detach().item()),
                }
            )
    discriminator.eval().requires_grad_(False)
    return {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "grad_penalty": float(grad_penalty),
        "validated_weight": float(validated_weight),
        "motion_holdout": {
            "modulus": int(holdout_modulus),
            "remainder": int(holdout_remainder),
        },
        "history": history,
        "train": _evaluate_discriminator_split(discriminator, dataset, splits["train"], device=device),
        "holdout": _evaluate_discriminator_split(discriminator, dataset, splits["holdout"], device=device),
    }


def _review_manifest(
    *,
    indices: FrozenWindowIndices,
    replay: FrozenSelectiveReplay,
    evaluation: GateEvaluation,
    expert_buffer: TrajectoryDictBuffer,
    per_bucket: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    motion = replay.sequence_field("prior_motion_id", indices)[:, 0, 0].to(torch.long)
    context = replay.sequence_field("heading_context_id", indices)[:, 0, 0].to(torch.long)
    reference = replay.sequence_field("prior_reference_index", indices)[:, 0, 0].to(torch.long)
    names = getattr(expert_buffer, "file_names", ())
    buckets = {
        "good": evaluation.good_window,
        "bad": evaluation.bad_local.any(dim=1),
        "unknown": ~evaluation.good_window & ~evaluation.bad_local.any(dim=1),
        "getup_success": evaluation.getup_success,
    }
    output: list[dict[str, Any]] = []
    for bucket, mask in buckets.items():
        candidates = torch.nonzero(mask, as_tuple=False).reshape(-1).cpu().numpy()
        rng.shuffle(candidates)
        for row in candidates[:per_bucket]:
            identifier = int(motion[row].item())
            output.append(
                {
                    "bucket": bucket,
                    "buffer_time": int(indices.time[row, 0]),
                    "environment": int(indices.env[row, 0]),
                    "context_id": int(context[row].item()),
                    "motion_id": identifier,
                    "motion_name": str(names[identifier]) if 0 <= identifier < len(names) else None,
                    "reference_index": int(reference[row].item()),
                    "semantic_mean": float(evaluation.semantic_mean[row].item()),
                    "semantic_min": float(evaluation.semantic_min[row].item()),
                    "pathology_frames": int(evaluation.pathology[row].sum().item()),
                    "done_frames": int(evaluation.transition_done[row].sum().item()),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--buffer-rank", type=int, default=0)
    parser.add_argument("--expert-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--windows", type=int, default=4096)
    parser.add_argument("--gate-window", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4831)
    parser.add_argument("--review-per-bucket", type=int, default=32)
    parser.add_argument("--good-cosine-mean", type=float, default=0.75)
    parser.add_argument("--good-cosine-min", type=float, default=0.35)
    parser.add_argument("--bad-cosine-mean", type=float, default=0.0)
    parser.add_argument("--bad-sustain-fraction", type=float, default=0.5)
    parser.add_argument("--good-heading-cost-mean-max", type=float, default=0.30)
    parser.add_argument("--bad-heading-cost-mean-min", type=float, default=1.0)
    parser.add_argument("--skip-offline-d", action="store_true")
    parser.add_argument("--offline-d-steps", type=int, default=2000)
    parser.add_argument("--offline-d-batch-size", type=int, default=1024)
    parser.add_argument("--offline-d-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--offline-d-grad-penalty", type=float, default=10.0)
    parser.add_argument("--validated-weight", type=float, default=0.5)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else run_dir / "checkpoint"
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expert_cache = args.expert_cache.expanduser().resolve()
    load_device = checkpoint_load_device(args.device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=load_device)
    model.to(args.device).eval()
    training_state = torch.load(
        checkpoint_dir / "training_state.pth",
        map_location="cpu",
        weights_only=True,
    )
    encoder, normalizer = _snapshot_modules(model, training_state, args.device)
    expert_buffer = _load_expert_buffer(expert_cache)
    replay = FrozenSelectiveReplay(checkpoint_dir / "buffers" / f"train_rank_{args.buffer_rank}")
    total = int(args.gate_window + args.sequence_length)
    indices = replay.sample_exact_tracking_windows(
        args.windows,
        length=total,
        seed=args.seed,
    )
    thresholds = GateThresholds(
        good_cosine_mean=args.good_cosine_mean,
        good_cosine_min=args.good_cosine_min,
        bad_cosine_mean=args.bad_cosine_mean,
        bad_sustain_fraction=args.bad_sustain_fraction,
        good_heading_cost_mean_max=args.good_heading_cost_mean_max,
        bad_heading_cost_mean_min=args.bad_heading_cost_mean_min,
    )
    evaluation = evaluate_frozen_gate(
        model=model,
        encoder=encoder,
        normalizer=normalizer,
        replay=replay,
        expert_buffer=expert_buffer,
        indices=indices,
        gate_window=args.gate_window,
        sequence_length=args.sequence_length,
        thresholds=thresholds,
        device=args.device,
    )
    motion = replay.sequence_field("prior_motion_id", indices)[:, 0, 0]
    context = replay.sequence_field("heading_context_id", indices)[:, 0, 0]
    reference = replay.sequence_field("prior_reference_index", indices)[:, 0, 0]
    env = torch.from_numpy(indices.env[:, 0]).to(torch.long)
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "buffer_rank": int(args.buffer_rank),
        "expert_cache": str(expert_cache),
        "windows": int(args.windows),
        "gate_window": int(args.gate_window),
        "sequence_length": int(args.sequence_length),
        "thresholds": asdict(thresholds),
        "gate": gate_report(
            evaluation,
            motion_id=motion,
            context_id=context,
            reference_index=reference,
            env_id=env,
            expert_buffer=expert_buffer,
            holdout_modulus=args.holdout_modulus,
        ),
    }
    manifest = _review_manifest(
        indices=indices,
        replay=replay,
        evaluation=evaluation,
        expert_buffer=expert_buffer,
        per_bucket=args.review_per_bucket,
        seed=args.seed + 1,
    )
    with (output_dir / "review_manifest.jsonl").open("w") as stream:
        for row in manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "gate_windows.npz",
        time=indices.time,
        env=indices.env,
        cosine=evaluation.cosine.numpy(),
        semantic_mean=evaluation.semantic_mean.numpy(),
        semantic_min=evaluation.semantic_min.numpy(),
        good=evaluation.good_window.numpy(),
        bad_local=evaluation.bad_local.numpy(),
        motion_id=motion.numpy(),
    )
    if not args.skip_offline_d:
        dataset = build_offline_prior_dataset(
            model=model,
            encoder=encoder,
            normalizer=normalizer,
            replay=replay,
            expert_buffer=expert_buffer,
            indices=indices,
            evaluation=evaluation,
            gate_window=args.gate_window,
            sequence_length=args.sequence_length,
            device=args.device,
        )
        report["offline_discriminator"] = fit_offline_discriminator(
            model=model,
            normalizer=normalizer,
            dataset=dataset,
            steps=args.offline_d_steps,
            batch_size=args.offline_d_batch_size,
            learning_rate=args.offline_d_learning_rate,
            grad_penalty=args.offline_d_grad_penalty,
            validated_weight=args.validated_weight,
            holdout_modulus=args.holdout_modulus,
            holdout_remainder=args.holdout_remainder,
            seed=args.seed,
            device=args.device,
        )
    report_path = output_dir / "selective_prior_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[INFO] wrote {report_path}")


if __name__ == "__main__":
    main()
