"""Selective online expansion for the behavior-prior support.

The discriminator is deliberately absent from admission decisions.  A slow
behavior encoder snapshot labels only high-confidence policy windows; all
ambiguous samples remain UNKNOWN and are invisible to D, Q_D, and Actor-D.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

import torch

PROVENANCE_SCHEMA_VERSION = 2
TEMPORAL_ENCODING_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class TemporalEncodingContract:
    """The single source of truth for turning reference futures into z.

    Tracking contexts are encoded from *next* observations and average at
    most ``sequence_length`` future frames. The explicit contract version is
    part of every active-prior fingerprint so changing this rule can never
    silently reinterpret an existing provenance bank.
    """

    sequence_length: int
    uses_next_observation: bool = True
    boundary_clipping: bool = True
    version: int = TEMPORAL_ENCODING_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.uses_next_observation:
            raise ValueError("Selective-prior tracking provenance must encode next observations")
        if not self.boundary_clipping:
            raise ValueError("Selective-prior tracking provenance requires context-tail clipping")

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceProvenanceBatch:
    """Tensor representation of immutable exact-tracking behavior identity."""

    motion_id: torch.Tensor
    reference_index: torch.Tensor
    reference_horizon: torch.Tensor
    next_motion_id: torch.Tensor | None = None
    next_reference_index: torch.Tensor | None = None
    next_reference_horizon: torch.Tensor | None = None
    context_id: torch.Tensor | None = None
    source_type: torch.Tensor | None = None

    def validate(self, *, require_next: bool = False) -> None:
        current = (self.motion_id, self.reference_index, self.reference_horizon)
        if any(value.shape != self.motion_id.shape for value in current):
            raise ValueError("Current provenance tensors must have identical shapes")
        if bool((self.motion_id < 0).any()) or bool((self.reference_index < 0).any()):
            raise ValueError("Reference provenance requires non-negative motion and reference indices")
        if bool((self.reference_horizon <= 0).any()):
            raise ValueError("Reference provenance horizon must be positive")
        next_values = (self.next_motion_id, self.next_reference_index, self.next_reference_horizon)
        if require_next and any(value is None for value in next_values):
            raise ValueError("Bellman provenance requires explicit p_t and p_t+1")
        for value in next_values:
            if value is not None and value.shape != self.motion_id.shape:
                raise ValueError("Next provenance tensors must match current provenance shape")

    def successor(self) -> "ReferenceProvenanceBatch":
        self.validate(require_next=True)
        assert self.next_motion_id is not None
        assert self.next_reference_index is not None
        assert self.next_reference_horizon is not None
        return ReferenceProvenanceBatch(
            motion_id=self.next_motion_id,
            reference_index=self.next_reference_index,
            reference_horizon=self.next_reference_horizon,
            context_id=self.context_id,
            source_type=self.source_type,
        )


@dataclass(frozen=True)
class PriorCoordinateContract:
    """Versioned frozen coordinate system bound to one active support bank."""

    version: int
    bank_version: int
    encoder_fingerprint: str
    normalizer_fingerprint: str
    expert_dataset_fingerprint: str
    temporal_contract: dict[str, Any]
    provenance_schema_version: int = PROVENANCE_SCHEMA_VERSION

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def state_dict(self) -> dict[str, Any]:
        state = asdict(self)
        state["fingerprint"] = self.fingerprint()
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PriorCoordinateContract":
        payload = {key: value for key, value in state.items() if key != "fingerprint"}
        contract = cls(**payload)
        expected = state.get("fingerprint")
        if expected is not None and expected != contract.fingerprint():
            raise ValueError("Prior coordinate contract fingerprint mismatch")
        return contract


def module_state_fingerprint(module: torch.nn.Module) -> str:
    """Hash a frozen module's exact state without serialization metadata."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def masked_temporal_mean(embeddings: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
    """Average the first ``horizon`` embeddings for independently clipped rows."""

    if embeddings.ndim != 3:
        raise ValueError(f"embeddings must be [batch,time,dim], got {tuple(embeddings.shape)}")
    horizon = horizon.to(device=embeddings.device, dtype=torch.long).reshape(-1)
    if horizon.shape[0] != embeddings.shape[0]:
        raise ValueError("horizon batch dimension does not match embeddings")
    if bool((horizon <= 0).any()) or bool((horizon > embeddings.shape[1]).any()):
        raise ValueError("horizon must be in [1, embeddings.shape[1]]")
    ages = torch.arange(embeddings.shape[1], device=embeddings.device)[None, :]
    mask = ages < horizon[:, None]
    return (embeddings * mask[..., None]).sum(dim=1) / horizon[:, None].to(embeddings.dtype)


def future_window_means(embeddings: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
    """Apply the tracking future-average rule to policy next-state embeddings.

    ``embeddings`` contains B(next_obs) for a contiguous sequence. ``horizons``
    is [batch, output_time], so output t uses embeddings[t:t+h_t].
    """

    if embeddings.ndim != 3:
        raise ValueError(f"embeddings must be [batch,time,dim], got {tuple(embeddings.shape)}")
    horizons = horizons.to(device=embeddings.device, dtype=torch.long)
    if horizons.ndim != 2 or horizons.shape[0] != embeddings.shape[0]:
        raise ValueError("horizons must be [batch,output_time]")
    output_time = horizons.shape[1]
    if output_time > embeddings.shape[1]:
        raise ValueError("output_time exceeds the available embedding sequence")
    starts = torch.arange(output_time, device=embeddings.device)[None, :].expand_as(horizons)
    ends = starts + horizons
    if bool((horizons <= 0).any()) or bool((ends > embeddings.shape[1]).any()):
        raise ValueError("Every future window must stay inside the supplied sequence")
    prefix = torch.cat((torch.zeros_like(embeddings[:, :1]), embeddings.cumsum(dim=1)), dim=1)
    gather_shape = (*ends.shape, embeddings.shape[-1])
    end_values = torch.gather(prefix, 1, ends[..., None].expand(gather_shape))
    start_values = torch.gather(prefix, 1, starts[..., None].expand(gather_shape))
    return (end_values - start_values) / horizons[..., None].to(embeddings.dtype)


class PriorLabel(IntEnum):
    UNKNOWN = 0
    VALIDATED = 1
    BAD = 2
    # Candidate is deliberately not part of FINALIZED_LABELS. It has passed
    # one D-independent gate observation but must age and pass a later,
    # outcome-aware revalidation before entering any prior objective.
    CANDIDATE = 3


class PriorPhase(IntEnum):
    BOOTSTRAP = 0
    FIT_D = 1
    FIT_QD = 2
    ACTOR_PRIOR = 3


class BehaviorFamily(IntEnum):
    OTHER = 0
    LOCOMOTION = 1
    DANCE_AGILE = 2
    GETUP_RECOVERY = 3


def behavior_family_from_name(name: str) -> BehaviorFamily:
    """Coarse sampling stratum only; never part of latent derivation."""

    normalized = str(name).lower().replace("_", "").replace("-", "")
    if any(token in normalized for token in ("getup", "fallandgetup", "recovery")):
        return BehaviorFamily.GETUP_RECOVERY
    if any(token in normalized for token in ("dance", "cartwheel", "jump", "hop", "kick")):
        return BehaviorFamily.DANCE_AGILE
    if any(token in normalized for token in ("walk", "run", "jog", "locomotion")):
        return BehaviorFamily.LOCOMOTION
    return BehaviorFamily.OTHER


def independent_support_counts(
    mask: torch.Tensor,
    *,
    motion_id: torch.Tensor,
    context_id: torch.Tensor,
    reference_index: torch.Tensor,
    window: int,
) -> tuple[int, int, int]:
    """Count independent non-overlap windows, contexts, and motions."""

    if window <= 0:
        raise ValueError("window must be positive")
    if mask.shape != motion_id.shape or mask.shape != context_id.shape or mask.shape != reference_index.shape:
        raise ValueError("Support-count tensors must have identical [time,env] shapes")
    selected = mask.to(torch.bool)
    if not bool(selected.any()):
        return 0, 0, 0
    time, env = motion_id.shape
    env_id = torch.arange(env, device=motion_id.device, dtype=torch.long)[None, :].expand(time, -1)
    motion = motion_id.to(torch.long)[selected]
    context = context_id.to(torch.long)[selected]
    reference_bin = torch.div(reference_index.to(torch.long)[selected], int(window), rounding_mode="floor")
    environment = env_id[selected]
    # Hash tuples before unique; constants are pairwise odd and counts are
    # diagnostic/readiness values, never persistent semantic identifiers.
    context_key = environment * 2_654_435_761 + context * 19_349_663 + motion * 73_856_093
    window_key = context_key * 83_492_791 + reference_bin
    return (
        int(torch.unique(window_key).numel()),
        int(torch.unique(context_key).numel()),
        int(torch.unique(motion).numel()),
    )


FINALIZED_LABELS = (int(PriorLabel.VALIDATED), int(PriorLabel.BAD))


@dataclass
class SelectivePriorState:
    phase: int = int(PriorPhase.BOOTSTRAP)
    # ``bank_version`` always names the immutable ACTIVE bank consumed by
    # D/Q_D/Actor-D. ``gate_teacher_version`` names the independently built
    # SHADOW verifier bank and may advance without invalidating ACTIVE.
    bank_version: int = 0
    active_teacher_version: int = 0
    gate_teacher_version: int = 0
    shadow_building: int = 0
    shadow_started_update: int = 0
    shadow_started_policy_version: int = 0
    last_bank_swap_update: int = 0
    last_bank_swap_policy_version: int = 0
    prior_coordinate_version: int = 0
    discriminator_version: int = 0
    qd_reward_version: int = -1
    discriminator_bank_version: int = -1
    qd_bank_version: int = -1
    update_count: int = 0
    behavior_encoder_version: int = 0
    policy_version: int = 0
    discriminator_update_count: int = 0
    qd_update_count: int = 0
    d_ready_streak: int = 0
    qd_ready_streak: int = 0
    d_health_failure_streak: int = 0
    qd_health_failure_streak: int = 0

    def state_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SelectivePriorState":
        known = cls.__dataclass_fields__
        return cls(**{key: int(value) for key, value in state.items() if key in known})

    @property
    def phase_enum(self) -> PriorPhase:
        return PriorPhase(self.phase)

    def set_phase(self, phase: PriorPhase) -> None:
        self.phase = int(phase)


def shadow_refresh_due(
    *,
    phase: PriorPhase | int,
    shadow_building: bool,
    policy_version: int,
    last_bank_swap_policy_version: int,
    refresh_policy_updates: int,
) -> bool:
    """Return whether ACTIVE should start building its next SHADOW bank.

    ``policy_version`` advances once per optimizer update, whereas selective
    gate ``update_count`` advances once per collector step. Mixing these two
    clocks made the old refresh interval outlive the policy-age validity of
    every ACTIVE sample. Keep this decision entirely in optimizer-update
    units so it can be compared directly with ``max_policy_version_age``.
    """

    if refresh_policy_updates <= 0:
        raise ValueError("refresh_policy_updates must be positive")
    return (
        PriorPhase(int(phase)) is PriorPhase.ACTOR_PRIOR
        and not bool(shadow_building)
        and int(policy_version) - int(last_bank_swap_policy_version) >= int(refresh_policy_updates)
    )


def finalized_mask(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(torch.long)
    return (labels == int(PriorLabel.VALIDATED)) | (labels == int(PriorLabel.BAD))


def fresh_mask(label_step: torch.Tensor, *, step: int, ttl_steps: int) -> torch.Tensor:
    if ttl_steps <= 0:
        return label_step > 0
    age = int(step) - label_step.to(torch.long)
    return (label_step > 0) & (age >= 0) & (age <= int(ttl_steps))


def active_finalized_mask(
    labels: torch.Tensor,
    label_step: torch.Tensor,
    *,
    step: int,
    ttl_steps: int,
) -> torch.Tensor:
    return finalized_mask(labels) & fresh_mask(label_step, step=step, ttl_steps=ttl_steps)


def qd_interior_mask(
    *,
    active: torch.Tensor,
    context_id: torch.Tensor,
    transition_done: torch.Tensor,
    successor_available: torch.Tensor,
) -> torch.Tensor:
    """Mask Q_D transitions without treating UNKNOWN as an auxiliary terminal.

    Real environment endings are valid terminal targets.  Nonterminal samples
    require a fresh finalized successor under the same behavior context.
    """

    active = active.to(torch.bool)
    done = transition_done.to(torch.bool)
    successor_active = torch.roll(active, shifts=-1, dims=0)
    successor_context = torch.roll(context_id.to(torch.long), shifts=-1, dims=0)
    same_context = context_id.to(torch.long) == successor_context
    interior = successor_available.to(torch.bool) & successor_active & same_context
    return active & (done | interior)


def actor_prior_interior_mask(
    *,
    active: torch.Tensor,
    context_id: torch.Tensor,
    transition_done: torch.Tensor,
    successor_available: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Require K consecutive fresh finalized states for Actor-D."""

    if horizon <= 0:
        raise ValueError(f"Actor-D interior horizon must be positive, got {horizon}")
    result = active.to(torch.bool).clone()
    base_context = context_id.to(torch.long)
    for offset in range(horizon):
        if offset:
            shifted_active = torch.roll(active.to(torch.bool), shifts=-offset, dims=0)
            shifted_context = torch.roll(base_context, shifts=-offset, dims=0)
            result &= shifted_active & (shifted_context == base_context)
        if offset < horizon - 1:
            result &= torch.roll(successor_available.to(torch.bool), shifts=-offset, dims=0)
            result &= ~torch.roll(transition_done.to(torch.bool), shifts=-offset, dims=0)
    return result


def approximate_pairwise_auc(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Exact empirical pairwise AUC with half credit for ties."""

    positive = positive.detach().float().reshape(-1)
    negative = negative.detach().float().reshape(-1)
    if positive.numel() == 0 or negative.numel() == 0:
        return torch.full((), float("nan"), device=positive.device)
    comparison = positive[:, None] - negative[None, :]
    return (comparison.gt(0).float() + 0.5 * comparison.eq(0).float()).mean()


def resolve_prior_proposals(
    *,
    good: torch.Tensor,
    bad: torch.Tensor,
    getup: torch.Tensor,
    getup_success: torch.Tensor,
    old_label: torch.Tensor,
    old_label_teacher_version: torch.Tensor,
    old_candidate_step: torch.Tensor,
    old_candidate_teacher_version: torch.Tensor,
    step: int,
    teacher_version: int,
    candidate_min_age_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve overlapping gate evidence without relying on enum ordering."""

    good_only = good.to(torch.bool) & ~bad.to(torch.bool)
    bad_only = bad.to(torch.bool) & ~good.to(torch.bool)
    old_label = old_label.to(torch.long)
    old_candidate_step = old_candidate_step.to(torch.long)
    same_teacher_candidate = old_label.eq(int(PriorLabel.CANDIDATE))
    same_teacher_candidate &= old_candidate_teacher_version.to(torch.long).eq(int(teacher_version))
    aged_candidate = same_teacher_candidate & (
        int(step) - old_candidate_step >= int(candidate_min_age_steps)
    )
    second_pass_validated = good_only & aged_candidate & ~getup.to(torch.bool)
    getup_validated = good_only & getup.to(torch.bool) & getup_success.to(torch.bool)
    # A new verifier coordinate system must re-admit old support through the
    # delayed candidate path.  A single favorable pass cannot silently renew
    # a label issued by another teacher snapshot.
    revalidated = good_only & old_label.eq(int(PriorLabel.VALIDATED))
    revalidated &= old_label_teacher_version.to(torch.long).eq(int(teacher_version))
    validated = second_pass_validated | getup_validated | revalidated
    candidate = good_only & ~validated

    label = torch.full_like(old_label, int(PriorLabel.UNKNOWN), dtype=torch.int8)
    label[candidate] = int(PriorLabel.CANDIDATE)
    label[validated] = int(PriorLabel.VALIDATED)
    label[bad_only] = int(PriorLabel.BAD)
    candidate_step = torch.where(
        candidate & same_teacher_candidate,
        old_candidate_step,
        torch.full_like(old_candidate_step, int(step)),
    )
    return label, candidate_step


__all__ = [
    "FINALIZED_LABELS",
    "BehaviorFamily",
    "PROVENANCE_SCHEMA_VERSION",
    "TEMPORAL_ENCODING_CONTRACT_VERSION",
    "PriorCoordinateContract",
    "PriorLabel",
    "PriorPhase",
    "ReferenceProvenanceBatch",
    "SelectivePriorState",
    "TemporalEncodingContract",
    "active_finalized_mask",
    "actor_prior_interior_mask",
    "approximate_pairwise_auc",
    "behavior_family_from_name",
    "finalized_mask",
    "fresh_mask",
    "future_window_means",
    "independent_support_counts",
    "masked_temporal_mean",
    "module_state_fingerprint",
    "qd_interior_mask",
    "resolve_prior_proposals",
    "shadow_refresh_due",
]
