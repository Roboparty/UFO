"""Selective online expansion for the behavior-prior support.

The discriminator is deliberately absent from admission decisions.  A slow
behavior encoder snapshot labels only high-confidence policy windows; all
ambiguous samples remain UNKNOWN and are invisible to D, Q_D, and Actor-D.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

import torch


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


FINALIZED_LABELS = (int(PriorLabel.VALIDATED), int(PriorLabel.BAD))


@dataclass
class SelectivePriorState:
    phase: int = int(PriorPhase.BOOTSTRAP)
    bank_version: int = 0
    gate_teacher_version: int = 0
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
    "PriorLabel",
    "PriorPhase",
    "SelectivePriorState",
    "active_finalized_mask",
    "actor_prior_interior_mask",
    "approximate_pairwise_auc",
    "finalized_mask",
    "fresh_mask",
    "qd_interior_mask",
    "resolve_prior_proposals",
]
