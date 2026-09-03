# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import typing as tp
from dataclasses import dataclass
from typing import Dict

import pydantic
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ...distributed import broadcast_module_state
from ..base import BaseConfig
from ..behavior_context import (
    HEADING_SOURCE_EXACT_TRACKING,
    HEADING_SOURCE_INVALID,
    heading_observation,
    relative_heading_target,
)
from ..fb_cpr.agent import FBcprAgent, FBcprAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode, weight_init
from ..selective_prior import (
    PriorLabel,
    PriorPhase,
    SelectivePriorState,
    active_finalized_mask,
    actor_prior_interior_mask,
    approximate_pairwise_auc,
    qd_interior_mask,
)
from .model import FBcprAuxModelConfig


def prior_transition_discount(
    prior_batch: dict[str, tp.Any],
    *,
    gamma: float,
    device: str | torch.device,
) -> torch.Tensor:
    """Return the behavior-prior discount aligned with the sampled action.

    Both main and canonical-plane collectors store the outcome of the action
    in the current replay slot.  This is required because MJLab auto-reset
    returns reset observations and because ``truncated`` includes artificial
    terrain-boundary resets.  The legacy fallback is retained for old replay
    buffers that predate action-outcome metadata.
    """

    if "transition_terminated" in prior_batch and "transition_truncated" in prior_batch:
        done = prior_batch["transition_terminated"].to(device) | prior_batch["transition_truncated"].to(device)
    else:
        done = prior_batch["next"]["terminated"].to(device)
        if "truncated" in prior_batch["next"]:
            done = done | prior_batch["next"]["truncated"].to(device)
    return float(gamma) * ~done


class FBcprAuxAgentTrainConfig(FBcprAgentTrainConfig):
    lr_aux_critic: float = 1e-4
    reg_coeff_aux: float = 1.0
    aux_critic_pessimism_penalty: float = 0.5
    lr_heading_critic: float = 1e-4
    reg_coeff_heading: float = 0.0
    heading_critic_pessimism_penalty: float = 0.5
    # Expert-anchored selective online prior expansion.  This is opt-in so
    # legacy/no-D/canonical-plane runs retain byte-for-byte update topology.
    selective_prior_enabled: bool = False
    selective_prior_validated_weight: float = 0.5
    selective_prior_expert_fraction: float = 0.50
    selective_prior_validated_fraction: float = 0.33
    selective_prior_bad_fraction: float = 0.17
    selective_prior_gate_window: int = 8
    selective_prior_gate_future: int = 8
    selective_prior_gate_windows_per_refresh: int = 64
    selective_prior_gate_teacher_refresh_updates: int = 4096
    selective_prior_expansion_refresh_updates: int = 1024
    selective_prior_label_ttl_steps: int = 20_000_000
    selective_prior_good_cosine_mean: float = 0.75
    selective_prior_good_cosine_min: float = 0.35
    selective_prior_bad_cosine_mean: float = 0.0
    selective_prior_bad_sustain_fraction: float = 0.5
    selective_prior_good_heading_cost_mean_max: float = 0.30
    selective_prior_bad_heading_cost_mean_min: float = 1.0
    selective_prior_actor_interior_horizon: int = 4
    selective_prior_min_validated_per_rank: int = 4096
    selective_prior_min_bad_per_rank: int = 2048
    selective_prior_min_balanced_motion_strata_per_rank: int = 8
    selective_prior_balance_ratio_min: float = 0.5
    selective_prior_balance_ratio_max: float = 2.0
    selective_prior_d_min_updates: int = 512
    selective_prior_d_ready_streak: int = 64
    selective_prior_d_positive_min: float = 0.5
    selective_prior_d_bad_max: float = 0.0
    selective_prior_d_expert_validated_gap_max: float = 0.35
    selective_prior_d_expert_validated_auc_max: float = 0.65
    selective_prior_d_validated_bad_auc_min: float = 0.80
    selective_prior_qd_min_updates: int = 512
    selective_prior_qd_ready_streak: int = 64
    selective_prior_qd_relative_uncertainty_max: float = 0.20


@dataclass
class RelabeledBehaviorContext:
    """A z and its matching deployable heading observation pair."""

    z: torch.Tensor
    heading: torch.Tensor
    next_heading: torch.Tensor
    heading_valid: torch.Tensor
    source_type: torch.Tensor


def _distribution_metrics(prefix: str, values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Summarize a diagnostic distribution without synchronizing to the CPU."""

    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        zero = torch.zeros((), device=values.device, dtype=torch.float32)
        return {
            f"{prefix}/mean": zero,
            f"{prefix}/p10": zero,
            f"{prefix}/p50": zero,
            f"{prefix}/p90": zero,
        }
    quantiles = torch.quantile(
        flat,
        torch.tensor((0.1, 0.5, 0.9), device=flat.device, dtype=flat.dtype),
    )
    return {
        f"{prefix}/mean": flat.mean(),
        f"{prefix}/p10": quantiles[0],
        f"{prefix}/p50": quantiles[1],
        f"{prefix}/p90": quantiles[2],
    }


class FBcprAuxAgentConfig(BaseConfig):
    name: tp.Literal["FBcprAuxAgent"] = "FBcprAuxAgent"

    model: FBcprAuxModelConfig = FBcprAuxModelConfig()
    train: FBcprAuxAgentTrainConfig = FBcprAuxAgentTrainConfig()
    aux_rewards: list[str] = pydantic.Field(default_factory=list)
    aux_rewards_scaling: dict[str, float] = pydantic.Field(default_factory=dict)
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim: int) -> "FBcprAuxAgent":
        return self.object_class(
            obs_space=obs_space,
            action_dim=action_dim,
            cfg=self,
        )

    @property
    def object_class(self):
        return FBcprAuxAgent


class FBcprAuxAgent(FBcprAgent):
    config_class = FBcprAuxAgentConfig

    @property
    def _behavior_prior_enabled(self) -> bool:
        return bool(getattr(self.cfg.train, "behavior_prior_enabled", True))

    @property
    def _heading_context_enabled(self) -> bool:
        return bool(getattr(getattr(self.cfg, "model", None), "heading_context_enabled", False))

    @property
    def _heading_critic_enabled(self) -> bool:
        return bool(getattr(getattr(self.cfg, "model", None), "heading_critic_enabled", False))

    @property
    def _selective_prior_enabled(self) -> bool:
        return bool(getattr(self.cfg.train, "selective_prior_enabled", False))

    def setup_training(self) -> None:
        super().setup_training()

        self._selective_prior_state = SelectivePriorState()
        self._last_selective_gate_step: int | None = None
        self._last_selective_mask_step: int | None = None
        self._cached_selective_masks: dict[str, torch.Tensor] | None = None
        self._last_selective_teacher_refresh_update = 0
        self._gate_backward_teacher = None
        self._prior_reward_discriminator = None
        if self._selective_prior_enabled:
            if not self._behavior_prior_enabled:
                raise ValueError("selective_prior_enabled requires behavior_prior_enabled")
            if self.cfg.train.discriminator_loss != "lsgan":
                raise ValueError("Selective prior expansion requires LSGAN discriminator targets")
            if self.cfg.train.discriminator_reward != "amp":
                raise ValueError("Selective prior expansion requires bounded AMP discriminator reward")
            fractions = (
                self.cfg.train.selective_prior_expert_fraction,
                self.cfg.train.selective_prior_validated_fraction,
                self.cfg.train.selective_prior_bad_fraction,
            )
            if any(value <= 0.0 for value in fractions) or abs(sum(fractions) - 1.0) > 1.0e-6:
                raise ValueError(f"Selective prior E/V/B fractions must be positive and sum to one, got {fractions}")
            if self.cfg.train.selective_prior_validated_weight <= 0.0:
                raise ValueError("selective_prior_validated_weight must be positive")
            if self.cfg.train.selective_prior_actor_interior_horizon < 2:
                raise ValueError("Actor-D requires an interior horizon of at least two states")
            # B is small relative to F/Actor.  A frozen snapshot gives the gate
            # a genuinely stationary, D-independent admission rule without a
            # second policy or an additional optimizer.
            self._gate_backward_teacher = copy.deepcopy(self._model._target_backward_map)
            self._gate_backward_teacher.eval().requires_grad_(False)
            self._prior_reward_discriminator = copy.deepcopy(self._model._discriminator)
            self._prior_reward_discriminator.eval().requires_grad_(False)

        # prepare parameter list
        self._aux_critic_map_paramlist = tuple(x for x in self._model._aux_critic.parameters())
        self._aux_target_critic_map_paramlist = tuple(x for x in self._model._target_aux_critic.parameters())

        self.aux_critic_optimizer = torch.optim.Adam(
            self._model._aux_critic.parameters(),
            lr=self.cfg.train.lr_aux_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        if self._heading_critic_enabled:
            self._heading_critic_map_paramlist = tuple(x for x in self._model._heading_critic.parameters())
            self._heading_target_critic_map_paramlist = tuple(x for x in self._model._target_heading_critic.parameters())
            self.heading_critic_optimizer = torch.optim.Adam(
                self._model._heading_critic.parameters(),
                lr=self.cfg.train.lr_heading_critic,
                capturable=self.cfg.cudagraphs and not self.cfg.compile,
                weight_decay=self.cfg.train.weight_decay,
            )

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers["aux_critic_optimizer"] = self.aux_critic_optimizer.state_dict()
        if self._heading_critic_enabled:
            optimizers["heading_critic_optimizer"] = self.heading_critic_optimizer.state_dict()
        return optimizers

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_aux_critic = torch.compile(self.update_aux_critic, mode=mode)
            if self._heading_critic_enabled:
                self.update_heading_critic = torch.compile(self.update_heading_critic, mode=mode)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_aux_critic = CudaGraphModule(self.update_aux_critic, warmup=5)
            if self._heading_critic_enabled:
                self.update_heading_critic = CudaGraphModule(self.update_heading_critic, warmup=5)

    def extra_training_state_dict(self) -> dict[str, tp.Any]:
        state: dict[str, tp.Any] = {
            "selective_prior_state": self._selective_prior_state.state_dict(),
            "last_selective_teacher_refresh_update": int(self._last_selective_teacher_refresh_update),
        }
        if self._gate_backward_teacher is not None:
            state["gate_backward_teacher"] = self._gate_backward_teacher.state_dict()
        if self._prior_reward_discriminator is not None:
            state["prior_reward_discriminator"] = self._prior_reward_discriminator.state_dict()
        return state

    def load_extra_training_state_dict(self, state: dict[str, tp.Any]) -> None:
        if not self._selective_prior_enabled:
            return
        prior_state = state.get("selective_prior_state")
        if not isinstance(prior_state, dict):
            self.reset_selective_prior_to_bootstrap(reason="missing_checkpoint_state")
            return
        self._selective_prior_state = SelectivePriorState.from_state_dict(prior_state)
        self._last_selective_teacher_refresh_update = int(state.get("last_selective_teacher_refresh_update", 0))
        teacher_state = state.get("gate_backward_teacher")
        if teacher_state is None or self._gate_backward_teacher is None:
            self.reset_selective_prior_to_bootstrap(reason="missing_gate_teacher")
            return
        self._gate_backward_teacher.load_state_dict(teacher_state)
        self._gate_backward_teacher.eval().requires_grad_(False)
        reward_discriminator_state = state.get("prior_reward_discriminator")
        if reward_discriminator_state is None or self._prior_reward_discriminator is None:
            self.reset_selective_prior_to_bootstrap(reason="missing_reward_discriminator_snapshot")
            return
        self._prior_reward_discriminator.load_state_dict(reward_discriminator_state)
        self._prior_reward_discriminator.eval().requires_grad_(False)

    def reset_selective_prior_to_bootstrap(self, *, reason: str) -> None:
        if not self._selective_prior_enabled:
            return
        previous = getattr(self, "_selective_prior_state", SelectivePriorState())
        self._selective_prior_state = SelectivePriorState(
            bank_version=int(previous.bank_version) + 1,
            gate_teacher_version=int(previous.gate_teacher_version),
            discriminator_version=int(previous.discriminator_version) + 1,
        )
        self._last_selective_gate_step = None
        self._last_selective_mask_step = None
        self._cached_selective_masks = None
        print(
            f"[SELECTIVE PRIOR] fail-closed reset to BOOTSTRAP: {reason}",
            flush=True,
        )

    def _reset_selective_qd(self) -> None:
        """Invalidate Q_D whenever its frozen reward model changes."""

        self._model._critic.apply(weight_init)
        broadcast_module_state(self._model._critic, src=0)
        self._model._target_critic.load_state_dict(self._model._critic.state_dict())
        self.critic_optimizer.state.clear()

    def _snapshot_selective_reward_model(self) -> None:
        if self._prior_reward_discriminator is None:
            raise RuntimeError("Selective prior reward discriminator is not initialized")
        self._prior_reward_discriminator.load_state_dict(self._model._discriminator.state_dict())
        self._prior_reward_discriminator.eval().requires_grad_(False)
        self._selective_prior_state.discriminator_version += 1
        self._selective_prior_state.qd_reward_version = self._selective_prior_state.discriminator_version
        self._selective_prior_state.discriminator_bank_version = self._selective_prior_state.bank_version
        self._selective_prior_state.qd_bank_version = self._selective_prior_state.bank_version
        self._reset_selective_qd()

    @torch.no_grad()
    def _discriminator_conditioning_diagnostics(
        self,
        *,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        prior_obs: torch.Tensor | dict[str, torch.Tensor],
        prior_rollout_z: torch.Tensor,
        prior_relabel_z: torch.Tensor,
        max_samples: int = 256,
    ) -> dict[str, torch.Tensor]:
        """Compare D logits before and after prior-context relabeling.

        D is trained with ``prior_rollout_z`` while Q_D consumes
        ``prior_relabel_z``.  Keeping both evaluations on the same states and
        the same post-update discriminator separates saturation from a
        conditioning-distribution mismatch.  A small subset keeps this
        training-only diagnostic cheap.
        """

        batch_size = min(
            int(expert_z.shape[0]),
            int(prior_rollout_z.shape[0]),
            int(prior_relabel_z.shape[0]),
            int(max_samples),
        )
        expert_obs = tree_map(lambda x: x[:batch_size], expert_obs)
        prior_obs = tree_map(lambda x: x[:batch_size], prior_obs)
        expert_z = expert_z[:batch_size]
        prior_rollout_z = prior_rollout_z[:batch_size]
        prior_relabel_z = prior_relabel_z[:batch_size]

        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            expert_logits = self._model._discriminator.compute_logits(expert_obs, expert_z)
            rollout_logits = self._model._discriminator.compute_logits(prior_obs, prior_rollout_z)
            relabel_logits = self._model._discriminator.compute_logits(prior_obs, prior_relabel_z)
            rollout_rewards = self.discriminator_reward_from_logits(rollout_logits)
            relabel_rewards = self.discriminator_reward_from_logits(relabel_logits)

        metrics: dict[str, torch.Tensor] = {}
        metrics.update(_distribution_metrics("disc_diag/expert_logit", expert_logits))
        metrics.update(_distribution_metrics("disc_diag/policy_rollout_logit", rollout_logits))
        metrics.update(_distribution_metrics("disc_diag/policy_relabel_logit", relabel_logits))
        metrics.update(_distribution_metrics("disc_diag/reward_original_z", rollout_rewards))
        metrics.update(_distribution_metrics("disc_diag/reward_relabel_z", relabel_rewards))
        if getattr(getattr(self.cfg, "train", None), "discriminator_reward", "log_odds") == "amp":
            metrics["disc_diag/zero_reward_fraction_original"] = (rollout_rewards <= 0.0).float().mean()
            metrics["disc_diag/zero_reward_fraction_relabel"] = (relabel_rewards <= 0.0).float().mean()
        metrics["disc_diag/reward_original_relabel_abs_diff"] = (relabel_rewards - rollout_rewards).abs().mean()

        relabeled = (prior_relabel_z.float() - prior_rollout_z.float()).abs().amax(dim=-1) > 1.0e-6
        logit_shift = (relabel_logits - rollout_logits).reshape(-1)
        metrics["disc_diag/relabel_fraction"] = relabeled.float().mean()
        metrics.update(
            _distribution_metrics(
                "disc_diag/relabel_logit_shift",
                logit_shift[relabeled],
            )
        )
        return metrics

    def _relabel_main_and_prior_z(
        self,
        *,
        main_next_obs: torch.Tensor | dict[str, torch.Tensor],
        prior_next_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        main_rollout_z: torch.Tensor,
        prior_rollout_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy z-only path for checkpoints without BehaviorContext inputs."""

        sampled_main_z = self.sample_mixed_z(train_goal=main_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(sampled_main_z)
        sampled_prior_z = self.sample_mixed_z(train_goal=prior_next_obs, expert_encodings=expert_z).clone()
        if self.cfg.train.relabel_ratio is None:
            return main_rollout_z, prior_rollout_z
        main_mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
        prior_mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
        return (
            torch.where(main_mask, sampled_main_z, main_rollout_z),
            torch.where(prior_mask, sampled_prior_z, prior_rollout_z),
        )

    def _sample_mixed_behavior_context(
        self,
        *,
        train_goal: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        root_heading_xy: torch.Tensor,
        next_root_heading_xy: torch.Tensor,
        expert_heading_xy: torch.Tensor,
        expert_next_heading_xy: torch.Tensor,
    ) -> RelabeledBehaviorContext:
        """Sample z together with a source-valid one-step heading context.

        Goal and random latents deliberately receive an invalid zero heading.
        Only an expert z sampled with exact expert transition metadata carries
        a relative-heading command.
        """

        sampled_z, mix_source, expert_perm = self.sample_mixed_context_components(
            train_goal,
            expert_z,
        )
        expert_selected = mix_source == 1
        target = root_heading_xy
        target_next = relative_heading_target(
            root_heading_xy,
            expert_heading_xy[expert_perm],
            expert_next_heading_xy[expert_perm],
        )
        heading = heading_observation(root_heading_xy, target, expert_selected)
        next_heading = heading_observation(next_root_heading_xy, target_next, expert_selected)
        source_type = torch.where(
            expert_selected,
            torch.full_like(mix_source, HEADING_SOURCE_EXACT_TRACKING),
            torch.full_like(mix_source, HEADING_SOURCE_INVALID),
        )
        return RelabeledBehaviorContext(
            z=sampled_z.clone(),
            heading=heading,
            next_heading=next_heading,
            heading_valid=expert_selected,
            source_type=source_type,
        )

    def _relabel_main_z(
        self,
        *,
        main_next_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        main_rollout_z: torch.Tensor,
    ) -> torch.Tensor:
        """Relabel only the main stream for the formal no-D configuration."""

        sampled_main_z = self.sample_mixed_z(
            train_goal=main_next_obs,
            expert_encodings=expert_z,
        ).clone()
        self.z_buffer.add(sampled_main_z)
        if self.cfg.train.relabel_ratio is None:
            return main_rollout_z
        main_mask = (
            torch.rand(
                (main_rollout_z.shape[0], 1),
                device=self.device,
            )
            <= self.cfg.train.relabel_ratio
        )
        return torch.where(main_mask, sampled_main_z, main_rollout_z)

    def _relabel_main_context(
        self,
        *,
        main_obs: dict[str, torch.Tensor],
        main_next_obs: dict[str, torch.Tensor],
        main_batch: dict[str, tp.Any],
        expert_z: torch.Tensor,
        expert_heading_xy: torch.Tensor,
        expert_next_heading_xy: torch.Tensor,
        main_rollout_z: torch.Tensor,
    ) -> tuple[RelabeledBehaviorContext, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Atomically relabel main z/heading without sampling a dummy prior stream."""

        sampled = self._sample_mixed_behavior_context(
            train_goal=main_next_obs,
            expert_z=expert_z,
            root_heading_xy=main_batch["root_heading_xy"].to(self.device),
            next_root_heading_xy=main_batch["next_root_heading_xy"].to(self.device),
            expert_heading_xy=expert_heading_xy,
            expert_next_heading_xy=expert_next_heading_xy,
        )
        self.z_buffer.add(sampled.z)
        original = RelabeledBehaviorContext(
            z=main_rollout_z,
            heading=main_obs["heading"],
            next_heading=main_next_obs["heading"],
            heading_valid=main_batch["heading_valid"].to(self.device),
            source_type=main_batch["heading_source_type"].to(self.device),
        )
        if self.cfg.train.relabel_ratio is None:
            return original, main_obs, main_next_obs

        mask = (
            torch.rand(
                (main_rollout_z.shape[0], 1),
                device=self.device,
            )
            <= self.cfg.train.relabel_ratio
        )
        merged = RelabeledBehaviorContext(
            z=torch.where(mask, sampled.z, original.z),
            heading=torch.where(mask, sampled.heading, original.heading),
            next_heading=torch.where(mask, sampled.next_heading, original.next_heading),
            heading_valid=torch.where(mask, sampled.heading_valid, original.heading_valid),
            source_type=torch.where(mask, sampled.source_type, original.source_type),
        )
        relabeled_obs = dict(main_obs)
        relabeled_next_obs = dict(main_next_obs)
        relabeled_obs["heading"] = merged.heading
        relabeled_next_obs["heading"] = merged.next_heading
        return merged, relabeled_obs, relabeled_next_obs

    def _relabel_main_and_prior_context(
        self,
        *,
        main_obs: dict[str, torch.Tensor],
        main_next_obs: dict[str, torch.Tensor],
        prior_obs: dict[str, torch.Tensor],
        prior_next_obs: dict[str, torch.Tensor],
        main_batch: dict[str, tp.Any],
        prior_batch: dict[str, tp.Any],
        expert_z: torch.Tensor,
        expert_heading_xy: torch.Tensor,
        expert_next_heading_xy: torch.Tensor,
        main_rollout_z: torch.Tensor,
        prior_rollout_z: torch.Tensor,
    ) -> tuple[RelabeledBehaviorContext, RelabeledBehaviorContext, dict, dict, dict, dict]:
        """Relabel both streams atomically: z and heading always change together."""

        sampled_main = self._sample_mixed_behavior_context(
            train_goal=main_next_obs,
            expert_z=expert_z,
            root_heading_xy=main_batch["root_heading_xy"].to(self.device),
            next_root_heading_xy=main_batch["next_root_heading_xy"].to(self.device),
            expert_heading_xy=expert_heading_xy,
            expert_next_heading_xy=expert_next_heading_xy,
        )
        self.z_buffer.add(sampled_main.z)
        sampled_prior = self._sample_mixed_behavior_context(
            train_goal=prior_next_obs,
            expert_z=expert_z,
            root_heading_xy=prior_batch["root_heading_xy"].to(self.device),
            next_root_heading_xy=prior_batch["next_root_heading_xy"].to(self.device),
            expert_heading_xy=expert_heading_xy,
            expert_next_heading_xy=expert_next_heading_xy,
        )

        def original_context(batch, rollout_z, obs, next_obs):
            return RelabeledBehaviorContext(
                z=rollout_z,
                heading=obs["heading"],
                next_heading=next_obs["heading"],
                heading_valid=batch["heading_valid"].to(self.device),
                source_type=batch["heading_source_type"].to(self.device),
            )

        main_original = original_context(main_batch, main_rollout_z, main_obs, main_next_obs)
        prior_original = original_context(prior_batch, prior_rollout_z, prior_obs, prior_next_obs)

        if self.cfg.train.relabel_ratio is None:
            return main_original, prior_original, main_obs, main_next_obs, prior_obs, prior_next_obs
        main_mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
        prior_mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio

        def merge(original, sampled, mask, obs, next_obs):
            merged = RelabeledBehaviorContext(
                z=torch.where(mask, sampled.z, original.z),
                heading=torch.where(mask, sampled.heading, original.heading),
                next_heading=torch.where(mask, sampled.next_heading, original.next_heading),
                heading_valid=torch.where(mask, sampled.heading_valid, original.heading_valid),
                source_type=torch.where(mask, sampled.source_type, original.source_type),
            )
            relabeled_obs = dict(obs)
            relabeled_next_obs = dict(next_obs)
            relabeled_obs["heading"] = merged.heading
            relabeled_next_obs["heading"] = merged.next_heading
            return merged, relabeled_obs, relabeled_next_obs

        main_context, main_obs, main_next_obs = merge(main_original, sampled_main, main_mask, main_obs, main_next_obs)
        prior_context, prior_obs, prior_next_obs = merge(prior_original, sampled_prior, prior_mask, prior_obs, prior_next_obs)
        return (
            main_context,
            prior_context,
            main_obs,
            main_next_obs,
            prior_obs,
            prior_next_obs,
        )

    @torch.compiler.disable
    def _global_min_int(self, value: int) -> int:
        tensor = torch.tensor(int(value), device=self.device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return int(tensor.item())

    @torch.compiler.disable
    def _global_all(self, value: bool) -> bool:
        return self._global_min_int(int(bool(value))) == 1

    @torch.compiler.disable
    def _global_any(self, value: bool) -> bool:
        tensor = torch.tensor(int(bool(value)), device=self.device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(tensor.item())

    def _selective_prior_active_masks(self, replay, step: int) -> dict[str, torch.Tensor]:
        storage = replay.storage
        required = {
            "prior_label",
            "prior_label_step",
            "heading_context_id",
            "prior_motion_id",
            "transition_terminated",
            "transition_truncated",
        }
        missing = sorted(required.difference(storage))
        if missing:
            raise RuntimeError(f"Selective-prior replay metadata is incomplete: {missing}")
        active = active_finalized_mask(
            storage["prior_label"].squeeze(-1),
            storage["prior_label_step"].squeeze(-1),
            step=step,
            ttl_steps=self.cfg.train.selective_prior_label_ttl_steps,
        )
        valid_slots = replay._valid_slot_mask()
        active &= valid_slots
        labels = storage["prior_label"].squeeze(-1).to(torch.long)
        validated = active & (labels == int(PriorLabel.VALIDATED))
        bad = active & (labels == int(PriorLabel.BAD))
        motion_ids = storage["prior_motion_id"].squeeze(-1).to(torch.long)
        validated_ids = torch.unique(motion_ids[validated & (motion_ids >= 0)])
        bad_ids = torch.unique(motion_ids[bad & (motion_ids >= 0)])
        common_ids = validated_ids[torch.isin(validated_ids, bad_ids)]
        balanced_validated = validated & torch.isin(motion_ids, common_ids)
        balanced_bad = bad & torch.isin(motion_ids, common_ids)
        done = storage["transition_terminated"].squeeze(-1).to(torch.bool)
        done |= storage["transition_truncated"].squeeze(-1).to(torch.bool)
        successor_available = replay.successor_available_mask()
        contexts = storage["heading_context_id"].squeeze(-1)
        qd = qd_interior_mask(
            active=active,
            context_id=contexts,
            transition_done=done,
            successor_available=successor_available,
        )
        actor = actor_prior_interior_mask(
            active=active,
            context_id=contexts,
            transition_done=done,
            successor_available=successor_available,
            horizon=self.cfg.train.selective_prior_actor_interior_horizon,
        )
        return {
            "active": active,
            "validated": validated,
            "bad": bad,
            "balanced_validated": balanced_validated,
            "balanced_bad": balanced_bad,
            "balanced_strata_count": torch.tensor(int(common_ids.numel()), device=motion_ids.device, dtype=torch.long),
            "qd": qd,
            "actor": actor,
        }

    @staticmethod
    def _window_all_same(values: torch.Tensor) -> torch.Tensor:
        return values.eq(values[:, :1]).all(dim=1)

    def _selective_pathology_mask(self, batch: dict[str, tp.Any]) -> torch.Tensor:
        """Conservative, local high-confidence pathology detector.

        Missing reward components never become implicit BAD labels.  Thresholds
        are deliberately severe: ordinary task failure remains UNKNOWN.
        """

        shape = batch["z"].shape[:-1]
        severe = torch.zeros(shape, device=self.device, dtype=torch.bool)
        rewards = batch.get("aux_rewards", {})
        thresholds = {
            "limits_dof_pos": 0.35,
            "penalty_body_impact": 20.0,
            "penalty_slippage": 3.0,
            "penalty_ankle_roll": 1.0,
            "penalty_action_rate": 150.0,
            "feet_stumble": 0.75,
            "feet_at_plane": 0.75,
        }
        for name, threshold in thresholds.items():
            value = rewards.get(name)
            if value is not None:
                severe |= value.to(self.device).reshape(shape).float() >= threshold
        action = batch.get("action")
        if action is not None:
            severe |= ~torch.isfinite(action.to(self.device)).all(dim=-1)
        return severe

    @torch.no_grad()
    @torch.compiler.disable
    def _refresh_selective_prior_labels(self, replay, step: int) -> dict[str, torch.Tensor]:
        if self._last_selective_gate_step == int(step):
            return {}
        self._last_selective_gate_step = int(step)
        state = self._selective_prior_state
        state.update_count += 1

        refresh_every = self.cfg.train.selective_prior_gate_teacher_refresh_updates
        if self._gate_backward_teacher is not None and state.update_count - self._last_selective_teacher_refresh_update >= refresh_every:
            self._gate_backward_teacher.load_state_dict(self._model._target_backward_map.state_dict())
            self._gate_backward_teacher.eval().requires_grad_(False)
            state.gate_teacher_version += 1
            self._last_selective_teacher_refresh_update = state.update_count

        if state.phase_enum in (PriorPhase.FIT_D, PriorPhase.FIT_QD):
            return {"prior/gate_paused_for_calibration": torch.ones((), device=self.device)}
        if (
            state.phase_enum == PriorPhase.ACTOR_PRIOR
            and state.update_count % self.cfg.train.selective_prior_expansion_refresh_updates != 0
        ):
            return {"prior/gate_paused_for_calibration": torch.zeros((), device=self.device)}

        window = self.cfg.train.selective_prior_gate_window
        future = self.cfg.train.selective_prior_gate_future
        total = window + future
        windows = self.cfg.train.selective_prior_gate_windows_per_refresh
        batch = idxs = None
        try:
            batch, idxs = replay.sample(
                batch_size=windows * total,
                seq_length=total,
                return_indices=True,
                restore_depth=False,
            )
        except ValueError:
            pass
        if not self._global_all(batch is not None):
            return {"prior/gate_ready": torch.zeros((), device=self.device)}
        if batch is None or idxs is None:  # narrowed by the synchronized gate
            raise RuntimeError("Selective gate availability synchronization failed")

        observation = tree_map(lambda value: value.to(self.device), batch["observation"])
        with eval_mode(self._model._obs_normalizer):
            observation = self._model._obs_normalizer(observation)
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            encoded = self._gate_backward_teacher(observation)
            encoded = self._model.project_z(encoded)
        rollout_z = batch["z"].to(self.device)
        cosine = F.cosine_similarity(encoded.float(), rollout_z.float(), dim=-1).reshape(windows, total)
        source = batch["heading_source_type"].to(self.device).reshape(windows, total, -1)[..., 0]
        contexts = batch["heading_context_id"].to(self.device).reshape(windows, total, -1)[..., 0]
        exact_tracking = source.eq(HEADING_SOURCE_EXACT_TRACKING).all(dim=1)
        same_context = self._window_all_same(contexts)
        finite = torch.isfinite(cosine).all(dim=1)
        pathology = self._selective_pathology_mask(batch).reshape(windows, total)
        heading_cost = batch["observation"]["heading"].to(self.device)[..., 0].reshape(windows, total)

        semantic_mean = cosine.mean(dim=1)
        semantic_min = cosine.amin(dim=1)
        good_window = (
            exact_tracking
            & same_context
            & finite
            & (semantic_mean >= self.cfg.train.selective_prior_good_cosine_mean)
            & (semantic_min >= self.cfg.train.selective_prior_good_cosine_min)
            & (heading_cost.mean(dim=1) <= self.cfg.train.selective_prior_good_heading_cost_mean_max)
            & ~pathology.any(dim=1)
        )
        bad_semantic = (cosine <= self.cfg.train.selective_prior_bad_cosine_mean).float().mean(
            dim=1
        ) >= self.cfg.train.selective_prior_bad_sustain_fraction
        bad_heading = heading_cost.mean(dim=1) >= self.cfg.train.selective_prior_bad_heading_cost_mean_min
        bad_window = exact_tracking & same_context & finite & (bad_semantic | bad_heading | (pathology.float().mean(dim=1) >= 0.25))
        # A contradictory window is never admitted as positive.
        good_window &= ~bad_window

        current_count = windows * window
        label = torch.full(
            (windows, window, 1),
            int(PriorLabel.UNKNOWN),
            dtype=torch.int8,
            device=self.device,
        )
        label[good_window] = int(PriorLabel.VALIDATED)
        label[bad_window] = int(PriorLabel.BAD)
        confidence = semantic_mean.abs().clamp(0.0, 1.0)[:, None, None].expand(-1, window, -1)

        time_idx = idxs[0].reshape(windows, total)[:, :window].reshape(-1)
        env_idx = idxs[1].reshape(windows, total)[:, :window].reshape(-1)
        generation = batch["prior_generation"].reshape(windows, total, -1)[:, :window].reshape(-1, 1)
        label = label.reshape(current_count, 1)
        confidence = confidence.reshape(current_count, 1)
        # Windows are sampled with replacement. Consolidate duplicate slot
        # proposals deterministically (BAD > VALIDATED > UNKNOWN), then write
        # UNKNOWN too so failed revalidation immediately revokes old labels.
        num_envs = replay.storage["prior_label"].shape[1]
        linear = time_idx * num_envs + env_idx
        unique_linear, inverse = torch.unique(linear, sorted=False, return_inverse=True)
        aggregated_label = torch.zeros(unique_linear.shape[0], device=self.device, dtype=torch.int8)
        aggregated_label.scatter_reduce_(0, inverse, label.reshape(-1), reduce="amax", include_self=True)
        aggregated_confidence = torch.zeros(unique_linear.shape[0], device=self.device, dtype=torch.float32)
        aggregated_confidence.scatter_reduce_(0, inverse, confidence.reshape(-1), reduce="amax", include_self=True)
        aggregated_generation = torch.zeros(unique_linear.shape[0], device=self.device, dtype=torch.long)
        aggregated_generation.scatter_reduce_(
            0,
            inverse,
            generation.reshape(-1).to(torch.long),
            reduce="amax",
            include_self=True,
        )
        selected_idxs = (
            torch.div(unique_linear, num_envs, rounding_mode="floor"),
            torch.remainder(unique_linear, num_envs),
        )
        old_label = replay.storage["prior_label"][selected_idxs].reshape(-1).clone()
        active_label = aggregated_label != int(PriorLabel.UNKNOWN)
        written = replay.set_fields_at_indices(
            selected_idxs,
            {
                "prior_label": aggregated_label.unsqueeze(-1),
                "prior_label_step": torch.where(
                    active_label,
                    torch.full_like(aggregated_generation, int(step)),
                    torch.zeros_like(aggregated_generation),
                ).unsqueeze(-1),
                "prior_teacher_version": torch.full(
                    (unique_linear.shape[0], 1),
                    state.gate_teacher_version,
                    device=self.device,
                    dtype=torch.long,
                ),
                "prior_confidence": torch.where(
                    active_label,
                    aggregated_confidence,
                    torch.zeros_like(aggregated_confidence),
                ).unsqueeze(-1),
            },
            expected_generation=aggregated_generation.unsqueeze(-1),
        )
        revoked = ((old_label != int(PriorLabel.UNKNOWN)) & (aggregated_label == int(PriorLabel.UNKNOWN))).sum()
        promoted = ((old_label == int(PriorLabel.UNKNOWN)) & (aggregated_label != int(PriorLabel.UNKNOWN))).sum()
        local_bank_changed = written and bool((old_label != aggregated_label).any())
        bank_changed = self._global_any(local_bank_changed)
        if bank_changed:
            state.bank_version += 1
            if state.phase_enum == PriorPhase.ACTOR_PRIOR:
                # Slow support expansion is reversible, but the old Q_D reward
                # snapshot is no longer certified on this bank. Reopen D then
                # Q_D calibration before Actor-D sees the expanded support.
                state.set_phase(PriorPhase.FIT_D)
                state.qd_reward_version = -1
                state.qd_bank_version = -1
                state.d_ready_streak = 0
                state.qd_ready_streak = 0

        return {
            "prior/gate_ready": torch.ones((), device=self.device),
            "prior/gate_good_window_fraction": good_window.float().mean(),
            "prior/gate_bad_window_fraction": bad_window.float().mean(),
            "prior/gate_unknown_window_fraction": (~good_window & ~bad_window).float().mean(),
            "prior/gate_semantic_cosine_mean": semantic_mean.mean(),
            "prior/gate_heading_cost_mean": heading_cost.mean(),
            "prior/gate_written_frames": torch.tensor(float(written), device=self.device),
            "prior/gate_promoted_frames": promoted.float(),
            "prior/gate_revoked_frames": revoked.float(),
            "prior/gate_teacher_version": torch.tensor(float(state.gate_teacher_version), device=self.device),
        }

    @torch.compiler.disable
    def _selective_prior_phase_from_coverage(
        self,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        state = self._selective_prior_state
        local_v = int(masks["balanced_validated"].sum().item())
        local_b = int(masks["balanced_bad"].sum().item())
        local_strata = int(masks["balanced_strata_count"].item())
        local_qd = int(masks["qd"].sum().item())
        local_actor = int(masks["actor"].sum().item())
        min_v = self._global_min_int(local_v)
        min_b = self._global_min_int(local_b)
        min_strata = self._global_min_int(local_strata)
        min_qd = self._global_min_int(local_qd)
        min_actor = self._global_min_int(local_actor)
        effective_v = min_v * self.cfg.train.selective_prior_validated_weight
        ratio = effective_v / max(float(min_b), 1.0)
        ready = (
            min_v >= self.cfg.train.selective_prior_min_validated_per_rank
            and min_b >= self.cfg.train.selective_prior_min_bad_per_rank
            and min_strata >= self.cfg.train.selective_prior_min_balanced_motion_strata_per_rank
            and min_qd > 0
            and min_actor > 0
            and self.cfg.train.selective_prior_balance_ratio_min <= ratio <= self.cfg.train.selective_prior_balance_ratio_max
        )
        if state.phase_enum > PriorPhase.BOOTSTRAP and not ready:
            # Expiry/revocation can invalidate the bank after Actor-D was
            # enabled.  Reopen the loop only after D then Q_D are recalibrated.
            self.reset_selective_prior_to_bootstrap(reason="coverage_or_balance_health_degraded")
            state = self._selective_prior_state
        if state.phase_enum == PriorPhase.BOOTSTRAP and ready:
            state.set_phase(PriorPhase.FIT_D)
            state.d_ready_streak = 0
        return {
            "prior/local_validated_count": torch.tensor(float(local_v), device=self.device),
            "prior/local_bad_count": torch.tensor(float(local_b), device=self.device),
            "prior/local_balanced_motion_strata": torch.tensor(float(local_strata), device=self.device),
            "prior/local_qd_interior_count": torch.tensor(float(local_qd), device=self.device),
            "prior/local_actor_interior_count": torch.tensor(float(local_actor), device=self.device),
            "prior/global_min_validated_count": torch.tensor(float(min_v), device=self.device),
            "prior/global_min_bad_count": torch.tensor(float(min_b), device=self.device),
            "prior/global_min_balanced_motion_strata": torch.tensor(float(min_strata), device=self.device),
            "prior/global_min_qd_interior_count": torch.tensor(float(min_qd), device=self.device),
            "prior/global_min_actor_interior_count": torch.tensor(float(min_actor), device=self.device),
            "prior/effective_validated_bad_ratio": torch.tensor(float(ratio), device=self.device),
            "prior/coverage_ready": torch.tensor(float(ready), device=self.device),
            "prior/phase": torch.tensor(float(state.phase), device=self.device),
        }

    def _selective_masks_and_coverage(
        self,
        replay,
        step: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute large replay masks and DDP coverage once per collector step."""

        if self._last_selective_mask_step == int(step) and self._cached_selective_masks is not None:
            return self._cached_selective_masks, {}
        masks = self._selective_prior_active_masks(replay, step)
        metrics = self._selective_prior_phase_from_coverage(masks)
        self._last_selective_mask_step = int(step)
        self._cached_selective_masks = masks
        return masks, metrics

    def update_selective_discriminator(
        self,
        *,
        expert_obs: dict[str, torch.Tensor] | torch.Tensor,
        expert_z: torch.Tensor,
        validated_obs: dict[str, torch.Tensor] | torch.Tensor,
        validated_z: torch.Tensor,
        validated_confidence: torch.Tensor,
        bad_obs: dict[str, torch.Tensor] | torch.Tensor,
        bad_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> dict[str, torch.Tensor]:
        """Fit E,V -> +1 and high-confidence B -> -1 with weighted LSGAN."""

        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            stage = self._training_stage("discriminator")
            if stage is None:
                expert_logits = self._model._discriminator.compute_logits(expert_obs, expert_z)
                validated_logits = self._model._discriminator.compute_logits(validated_obs, validated_z)
                bad_logits = self._model._discriminator.compute_logits(bad_obs, bad_z)
            else:
                expert_logits = stage(obs=expert_obs, z=expert_z)
                validated_logits = stage(obs=validated_obs, z=validated_z)
                bad_logits = stage(obs=bad_obs, z=bad_z)
            expert_loss_raw = 0.5 * (expert_logits - 1.0).square().mean()
            expert_loss = self.cfg.train.selective_prior_expert_fraction * expert_loss_raw
            confidence = validated_confidence.to(self.device).float().clamp(0.0, 1.0)
            validated_error = 0.5 * (validated_logits - 1.0).square()
            validated_loss = (validated_error * confidence).sum() / confidence.sum().clamp_min(1.0)
            validated_loss_raw = validated_loss
            validated_loss = (
                self.cfg.train.selective_prior_validated_fraction * self.cfg.train.selective_prior_validated_weight * validated_loss_raw
            )
            bad_loss_raw = 0.5 * (bad_logits + 1.0).square().mean()
            bad_loss = self.cfg.train.selective_prior_bad_fraction * bad_loss_raw
            data_loss = expert_loss + validated_loss + bad_loss
            weighted_gp = torch.zeros((), device=self.device, dtype=data_loss.dtype)
            loss = data_loss
            if grad_penalty is not None:
                # Keep the existing GP definition and its experiment control.
                # Expert-vs-BAD is the actual positive/negative boundary; V is
                # a lower-confidence member of the same positive class.
                count = min(expert_z.shape[0], bad_z.shape[0])
                gp = self.gradient_penalty_wgan(
                    tree_map(lambda value: value[:count], expert_obs),
                    expert_z[:count],
                    tree_map(lambda value: value[:count], bad_obs),
                    bad_z[:count],
                )
                weighted_gp = float(grad_penalty) * gp
                loss = loss + weighted_gp

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._sync_gradients_if_manual(self._model._discriminator.parameters())
        self.discriminator_optimizer.step()

        with torch.no_grad():
            ev_auc = approximate_pairwise_auc(expert_logits, validated_logits)
            vb_auc = approximate_pairwise_auc(validated_logits, bad_logits)
            state = self._selective_prior_state
            state.discriminator_update_count += 1
            locally_ready = (
                expert_logits.mean().item() >= self.cfg.train.selective_prior_d_positive_min
                and validated_logits.mean().item() >= self.cfg.train.selective_prior_d_positive_min
                and bad_logits.mean().item() <= self.cfg.train.selective_prior_d_bad_max
                and abs(expert_logits.mean().item() - validated_logits.mean().item())
                <= self.cfg.train.selective_prior_d_expert_validated_gap_max
                and ev_auc.item() <= self.cfg.train.selective_prior_d_expert_validated_auc_max
                and vb_auc.item() >= self.cfg.train.selective_prior_d_validated_bad_auc_min
            )
            globally_ready = self._global_all(locally_ready)
            state.d_ready_streak = state.d_ready_streak + 1 if globally_ready else 0
            if state.phase_enum > PriorPhase.FIT_D:
                state.d_health_failure_streak = 0 if globally_ready else state.d_health_failure_streak + 1
                if state.d_health_failure_streak >= self.cfg.train.selective_prior_d_ready_streak:
                    state.set_phase(PriorPhase.FIT_D)
                    state.qd_reward_version = -1
                    state.qd_ready_streak = 0
                    state.qd_update_count = 0
                    state.d_health_failure_streak = 0
            if (
                state.phase_enum == PriorPhase.FIT_D
                and state.discriminator_update_count >= self.cfg.train.selective_prior_d_min_updates
                and state.d_ready_streak >= self.cfg.train.selective_prior_d_ready_streak
            ):
                self._snapshot_selective_reward_model()
                state.set_phase(PriorPhase.FIT_QD)
                state.qd_ready_streak = 0

            return {
                "disc_loss": loss.detach(),
                "disc/data_loss": data_loss.detach(),
                "disc/expert_lsgan": expert_loss.detach(),
                "disc/expert_lsgan_raw": expert_loss_raw.detach(),
                "disc/validated_lsgan_weighted": validated_loss.detach(),
                "disc/validated_lsgan_raw": validated_loss_raw.detach(),
                "disc/bad_lsgan": bad_loss.detach(),
                "disc/bad_lsgan_raw": bad_loss_raw.detach(),
                "disc/gp_raw": gp.detach() if grad_penalty is not None else torch.zeros_like(loss.detach()),
                "disc/gp_weighted": weighted_gp.detach(),
                "disc/total_loss": loss.detach(),
                "disc/selective_expert_mean": expert_logits.mean().detach(),
                "disc/selective_validated_mean": validated_logits.mean().detach(),
                "disc/selective_bad_mean": bad_logits.mean().detach(),
                "disc/selective_expert_validated_auc": ev_auc.detach(),
                "disc/selective_validated_bad_auc": vb_auc.detach(),
                "prior/d_calibration_ready": torch.tensor(float(globally_ready), device=self.device),
                "prior/d_ready_streak": torch.tensor(float(state.d_ready_streak), device=self.device),
            }

    def update_selective_prior_critic(
        self,
        *,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor] | torch.Tensor,
        z: torch.Tensor,
        next_z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fit Q_D only on verified-support interior transitions."""

        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            with torch.no_grad():
                if self._prior_reward_discriminator is None:
                    raise RuntimeError("Q_D cannot train without a calibrated reward snapshot")
                reward_logits = self._prior_reward_discriminator.compute_logits(obs=obs, z=z)
                reward = self.discriminator_reward_from_logits(reward_logits)
                dist_next = self._model._actor(next_obs, next_z, self._model.cfg.actor_std)
                next_action = dist_next.sample(clip=self.cfg.train.stddev_clip)
                next_qs = self._model._target_critic(next_obs, next_z, next_action)
                q_mean, q_unc, next_v = self.get_targets_uncertainty(next_qs, self.cfg.train.critic_pessimism_penalty)
                target_q = reward + discount * next_v
                expanded = target_q.expand(num_parallel, -1, -1)
            stage = self._training_stage("critic")
            critic = self._model._critic if stage is None else stage
            qs = critic(obs, z, action)
            critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._sync_gradients_if_manual(self._model._critic.parameters())
        self.critic_optimizer.step()

        with torch.no_grad():
            abs_scale = q_mean.abs().mean().clamp_min(1.0e-8)
            relative_uncertainty = q_unc.mean() / abs_scale
            state = self._selective_prior_state
            state.qd_update_count += 1
            globally_ready = self._global_all(
                torch.isfinite(critic_loss).item()
                and relative_uncertainty.item() <= self.cfg.train.selective_prior_qd_relative_uncertainty_max
            )
            state.qd_ready_streak = state.qd_ready_streak + 1 if globally_ready else 0
            if state.phase_enum == PriorPhase.ACTOR_PRIOR:
                state.qd_health_failure_streak = 0 if globally_ready else state.qd_health_failure_streak + 1
                if state.qd_health_failure_streak >= self.cfg.train.selective_prior_qd_ready_streak:
                    state.set_phase(PriorPhase.FIT_QD)
                    state.qd_ready_streak = 0
                    state.qd_health_failure_streak = 0
            if (
                state.phase_enum == PriorPhase.FIT_QD
                and state.qd_update_count >= self.cfg.train.selective_prior_qd_min_updates
                and state.qd_ready_streak >= self.cfg.train.selective_prior_qd_ready_streak
                and state.qd_reward_version == state.discriminator_version
                and state.qd_bank_version == state.discriminator_bank_version
            ):
                state.set_phase(PriorPhase.ACTOR_PRIOR)
            return {
                "target_Q": target_q.mean().detach(),
                "Q1": qs.mean().detach(),
                "mean_next_Q": q_mean.mean().detach(),
                "unc_Q": q_unc.mean().detach(),
                "qd/target_abs_scale": abs_scale.detach(),
                "qd/relative_uncertainty": relative_uncertainty.detach(),
                "critic_loss": critic_loss.detach(),
                "mean_disc_reward": reward.mean().detach(),
                "prior/qd_calibration_ready": torch.tensor(float(globally_ready), device=self.device),
                "prior/qd_ready_streak": torch.tensor(float(state.qd_ready_streak), device=self.device),
            }

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        if self._selective_prior_enabled:
            return self._update_selective_prior(replay_buffer, step)
        behavior_prior_enabled = self._behavior_prior_enabled
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        main_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        dedicated_prior = behavior_prior_enabled and "prior" in replay_buffer
        # No dedicated replay means the original UFO topology: D, Q_D and the
        # Actor's behavior-prior term use the same main-terrain batch.  Keeping
        # the optional prior provider preserves the canonical-plane ablation
        # without forking the optimizer implementation.
        prior_batch = None
        if behavior_prior_enabled:
            prior_batch = replay_buffer["prior"].sample(self.cfg.train.batch_size) if dedicated_prior else main_batch

        main_obs, main_action, main_next_obs = (
            tree_map(lambda x: x.to(self.device), main_batch["observation"]),
            main_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), main_batch["next"]["observation"]),
        )
        main_discount = self.cfg.train.discount * ~main_batch["next"]["terminated"].to(self.device)
        prior_obs = prior_action = prior_next_obs = prior_discount = None
        if behavior_prior_enabled:
            if prior_batch is None:
                raise RuntimeError("behavior_prior_enabled requires a behavior-prior batch")
            prior_obs, prior_action, prior_next_obs = (
                tree_map(lambda x: x.to(self.device), prior_batch["observation"]),
                prior_batch["action"].to(self.device),
                tree_map(lambda x: x.to(self.device), prior_batch["next"]["observation"]),
            )
            prior_discount = prior_transition_discount(
                prior_batch,
                gamma=self.cfg.train.discount,
                device=self.device,
            )
        expert_obs, expert_next_obs = (
            tree_map(lambda x: x.to(self.device), expert_batch["observation"]),
            tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"]),
        )

        if self._heading_context_enabled:
            required_main = {
                "heading_next",
                "heading_z_next",
                "heading_valid",
                "heading_source_type",
                "heading_reward",
                "heading_context_continues",
                "root_heading_xy",
                "next_root_heading_xy",
                "transition_terminated",
                "transition_truncated",
            }
            required_prior = {
                "heading_next",
                "heading_valid",
                "heading_source_type",
                "root_heading_xy",
                "next_root_heading_xy",
            }
            missing_main = sorted(required_main.difference(main_batch))
            missing_prior = sorted(required_prior.difference(prior_batch)) if behavior_prior_enabled and prior_batch is not None else []
            if missing_main or missing_prior:
                raise RuntimeError(
                    "Heading-context replay metadata is incomplete; start a new replay buffer. "
                    f"main_missing={missing_main} prior_missing={missing_prior}"
                )
            # Trajectory replay obtains next observations from the next slot,
            # whose command may already have switched. Rebind s_{t+1} to the
            # context that generated action_t.
            main_next_obs = dict(main_next_obs)
            main_next_obs["heading"] = main_batch["heading_next"].to(self.device)
            if dedicated_prior:
                prior_next_obs = dict(prior_next_obs)
                prior_next_obs["heading"] = prior_batch["heading_next"].to(self.device)

        # Shared running statistics are updated by the main terrain stream
        # only. The canonical-plane prior is transformed in eval mode below.
        self._model._obs_normalizer(main_obs)
        self._model._obs_normalizer(main_next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            main_obs, main_next_obs = (
                self._model._obs_normalizer(main_obs),
                self._model._obs_normalizer(main_next_obs),
            )
            if dedicated_prior:
                prior_obs, prior_next_obs = (
                    self._model._obs_normalizer(prior_obs),
                    self._model._obs_normalizer(prior_next_obs),
                )
            elif behavior_prior_enabled:
                prior_obs, prior_next_obs = main_obs, main_next_obs
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        prior_terrain_var = None
        if dedicated_prior and isinstance(prior_obs, dict) and "terrain_priv" in prior_obs:
            raw_prior = prior_batch["observation"].get("terrain_priv")
            raw_prior_next = prior_batch["next"]["observation"].get("terrain_priv")
            if raw_prior is None or raw_prior_next is None:
                raise RuntimeError("Prior observations must contain terrain_priv at t and t+1")
            # The Agent intentionally accepts any prior_batch provider so A*/B/C
            # ablations do not fork the optimizer code. Workspace collection is
            # the source-of-truth assertion that C's plane stream is raw zero;
            # when this batch is canonical plane, additionally prove that shared
            # normalization did not create sample-wise geometry information.
            raw_plane = torch.count_nonzero(raw_prior).item() == 0 and torch.count_nonzero(raw_prior_next).item() == 0
            if raw_plane:
                prior_terrain_var = torch.stack(
                    (
                        prior_obs["terrain_priv"].float().var(dim=0, unbiased=False).amax(),
                        prior_next_obs["terrain_priv"].float().var(dim=0, unbiased=False).amax(),
                    )
                ).amax()
                if prior_terrain_var.item() > 1.0e-7:
                    raise RuntimeError(
                        "Canonical-plane terrain_priv acquired sample-wise variation during normalization: "
                        f"max_variance={prior_terrain_var.item():.3e}"
                    )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        main_rollout_z = main_batch["z"].to(self.device)
        prior_rollout_z = prior_batch["z"].to(self.device) if behavior_prior_enabled and prior_batch is not None else None
        main_z = main_rollout_z
        prior_z = prior_rollout_z
        main_collection_obs = main_obs
        main_collection_next_obs = main_next_obs

        # D sees the selected policy stream's original rollout z, matching the
        # original UFO occupancy supervision. Relabeling happens only after D.
        metrics: dict[str, torch.Tensor] = {}
        if behavior_prior_enabled:
            grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
            metrics.update(
                self.update_discriminator(
                    expert_obs=expert_obs,
                    expert_z=expert_z,
                    train_obs=prior_obs,
                    train_z=prior_z,
                    grad_penalty=grad_penalty,
                )
            )

        if self._heading_context_enabled:
            expert_heading_xy = expert_batch["heading_forward_xy"].to(self.device)
            expert_next_heading_xy = expert_batch["next"]["heading_forward_xy"].to(self.device)
            if dedicated_prior:
                (
                    main_context,
                    prior_context,
                    main_obs,
                    main_next_obs,
                    prior_obs,
                    prior_next_obs,
                ) = self._relabel_main_and_prior_context(
                    main_obs=main_obs,
                    main_next_obs=main_next_obs,
                    prior_obs=prior_obs,
                    prior_next_obs=prior_next_obs,
                    main_batch=main_batch,
                    prior_batch=prior_batch,
                    expert_z=expert_z,
                    expert_heading_xy=expert_heading_xy,
                    expert_next_heading_xy=expert_next_heading_xy,
                    main_rollout_z=main_rollout_z,
                    prior_rollout_z=prior_rollout_z,
                )
                main_z, prior_z = main_context.z, prior_context.z
            else:
                main_context, main_obs, main_next_obs = self._relabel_main_context(
                    main_obs=main_obs,
                    main_next_obs=main_next_obs,
                    main_batch=main_batch,
                    expert_z=expert_z,
                    expert_heading_xy=expert_heading_xy,
                    expert_next_heading_xy=expert_next_heading_xy,
                    main_rollout_z=main_rollout_z,
                )
                main_z = main_context.z
                if behavior_prior_enabled:
                    prior_context = main_context
                    prior_obs, prior_next_obs, prior_z = main_obs, main_next_obs, main_z
                else:
                    prior_context = None
        else:
            if dedicated_prior:
                main_z, prior_z = self._relabel_main_and_prior_z(
                    main_next_obs=main_next_obs,
                    prior_next_obs=prior_next_obs,
                    expert_z=expert_z,
                    main_rollout_z=main_rollout_z,
                    prior_rollout_z=prior_rollout_z,
                )
            else:
                main_z = self._relabel_main_z(
                    main_next_obs=main_next_obs,
                    expert_z=expert_z,
                    main_rollout_z=main_rollout_z,
                )
                if behavior_prior_enabled:
                    prior_obs, prior_next_obs, prior_z = main_obs, main_next_obs, main_z

        # Compute the heavier logit quantiles only once per collector step,
        # rather than once for every optimizer update at that step.  This is
        # diagnostics-only and does not alter gradients, losses, or RNG state.
        if behavior_prior_enabled and getattr(self, "_last_discriminator_diagnostic_step", None) != int(step):
            metrics.update(
                self._discriminator_conditioning_diagnostics(
                    expert_obs=expert_obs,
                    expert_z=expert_z,
                    prior_obs=prior_obs,
                    prior_rollout_z=prior_rollout_z,
                    prior_relabel_z=prior_z,
                )
            )
            self._last_discriminator_diagnostic_step = int(step)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=main_obs,
                action=main_action,
                discount=main_discount,
                next_obs=main_next_obs,
                goal=main_next_obs,
                z=main_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        if behavior_prior_enabled:
            metrics.update(
                self.update_critic(
                    obs=prior_obs,
                    action=prior_action,
                    discount=prior_discount,
                    next_obs=prior_next_obs,
                    z=prior_z,
                )
            )
        # compute scalar auxiliary reward as a weighted sum of the auxiliary rewards
        aux_reward = torch.zeros(
            (self.cfg.train.batch_size, 1),
            device=self.device,
            dtype=torch.float32,
        )
        for aux_reward_name in self.cfg.aux_rewards:
            # let's log even this information
            metrics[f"aux_rew/{aux_reward_name}"] = main_batch["aux_rewards"][aux_reward_name].mean()
            aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * main_batch["aux_rewards"][aux_reward_name].to(self.device)

        aux_reward = self._model._aux_reward_normalizer(aux_reward)

        metrics.update(
            self.update_aux_critic(
                obs=main_obs,
                action=main_action,
                discount=main_discount,
                aux_reward=aux_reward,
                next_obs=main_next_obs,
                z=main_z,
            )
        )
        heading_branch_active = self._heading_critic_enabled
        if heading_branch_active:
            heading_discount = (
                self.cfg.train.discount
                * main_batch["heading_context_continues"].to(self.device)
                * ~main_batch["transition_terminated"].to(self.device)
                * ~main_batch["transition_truncated"].to(self.device)
            )
            metrics.update(
                self.update_heading_critic(
                    obs=main_collection_obs,
                    action=main_action,
                    discount=heading_discount,
                    heading_reward=main_batch["heading_reward"].to(self.device),
                    heading_valid=main_batch["heading_valid"].to(self.device),
                    next_obs=main_collection_next_obs,
                    z=main_batch["z"].to(self.device),
                    next_z=main_batch["heading_z_next"].to(self.device),
                )
            )
        metrics.update(
            self._run_actor_update(
                main_obs=main_obs,
                main_z=main_z,
                prior_obs=prior_obs if behavior_prior_enabled else None,
                prior_z=prior_z if behavior_prior_enabled else None,
                heading_obs=main_collection_obs if heading_branch_active else None,
                heading_z=main_batch["z"].to(self.device) if heading_branch_active else None,
                heading_valid=(main_batch["heading_valid"].to(self.device) if heading_branch_active else None),
                prior_is_main=behavior_prior_enabled and not dedicated_prior,
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics["cfg/behavior_prior_enabled"] = torch.tensor(
            float(behavior_prior_enabled),
            device=self.device,
        )
        if behavior_prior_enabled:
            metrics["prior/discount_mean"] = prior_discount.float().mean().detach()
            metrics["prior/source_is_main"] = torch.tensor(float(not dedicated_prior), device=self.device)
        if prior_terrain_var is not None:
            metrics["prior/terrain_priv_normalized_var_max"] = prior_terrain_var.detach()
        if self._heading_context_enabled:
            metrics["heading/collection_valid_fraction"] = main_batch["heading_valid"].float().mean().to(self.device)
            metrics["heading/relabel_valid_fraction_main"] = main_context.heading_valid.float().mean().detach()
            if behavior_prior_enabled:
                metrics["heading/relabel_valid_fraction_prior"] = prior_context.heading_valid.float().mean().detach()
            metrics["cfg/reg_coeff_heading"] = torch.tensor(self.cfg.train.reg_coeff_heading, device=self.device)

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            if behavior_prior_enabled:
                _soft_update_params(
                    self._critic_map_paramlist,
                    self._target_critic_map_paramlist,
                    self.cfg.train.critic_target_tau,
                )
            _soft_update_params(
                self._aux_critic_map_paramlist,
                self._aux_target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )
            if heading_branch_active:
                _soft_update_params(
                    self._heading_critic_map_paramlist,
                    self._heading_target_critic_map_paramlist,
                    self.cfg.train.critic_target_tau,
                )

        return metrics

    def _update_selective_prior(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        """Joint update with staged, selectively-labelled behavior prior.

        Main FB/Aux/QH retains the existing counterfactual BehaviorContext
        relabel.  D/Q_D/Actor-D use only collection-time context from verified
        replay views and never relabel z.
        """

        replay = replay_buffer["train"]
        metrics = self._refresh_selective_prior_labels(replay, step)
        masks, coverage_metrics = self._selective_masks_and_coverage(replay, step)
        metrics.update(coverage_metrics)
        phase_at_start = self._selective_prior_state.phase_enum

        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        main_batch = replay.sample(self.cfg.train.batch_size)
        validated_batch = bad_batch = qd_batch = actor_prior_batch = None
        if phase_at_start >= PriorPhase.FIT_D:
            validated_count = max(
                1,
                round(self.cfg.train.batch_size * self.cfg.train.selective_prior_validated_fraction),
            )
            bad_count = max(
                1,
                round(self.cfg.train.batch_size * self.cfg.train.selective_prior_bad_fraction),
            )
            strata = replay.storage["prior_motion_id"].squeeze(-1)
            validated_batch = replay.sample_from_mask(masks["balanced_validated"], validated_count, include_next=False, strata=strata)
            bad_batch = replay.sample_from_mask(masks["balanced_bad"], bad_count, include_next=False, strata=strata)
        if phase_at_start >= PriorPhase.FIT_QD:
            qd_batch = replay.sample_from_mask(masks["qd"], self.cfg.train.batch_size, include_next=True)
        if phase_at_start >= PriorPhase.ACTOR_PRIOR:
            actor_prior_batch = replay.sample_from_mask(masks["actor"], self.cfg.train.batch_size, include_next=False)

        main_obs = tree_map(lambda value: value.to(self.device), main_batch["observation"])
        main_next_obs = tree_map(lambda value: value.to(self.device), main_batch["next"]["observation"])
        main_action = main_batch["action"].to(self.device)
        main_discount = self.cfg.train.discount * ~main_batch["next"]["terminated"].to(self.device)
        if self._heading_context_enabled:
            main_next_obs = dict(main_next_obs)
            main_next_obs["heading"] = main_batch["heading_next"].to(self.device)
        expert_obs = tree_map(lambda value: value.to(self.device), expert_batch["observation"])
        expert_next_obs = tree_map(lambda value: value.to(self.device), expert_batch["next"]["observation"])

        def policy_obs(batch, *, with_next: bool):
            obs = tree_map(lambda value: value.to(self.device), batch["observation"])
            next_obs = None
            if with_next:
                next_obs = tree_map(lambda value: value.to(self.device), batch["next"]["observation"])
                if self._heading_context_enabled:
                    next_obs = dict(next_obs)
                    next_obs["heading"] = batch["heading_next"].to(self.device)
            return obs, next_obs

        validated_obs = bad_obs = qd_obs = qd_next_obs = actor_prior_obs = None
        if validated_batch is not None:
            validated_obs, _ = policy_obs(validated_batch, with_next=False)
            bad_obs, _ = policy_obs(bad_batch, with_next=False)
        if qd_batch is not None:
            qd_obs, qd_next_obs = policy_obs(qd_batch, with_next=True)
        if actor_prior_batch is not None:
            actor_prior_obs, _ = policy_obs(actor_prior_batch, with_next=False)

        # Only the all-terrain main stream owns normalizer statistics. Every
        # selective view and expert batch is a read-only transform.
        self._model._obs_normalizer(main_obs)
        self._model._obs_normalizer(main_next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            main_obs = self._model._obs_normalizer(main_obs)
            main_next_obs = self._model._obs_normalizer(main_next_obs)
            expert_obs = self._model._obs_normalizer(expert_obs)
            expert_next_obs = self._model._obs_normalizer(expert_next_obs)
            if validated_obs is not None:
                validated_obs = self._model._obs_normalizer(validated_obs)
                bad_obs = self._model._obs_normalizer(bad_obs)
            if qd_obs is not None:
                qd_obs = self._model._obs_normalizer(qd_obs)
                qd_next_obs = self._model._obs_normalizer(qd_next_obs)
            if actor_prior_obs is not None:
                actor_prior_obs = self._model._obs_normalizer(actor_prior_obs)

        main_collection_obs = main_obs
        main_collection_next_obs = main_next_obs

        torch.compiler.cudagraph_mark_step_begin()
        expert_z_all = self.encode_expert(next_obs=expert_next_obs)
        main_rollout_z = main_batch["z"].to(self.device)

        if phase_at_start >= PriorPhase.FIT_D:
            expert_count = max(
                1,
                round(self.cfg.train.batch_size * self.cfg.train.selective_prior_expert_fraction),
            )
            grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
            metrics.update(
                self.update_selective_discriminator(
                    expert_obs=tree_map(lambda value: value[:expert_count], expert_obs),
                    expert_z=expert_z_all[:expert_count],
                    validated_obs=validated_obs,
                    validated_z=validated_batch["z"].to(self.device),
                    validated_confidence=validated_batch["prior_confidence"].to(self.device),
                    bad_obs=bad_obs,
                    bad_z=bad_batch["z"].to(self.device),
                    grad_penalty=grad_penalty,
                )
            )
            # A failed D health check can regress ACTOR_PRIOR/FIT_QD during
            # this very update.  Fail closed immediately; do not allow one
            # final Q_D or Actor-D step against an invalidated reward model.
            if self._selective_prior_state.phase_enum < PriorPhase.FIT_QD:
                qd_batch = None
                qd_obs = None
                qd_next_obs = None
            if self._selective_prior_state.phase_enum < PriorPhase.ACTOR_PRIOR:
                actor_prior_batch = None
                actor_prior_obs = None

        if self._heading_context_enabled:
            expert_heading_xy = expert_batch["heading_forward_xy"].to(self.device)
            expert_next_heading_xy = expert_batch["next"]["heading_forward_xy"].to(self.device)
            main_context, main_obs, main_next_obs = self._relabel_main_context(
                main_obs=main_obs,
                main_next_obs=main_next_obs,
                main_batch=main_batch,
                expert_z=expert_z_all,
                expert_heading_xy=expert_heading_xy,
                expert_next_heading_xy=expert_next_heading_xy,
                main_rollout_z=main_rollout_z,
            )
            main_z = main_context.z
        else:
            main_z = self._relabel_main_z(
                main_next_obs=main_next_obs,
                expert_z=expert_z_all,
                main_rollout_z=main_rollout_z,
            )

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        metrics.update(
            self.update_fb(
                obs=main_obs,
                action=main_action,
                discount=main_discount,
                next_obs=main_next_obs,
                goal=main_next_obs,
                z=main_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )

        if qd_batch is not None:
            qd_done = qd_batch["transition_terminated"].to(self.device) | qd_batch["transition_truncated"].to(self.device)
            qd_next_z = qd_batch["heading_z_next"].to(self.device) if self._heading_context_enabled else qd_batch["z"].to(self.device)
            metrics.update(
                self.update_selective_prior_critic(
                    obs=qd_obs,
                    action=qd_batch["action"].to(self.device),
                    discount=self.cfg.train.discount * ~qd_done,
                    next_obs=qd_next_obs,
                    z=qd_batch["z"].to(self.device),
                    next_z=qd_next_z,
                )
            )
            if self._selective_prior_state.phase_enum < PriorPhase.ACTOR_PRIOR:
                actor_prior_batch = None
                actor_prior_obs = None

        aux_reward = torch.zeros((self.cfg.train.batch_size, 1), device=self.device, dtype=torch.float32)
        for name in self.cfg.aux_rewards:
            metrics[f"aux_rew/{name}"] = main_batch["aux_rewards"][name].mean()
            aux_reward += self.cfg.aux_rewards_scaling[name] * main_batch["aux_rewards"][name].to(self.device)
        aux_reward = self._model._aux_reward_normalizer(aux_reward)
        metrics.update(
            self.update_aux_critic(
                obs=main_obs,
                action=main_action,
                discount=main_discount,
                aux_reward=aux_reward,
                next_obs=main_next_obs,
                z=main_z,
            )
        )

        heading_branch_active = self._heading_critic_enabled
        if heading_branch_active:
            heading_discount = (
                self.cfg.train.discount
                * main_batch["heading_context_continues"].to(self.device)
                * ~main_batch["transition_terminated"].to(self.device)
                * ~main_batch["transition_truncated"].to(self.device)
            )
            metrics.update(
                self.update_heading_critic(
                    obs=main_collection_obs,
                    action=main_action,
                    discount=heading_discount,
                    heading_reward=main_batch["heading_reward"].to(self.device),
                    heading_valid=main_batch["heading_valid"].to(self.device),
                    next_obs=main_collection_next_obs,
                    z=main_rollout_z,
                    next_z=main_batch["heading_z_next"].to(self.device),
                )
            )

        actor_prior_z = actor_prior_batch["z"].to(self.device) if actor_prior_batch is not None else None
        metrics.update(
            self._run_actor_update(
                main_obs=main_obs,
                main_z=main_z,
                prior_obs=actor_prior_obs,
                prior_z=actor_prior_z,
                heading_obs=main_collection_obs if heading_branch_active else None,
                heading_z=main_rollout_z if heading_branch_active else None,
                heading_valid=(main_batch["heading_valid"].to(self.device) if heading_branch_active else None),
                prior_is_main=False,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.cfg.train.fb_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.cfg.train.fb_target_tau)
            if qd_batch is not None:
                _soft_update_params(
                    self._critic_map_paramlist,
                    self._target_critic_map_paramlist,
                    self.cfg.train.critic_target_tau,
                )
            _soft_update_params(
                self._aux_critic_map_paramlist,
                self._aux_target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )
            if heading_branch_active:
                _soft_update_params(
                    self._heading_critic_map_paramlist,
                    self._heading_target_critic_map_paramlist,
                    self.cfg.train.critic_target_tau,
                )

        metrics["cfg/behavior_prior_enabled"] = torch.ones((), device=self.device)
        metrics["cfg/selective_prior_enabled"] = torch.ones((), device=self.device)
        metrics["prior/phase"] = torch.tensor(float(self._selective_prior_state.phase), device=self.device)
        metrics["prior/d_enabled"] = torch.tensor(float(phase_at_start >= PriorPhase.FIT_D), device=self.device)
        metrics["prior/qd_enabled"] = torch.tensor(float(qd_batch is not None), device=self.device)
        metrics["prior/actor_d_enabled"] = torch.tensor(float(actor_prior_batch is not None), device=self.device)
        metrics["prior/bank_version"] = torch.tensor(float(self._selective_prior_state.bank_version), device=self.device)
        metrics["prior/discriminator_version"] = torch.tensor(float(self._selective_prior_state.discriminator_version), device=self.device)
        metrics["prior/qd_reward_version"] = torch.tensor(float(self._selective_prior_state.qd_reward_version), device=self.device)
        return metrics

    def update_aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            # compute target critic
            with torch.no_grad():
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_aux_critic(next_obs, z, next_action)  # num_parallel x batch x 1
                # TODO AL: should we have aux_critic parameters here?
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.aux_critic_pessimism_penalty)
                target_Q = aux_reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            # compute critic loss
            aux_critic_stage = self._training_stage("aux_critic")
            aux_critic = self._model._aux_critic if aux_critic_stage is None else aux_critic_stage
            Qs = aux_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        self._sync_gradients_if_manual(self._model._aux_critic.parameters())
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_auxQ": target_Q.mean().detach(),
                "auxQ1": Qs.mean().detach(),
                "mean_next_auxQ": Q_mean.mean().detach(),
                "unc_auxQ": Q_unc.mean().detach(),
                "aux_critic_loss": aux_critic_loss.mean().detach(),
                "mean_aux_reward": aux_reward.mean().detach(),
            }
        return output_metrics

    def update_heading_critic(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        heading_reward: torch.Tensor,
        heading_valid: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        z: torch.Tensor,
        next_z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Train Q_H only on collection-time context-matched transitions."""

        valid = heading_valid.to(device=self.device, dtype=torch.float32)
        valid_count = valid.sum().clamp_min(1.0)
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.heading_critic.num_parallel
            with torch.no_grad():
                dist = self._model._actor(next_obs, next_z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_heading_critic(next_obs, next_z, next_action)
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(
                    next_Qs,
                    self.cfg.train.heading_critic_pessimism_penalty,
                )
                target_Q = heading_reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            heading_stage = self._training_stage("heading_critic")
            heading_critic = self._model._heading_critic if heading_stage is None else heading_stage
            Qs = heading_critic(obs, z, action)
            squared_error = torch.square(Qs - expanded_targets)
            heading_critic_loss = 0.5 * (squared_error * valid.unsqueeze(0)).sum() / valid_count

        self.heading_critic_optimizer.zero_grad(set_to_none=True)
        heading_critic_loss.backward()
        self._sync_gradients_if_manual(self._model._heading_critic.parameters())
        self.heading_critic_optimizer.step()

        with torch.no_grad():
            valid_denom = valid_count
            return {
                "heading/target_Q": (target_Q * valid).sum().detach() / valid_denom,
                "heading/Q": (Qs.mean(dim=0) * valid).sum().detach() / valid_denom,
                "heading/next_Q": (Q_mean * valid).sum().detach() / valid_denom,
                "heading/uncertainty": (Q_unc * valid).sum().detach() / valid_denom,
                "heading/critic_loss": heading_critic_loss.detach(),
                "heading/reward": (heading_reward * valid).sum().detach() / valid_denom,
            }

    def update_actor(
        self,
        main_obs: torch.Tensor | dict[str, torch.Tensor],
        main_z: torch.Tensor,
        prior_obs: torch.Tensor | dict[str, torch.Tensor] | None,
        prior_z: torch.Tensor | None,
        heading_obs: torch.Tensor | dict[str, torch.Tensor] | None = None,
        heading_z: torch.Tensor | None = None,
        heading_valid: torch.Tensor | None = None,
        prior_is_main: bool = False,
        clip_grad_norm: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        actor_stage = self._training_stage("actor")
        actor = self._model._actor if actor_stage is None else actor_stage
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            # One Actor forward and optimizer step combines exactly the active
            # objective branches.  The formal no-D path omits the prior batch
            # entirely instead of feeding main terrain data through Q_D.
            obs_parts = [main_obs]
            z_parts = [main_z]
            if prior_obs is not None:
                if prior_z is None:
                    raise ValueError("prior_obs requires prior_z")
                if not prior_is_main:
                    obs_parts.append(prior_obs)
                    z_parts.append(prior_z)
            elif prior_z is not None:
                raise ValueError("prior_z requires prior_obs")
            if heading_obs is not None:
                if heading_z is None or heading_valid is None:
                    raise ValueError("Heading Actor branch requires heading_z and heading_valid")
                obs_parts.append(heading_obs)
                z_parts.append(heading_z)
            combined_obs = tree_map(lambda *parts: torch.cat(parts, dim=0), *obs_parts)
            combined_z = torch.cat(z_parts, dim=0)
            dist = actor(combined_obs, combined_z, self._model.cfg.actor_std)
            combined_action = dist.sample(clip=self.cfg.train.stddev_clip)
            main_batch_size = main_z.shape[0]
            main_action = combined_action[:main_batch_size]
            offset = main_batch_size
            prior_action = None
            if prior_obs is not None:
                if prior_z is None:
                    raise ValueError("prior_obs requires prior_z")
                if prior_is_main:
                    prior_action = main_action
                else:
                    prior_batch_size = prior_z.shape[0]
                    prior_action = combined_action[offset : offset + prior_batch_size]
                    offset += prior_batch_size
            heading_action = combined_action[offset:] if heading_obs is not None else None

            # Main RP1 terrain realization and physical auxiliary objectives.
            Qs_aux = self._model._aux_critic(main_obs, main_z, main_action)
            _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, self.cfg.train.actor_pessimism_penalty)  # batch

            Fs = self._model._forward_map(main_obs, main_z, main_action)
            Qs_fb = (Fs * main_z).sum(-1)  # num_parallel x batch
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

            weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            actor_loss = -Q_aux.mean() * self.cfg.train.reg_coeff_aux * weight - Q_fb.mean()
            Q_discriminator_mean = torch.zeros((), device=self.device, dtype=Q_fb.dtype)
            actor_prior_low_uncertainty_fraction = torch.zeros((), device=self.device, dtype=Q_fb.dtype)
            if prior_obs is not None:
                Qs_discriminator = self._model._critic(prior_obs, prior_z, prior_action)
                Q_discriminator_raw_mean, Q_discriminator_unc, Q_discriminator = self.get_targets_uncertainty(
                    Qs_discriminator,
                    self.cfg.train.actor_pessimism_penalty,
                )
                if self._selective_prior_enabled:
                    relative_uncertainty = Q_discriminator_unc.reshape(-1) / (Q_discriminator_raw_mean.abs().reshape(-1) + 1.0)
                    low_uncertainty = (relative_uncertainty <= self.cfg.train.selective_prior_qd_relative_uncertainty_max).to(
                        Q_discriminator.dtype
                    )
                    actor_prior_low_uncertainty_fraction = low_uncertainty.mean()
                    Q_discriminator_mean = (Q_discriminator.reshape(-1) * low_uncertainty).sum() / low_uncertainty.sum().clamp_min(1.0)
                else:
                    Q_discriminator_mean = Q_discriminator.mean()
                actor_loss = actor_loss - Q_discriminator_mean * self.cfg.train.reg_coeff * weight
            Q_heading_mean = torch.zeros((), device=self.device, dtype=Q_fb.dtype)
            if heading_action is not None:
                Qs_heading = self._model._heading_critic(heading_obs, heading_z, heading_action)
                _, _, Q_heading = self.get_targets_uncertainty(
                    Qs_heading,
                    self.cfg.train.actor_pessimism_penalty,
                )
                valid = heading_valid.to(device=self.device, dtype=Q_heading.dtype).reshape(-1)
                Q_heading_mean = (Q_heading.reshape(-1) * valid).sum() / valid.sum().clamp_min(1.0)
                actor_loss = actor_loss - Q_heading_mean * self.cfg.train.reg_coeff_heading * weight

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._sync_gradients_if_manual(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator_mean.detach(),
                "Q_aux": Q_aux.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
                "Q_heading": Q_heading_mean.detach(),
                "prior/actor_low_qd_uncertainty_fraction": actor_prior_low_uncertainty_fraction.detach(),
            }
        return output_metrics
