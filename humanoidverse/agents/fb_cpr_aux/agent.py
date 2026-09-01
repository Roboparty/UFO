# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass
from typing import Dict

import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..behavior_context import (
    HEADING_SOURCE_EXACT_TRACKING,
    HEADING_SOURCE_INVALID,
    heading_observation,
    relative_heading_target,
)
from ..fb_cpr.agent import FBcprAgent, FBcprAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode
from .model import FBcprAuxModelConfig


def prior_transition_discount(
    prior_batch: dict[str, tp.Any],
    *,
    gamma: float,
    device: str | torch.device,
) -> torch.Tensor:
    """Return the behavior-prior discount aligned with the sampled action.

    The canonical-plane collector stores the outcome of the action in the
    current replay slot.  The legacy fallback is retained for checkpoints
    produced before the dedicated prior stream existed.
    """

    if "transition_terminated" in prior_batch and "transition_truncated" in prior_batch:
        done = prior_batch["transition_terminated"].to(device) | prior_batch[
            "transition_truncated"
        ].to(device)
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


@dataclass
class RelabeledBehaviorContext:
    """A z and its matching deployable heading observation pair."""

    z: torch.Tensor
    heading: torch.Tensor
    next_heading: torch.Tensor
    heading_valid: torch.Tensor
    source_type: torch.Tensor


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
    def _heading_context_enabled(self) -> bool:
        return bool(
            getattr(getattr(self.cfg, "model", None), "heading_context_enabled", False)
        )

    @property
    def _heading_critic_enabled(self) -> bool:
        return bool(
            getattr(getattr(self.cfg, "model", None), "heading_critic_enabled", False)
        )

    def setup_training(self) -> None:
        super().setup_training()

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
            self._heading_target_critic_map_paramlist = tuple(
                x for x in self._model._target_heading_critic.parameters()
            )
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

        main_context, main_obs, main_next_obs = merge(
            main_original, sampled_main, main_mask, main_obs, main_next_obs
        )
        prior_context, prior_obs, prior_next_obs = merge(
            prior_original, sampled_prior, prior_mask, prior_obs, prior_next_obs
        )
        return (
            main_context,
            prior_context,
            main_obs,
            main_next_obs,
            prior_obs,
            prior_next_obs,
        )

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        main_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        prior_batch = replay_buffer.get("prior", replay_buffer["train"]).sample(self.cfg.train.batch_size)

        main_obs, main_action, main_next_obs = (
            tree_map(lambda x: x.to(self.device), main_batch["observation"]),
            main_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), main_batch["next"]["observation"]),
        )
        prior_obs, prior_action, prior_next_obs = (
            tree_map(lambda x: x.to(self.device), prior_batch["observation"]),
            prior_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), prior_batch["next"]["observation"]),
        )
        main_discount = self.cfg.train.discount * ~main_batch["next"]["terminated"].to(self.device)
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
            missing_prior = sorted(required_prior.difference(prior_batch))
            if missing_main or missing_prior:
                raise RuntimeError(
                    "Heading-context replay metadata is incomplete; start a new replay buffer. "
                    f"main_missing={missing_main} prior_missing={missing_prior}"
                )
            # Trajectory replay obtains next observations from the next slot,
            # whose command may already have switched. Rebind s_{t+1} to the
            # context that generated action_t.
            main_next_obs = dict(main_next_obs)
            prior_next_obs = dict(prior_next_obs)
            main_next_obs["heading"] = main_batch["heading_next"].to(self.device)
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
            prior_obs, prior_next_obs = (
                self._model._obs_normalizer(prior_obs),
                self._model._obs_normalizer(prior_next_obs),
            )
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        prior_terrain_var = None
        if isinstance(prior_obs, dict) and "terrain_priv" in prior_obs:
            raw_prior = prior_batch["observation"].get("terrain_priv")
            raw_prior_next = prior_batch["next"]["observation"].get("terrain_priv")
            if raw_prior is None or raw_prior_next is None:
                raise RuntimeError("Prior observations must contain terrain_priv at t and t+1")
            # The Agent intentionally accepts any prior_batch provider so A*/B/C
            # ablations do not fork the optimizer code. Workspace collection is
            # the source-of-truth assertion that C's plane stream is raw zero;
            # when this batch is canonical plane, additionally prove that shared
            # normalization did not create sample-wise geometry information.
            raw_plane = (
                torch.count_nonzero(raw_prior).item() == 0
                and torch.count_nonzero(raw_prior_next).item() == 0
            )
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
        main_z = main_batch["z"].to(self.device)
        prior_z = prior_batch["z"].to(self.device)
        main_collection_obs = main_obs
        main_collection_next_obs = main_next_obs

        # D sees the prior policy's original rollout z, matching the existing
        # algorithm. Relabeling happens independently below for both critics.
        grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=prior_obs,
            train_z=prior_z,
            grad_penalty=grad_penalty,
        )

        if self._heading_context_enabled:
            expert_heading_xy = expert_batch["heading_forward_xy"].to(self.device)
            expert_next_heading_xy = expert_batch["next"]["heading_forward_xy"].to(self.device)
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
                main_rollout_z=main_z,
                prior_rollout_z=prior_z,
            )
            main_z, prior_z = main_context.z, prior_context.z
        else:
            main_z, prior_z = self._relabel_main_and_prior_z(
                main_next_obs=main_next_obs,
                prior_next_obs=prior_next_obs,
                expert_z=expert_z,
                main_rollout_z=main_z,
                prior_rollout_z=prior_z,
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
                prior_obs=prior_obs,
                prior_z=prior_z,
                heading_obs=main_collection_obs if heading_branch_active else None,
                heading_z=main_batch["z"].to(self.device) if heading_branch_active else None,
                heading_valid=(
                    main_batch["heading_valid"].to(self.device) if heading_branch_active else None
                ),
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics["prior/discount_mean"] = prior_discount.float().mean().detach()
        if prior_terrain_var is not None:
            metrics["prior/terrain_priv_normalized_var_max"] = prior_terrain_var.detach()
        if self._heading_context_enabled:
            metrics["heading/collection_valid_fraction"] = (
                main_batch["heading_valid"].float().mean().to(self.device)
            )
            metrics["heading/relabel_valid_fraction_main"] = main_context.heading_valid.float().mean().detach()
            metrics["heading/relabel_valid_fraction_prior"] = prior_context.heading_valid.float().mean().detach()
            metrics["cfg/reg_coeff_heading"] = torch.tensor(
                self.cfg.train.reg_coeff_heading, device=self.device
            )

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
        prior_obs: torch.Tensor | dict[str, torch.Tensor],
        prior_z: torch.Tensor,
        heading_obs: torch.Tensor | dict[str, torch.Tensor] | None = None,
        heading_z: torch.Tensor | None = None,
        heading_valid: torch.Tensor | None = None,
        clip_grad_norm: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        actor_stage = self._training_stage("actor")
        actor = self._model._actor if actor_stage is None else actor_stage
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            # One actor forward and one optimizer step preserve DDP reducer and
            # optimization semantics while the two loss branches use their own
            # state distributions.
            if heading_obs is None:
                combined_obs = tree_map(lambda main, prior: torch.cat((main, prior), dim=0), main_obs, prior_obs)
                combined_z = torch.cat((main_z, prior_z), dim=0)
            else:
                if heading_z is None or heading_valid is None:
                    raise ValueError("Heading Actor branch requires heading_z and heading_valid")
                combined_obs = tree_map(
                    lambda main, prior, heading: torch.cat((main, prior, heading), dim=0),
                    main_obs,
                    prior_obs,
                    heading_obs,
                )
                combined_z = torch.cat((main_z, prior_z, heading_z), dim=0)
            dist = actor(combined_obs, combined_z, self._model.cfg.actor_std)
            combined_action = dist.sample(clip=self.cfg.train.stddev_clip)
            main_batch_size = main_z.shape[0]
            main_action = combined_action[:main_batch_size]
            prior_action = combined_action[main_batch_size : 2 * main_batch_size]
            heading_action = combined_action[2 * main_batch_size :] if heading_obs is not None else None

            # Canonical-plane behavior anchor.
            Qs_discriminator = self._model._critic(prior_obs, prior_z, prior_action)
            _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.cfg.train.actor_pessimism_penalty)  # batch

            # Main RP1 terrain realization and physical auxiliary objectives.
            Qs_aux = self._model._aux_critic(main_obs, main_z, main_action)
            _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, self.cfg.train.actor_pessimism_penalty)  # batch

            Fs = self._model._forward_map(main_obs, main_z, main_action)
            Qs_fb = (Fs * main_z).sum(-1)  # num_parallel x batch
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

            weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            actor_loss = (
                -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight
                - Q_aux.mean() * self.cfg.train.reg_coeff_aux * weight
                - Q_fb.mean()
            )
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
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_aux": Q_aux.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
                "Q_heading": Q_heading_mean.detach(),
            }
        return output_metrics
