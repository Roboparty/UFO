# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from typing import Dict

import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..gcr_rl_dist.agent import GcrRlDistAgent, GcrRlDistAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode
from ...distributed import average_gradients
from .model import GcrRlDistAuxModelConfig


class GcrRlDistAuxAgentTrainConfig(GcrRlDistAgentTrainConfig):
    lr_aux_critic: float = 3e-4
    reg_coeff_aux: float = 0.02
    aux_critic_pessimism_penalty: float = 0.5


class GcrRlDistAuxAgentConfig(BaseConfig):
    name: tp.Literal["GcrRlDistAuxAgent"] = "GcrRlDistAuxAgent"
    model: GcrRlDistAuxModelConfig = GcrRlDistAuxModelConfig()
    train: GcrRlDistAuxAgentTrainConfig = GcrRlDistAuxAgentTrainConfig()
    aux_rewards: list[str] = pydantic.Field(default_factory=list)
    aux_rewards_scaling: dict[str, float] = pydantic.Field(default_factory=dict)
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim: int) -> "GcrRlDistAuxAgent":
        return self.object_class(
            obs_space=obs_space,
            action_dim=action_dim,
            cfg=self,
        )

    @property
    def object_class(self):
        return GcrRlDistAuxAgent


class GcrRlDistAuxAgent(GcrRlDistAgent):
    config_class = GcrRlDistAuxAgentConfig

    def setup_training(self) -> None:
        super().setup_training()

        self._aux_critic_map_paramlist = tuple(x for x in self._model._aux_critic.parameters())
        self._aux_target_critic_map_paramlist = tuple(x for x in self._model._target_aux_critic.parameters())

        self.aux_critic_optimizer = torch.optim.Adam(
            self._model._aux_critic.parameters(),
            lr=self.cfg.train.lr_aux_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers["aux_critic_optimizer"] = self.aux_critic_optimizer.state_dict()
        return optimizers

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_aux_critic = torch.compile(self.update_aux_critic, mode=mode)

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
            with torch.no_grad():
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_qs = self._model._target_aux_critic(next_obs, z, next_action)
                q_mean, q_unc, next_v = self.get_targets_uncertainty(next_qs, self.cfg.train.aux_critic_pessimism_penalty)
                target_q = aux_reward + discount * next_v
                expanded_targets = target_q.expand(num_parallel, -1, -1)

            qs = self._model._aux_critic(obs, z, action)
            aux_critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded_targets)

        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        average_gradients(self._model._aux_critic.parameters())
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_auxQ": target_q.mean().detach(),
                "auxQ1": qs.mean().detach(),
                "mean_next_auxQ": q_mean.mean().detach(),
                "unc_auxQ": q_unc.mean().detach(),
                "aux_critic_loss": aux_critic_loss.mean().detach(),
                "mean_aux_reward": aux_reward.mean().detach(),
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)

            qs = self._model._critic(obs, z, action)
            _, _, q = self.get_targets_uncertainty(qs, self.cfg.train.actor_pessimism_penalty)

            qs_aux = self._model._aux_critic(obs, z, action)
            _, _, q_aux = self.get_targets_uncertainty(qs_aux, self.cfg.train.actor_pessimism_penalty)

            disc_reward = self._model._discriminator.compute_reward(obs=obs, z=z)
            weight = q.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            disc_actor_weight = self.cfg.train.reg_coeff_disc * weight
            aux_actor_weight = self.cfg.train.reg_coeff_aux * weight
            actor_loss = (
                -q.mean()
                - disc_actor_weight * disc_reward.mean()
                - aux_actor_weight * q_aux.mean()
            )

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        average_gradients(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "Q_actor": q.mean().detach(),
            "Q_aux": q_aux.mean().detach(),
            "disc_reward_actor": disc_reward.mean().detach(),
            "weight/disc_actor": disc_actor_weight.detach(),
            "weight/aux_actor": aux_actor_weight.detach(),
            "weight/scale_q": weight.detach() if isinstance(weight, torch.Tensor) else torch.tensor(weight, device=self.device),
            "cfg/reg_coeff_disc": torch.tensor(self.cfg.train.reg_coeff_disc, device=self.device),
            "cfg/reg_coeff_aux": torch.tensor(self.cfg.train.reg_coeff_aux, device=self.device),
            "cfg/disc_reward_coef": torch.tensor(self.cfg.train.disc_reward_coef, device=self.device),
        }

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        pre = self._maybe_gcr_pretrain_only(replay_buffer, step)
        if pre is not None:
            pre.update(
                {
                    "disc_loss": torch.tensor(0.0, device=self.device),
                    "disc_expert_loss": torch.tensor(0.0, device=self.device),
                    "disc_train_loss": torch.tensor(0.0, device=self.device),
                    "disc_expert_logit": torch.tensor(0.0, device=self.device),
                    "disc_train_logit": torch.tensor(0.0, device=self.device),
                    "mean_disc_reward": torch.tensor(0.0, device=self.device),
                    "aux_critic_loss": torch.tensor(0.0, device=self.device),
                    "mean_aux_reward": torch.tensor(0.0, device=self.device),
                    "mean_total_reward": torch.tensor(0.0, device=self.device),
                }
            )
            return pre

        self._maybe_freeze_goal_encoder(step)

        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs, train_action, train_next_obs = (
            tree_map(lambda x: x.to(self.device), train_batch["observation"]),
            train_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"]),
        )
        train_next_obs_raw = train_next_obs
        reward = train_batch["reward"].to(self.device)
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)
        expert_obs, expert_next_obs = (
            tree_map(lambda x: x.to(self.device), expert_batch["observation"]),
            tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"]),
        )

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs, train_next_obs = (
                self._model._obs_normalizer(train_obs),
                self._model._obs_normalizer(train_next_obs),
            )
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        gcr_metrics = None
        if self.cfg.train.gcr_contrastive_during_rl and step >= self.cfg.train.gcr_pretrain_env_steps:
            gcr_metrics = self.update_gcr_contrastive(train_next_obs_raw)

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device)

        grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=grad_penalty,
        )
        if gcr_metrics is not None:
            metrics.update(gcr_metrics)

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        disc_reward = self._model._discriminator.compute_reward(obs=train_obs, z=train_z)
        total_reward = reward + self.cfg.train.disc_reward_coef * disc_reward

        aux_reward = torch.zeros((self.cfg.train.batch_size, 1), device=self.device, dtype=torch.float32)
        for aux_reward_name in self.cfg.aux_rewards:
            metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
            aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name].to(self.device)
        aux_reward = self._model._aux_reward_normalizer(aux_reward)

        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
                reward=total_reward,
            )
        )
        metrics.update(
            self.update_aux_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                aux_reward=aux_reward,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        metrics.update(self.update_actor(obs=train_obs, z=train_z, clip_grad_norm=clip_grad_norm))
        metrics["mean_env_reward"] = reward.mean().detach()
        metrics["mean_disc_reward"] = disc_reward.mean().detach()
        metrics["mean_total_reward"] = total_reward.mean().detach()
        metrics["weight/disc_reward_coef"] = torch.tensor(self.cfg.train.disc_reward_coef, device=self.device)
        metrics["z_norm/train"] = torch.norm(train_z, dim=-1).mean().detach()
        metrics["z_norm/expert"] = torch.norm(expert_z, dim=-1).mean().detach()
        metrics["z_norm/mixed"] = torch.norm(z, dim=-1).mean().detach()
        metrics["cfg/train_goal_ratio"] = torch.tensor(self.cfg.train.train_goal_ratio, device=self.device)
        metrics["cfg/expert_asm_ratio"] = torch.tensor(self.cfg.train.expert_asm_ratio, device=self.device)
        metrics["cfg/relabel_ratio"] = torch.tensor(
            -1.0 if self.cfg.train.relabel_ratio is None else float(self.cfg.train.relabel_ratio), device=self.device
        )

        with torch.no_grad():
            _soft_update_params(
                self._aux_critic_map_paramlist,
                self._aux_target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )
        return metrics
