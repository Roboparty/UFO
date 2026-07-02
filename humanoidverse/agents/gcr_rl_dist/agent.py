# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Literal

import torch
import torch.nn.functional as F
from torch import autograd
from torch.amp import autocast

from ..base import BaseConfig
from ..nn_models import eval_mode
from ..pytree_utils import tree_get_batch_size
from ...distributed import average_gradients
from ..gcr_rl.agent import GcrRlAgent, GcrRlAgentTrainConfig
from .model import GcrRlDistModel, GcrRlDistModelConfig


class GcrRlDistAgentTrainConfig(GcrRlAgentTrainConfig):
    lr_discriminator: float = 1e-5
    weight_decay_discriminator: float = 0.0
    grad_penalty_discriminator: float = 10.0
    disc_reward_coef: float = 1.0
    reg_coeff_disc: float = 0.05
    scale_reg: bool = True


class GcrRlDistAgentConfig(BaseConfig):
    name: Literal["GcrRlDistAgent"] = "GcrRlDistAgent"
    model: GcrRlDistModelConfig = GcrRlDistModelConfig()
    train: GcrRlDistAgentTrainConfig = GcrRlDistAgentTrainConfig()
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim):
        return GcrRlDistAgent(obs_space, action_dim, self)

    @property
    def object_class(self):
        return GcrRlDistAgent


class GcrRlDistAgent(GcrRlAgent):
    config_class = GcrRlDistAgentConfig

    def __init__(self, obs_space, action_dim, cfg: GcrRlDistAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        seq_length = cfg.model.seq_length
        batch_size = cfg.train.batch_size
        assert (batch_size / seq_length) == (batch_size // seq_length), "Batch size should be divisible by seq_length"
        self._model: GcrRlDistModel = self.cfg.model.build(obs_space, action_dim)
        self.setup_training()
        self.setup_compile()
        self._model.to(self.device)
        self.env_idx_with_expert_rollout = None

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
        return optimizers

    def setup_training(self) -> None:
        super().setup_training()
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator.parameters(),
            lr=self.cfg.train.lr_discriminator,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay_discriminator,
        )

    @torch.compiler.disable
    def gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor | dict[str, torch.Tensor],
        real_z: torch.Tensor,
        fake_obs: torch.Tensor | dict[str, torch.Tensor],
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = tree_get_batch_size(real_obs)
        alpha = torch.rand(batch_size, 1, device=real_z.device)

        interpolated_obs = {}
        interpolated_obs_list = []
        if isinstance(real_obs, torch.Tensor):
            interpolated_obs = (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True)
            interpolated_obs_list.append(interpolated_obs)
        else:
            for key in real_obs.keys():
                real_obs_tensor = real_obs[key]
                fake_obs_tensor = fake_obs[key]
                if isinstance(real_obs_tensor, torch.Tensor):
                    interpolated_obs[key] = (alpha * real_obs_tensor + (1 - alpha) * fake_obs_tensor).requires_grad_(True)
                    interpolated_obs_list.append(interpolated_obs[key])
                else:
                    raise ValueError(f"Unsupported type for key {key}: {type(real_obs_tensor)}")

        interpolated_z = alpha * real_z + (1 - alpha) * fake_z
        interpolated_z = interpolated_z.requires_grad_(True)
        d_interpolates = self._model._discriminator.compute_logits(interpolated_obs, interpolated_z)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolated_obs_list + [interpolated_z],
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )
        gradients = [g for g in gradients if g is not None]
        cat_gradients = torch.cat(gradients, dim=1)
        return ((cat_gradients.norm(2, dim=1) - 1) ** 2).mean()

    def update_discriminator(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        train_obs: torch.Tensor | dict[str, torch.Tensor],
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
            unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)
            expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
            unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
            loss = torch.mean(expert_loss + unlabeled_loss)
            if grad_penalty is not None:
                wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
                loss += grad_penalty * wgan_gp

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        average_gradients(self._model._discriminator.parameters())
        self.discriminator_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean().detach(),
                "disc_train_loss": unlabeled_loss.detach().mean().detach(),
                "disc_expert_logit": expert_logits.mean().detach(),
                "disc_train_logit": unlabeled_logits.mean().detach(),
            }
            if grad_penalty is not None:
                output_metrics["disc_wgan_gp_loss"] = wgan_gp.detach()
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

            disc_reward = self._model._discriminator.compute_reward(obs=obs, z=z)
            weight = q.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            disc_actor_weight = self.cfg.train.reg_coeff_disc * weight
            actor_loss = -q.mean() - disc_actor_weight * disc_reward.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        average_gradients(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "Q_actor": q.mean().detach(),
            "disc_reward_actor": disc_reward.mean().detach(),
            "weight/disc_actor": disc_actor_weight.detach(),
            "weight/scale_q": weight.detach() if isinstance(weight, torch.Tensor) else torch.tensor(weight, device=self.device),
            "cfg/reg_coeff_disc": torch.tensor(self.cfg.train.reg_coeff_disc, device=self.device),
            "cfg/disc_reward_coef": torch.tensor(self.cfg.train.disc_reward_coef, device=self.device),
        }

    def update(self, replay_buffer: dict, step: int) -> Dict[str, torch.Tensor]:
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
                    "mean_total_reward": torch.tensor(0.0, device=self.device),
                }
            )
            return pre

        self._maybe_freeze_goal_encoder(step)

        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs = {k: v.to(self.device) for k, v in train_batch["observation"].items()}
        train_action = train_batch["action"].to(self.device)
        train_next_obs = {k: v.to(self.device) for k, v in train_batch["next"]["observation"].items()}
        train_next_obs_raw = train_next_obs
        reward = train_batch["reward"].to(self.device)
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)

        expert_obs = {k: v.to(self.device) for k, v in expert_batch["observation"].items()}
        expert_next_obs = {k: v.to(self.device) for k, v in expert_batch["next"]["observation"].items()}

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs = self._model._obs_normalizer(train_obs)
            train_next_obs = self._model._obs_normalizer(train_next_obs)
            expert_obs = self._model._obs_normalizer(expert_obs)
            expert_next_obs = self._model._obs_normalizer(expert_next_obs)

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

        z_mixed = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z_mixed)
        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z_mixed, train_z)

        disc_reward = self._model._discriminator.compute_reward(obs=train_obs, z=train_z)
        total_reward = reward + self.cfg.train.disc_reward_coef * disc_reward

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
        metrics.update(self.update_actor(obs=train_obs, z=train_z, clip_grad_norm=clip_grad_norm))
        metrics["mean_env_reward"] = reward.mean().detach()
        metrics["mean_disc_reward"] = disc_reward.mean().detach()
        metrics["mean_total_reward"] = total_reward.mean().detach()
        metrics["weight/disc_reward_coef"] = torch.tensor(self.cfg.train.disc_reward_coef, device=self.device)
        metrics["z_norm/train"] = torch.norm(train_z, dim=-1).mean().detach()
        metrics["z_norm/expert"] = torch.norm(expert_z, dim=-1).mean().detach()
        metrics["z_norm/mixed"] = torch.norm(z_mixed, dim=-1).mean().detach()
        metrics["cfg/train_goal_ratio"] = torch.tensor(self.cfg.train.train_goal_ratio, device=self.device)
        metrics["cfg/expert_asm_ratio"] = torch.tensor(self.cfg.train.expert_asm_ratio, device=self.device)
        metrics["cfg/relabel_ratio"] = torch.tensor(
            -1.0 if self.cfg.train.relabel_ratio is None else float(self.cfg.train.relabel_ratio), device=self.device
        )
        return metrics
