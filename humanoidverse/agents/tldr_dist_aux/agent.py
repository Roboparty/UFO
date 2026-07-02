# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""TLDR + discriminator + auxiliary critic."""

import math
from typing import Dict, Literal

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..gcr_rl_dist_aux.agent import GcrRlDistAuxAgent, GcrRlDistAuxAgentTrainConfig
from ..gcr_rl_dist_aux.model import GcrRlDistAuxModelConfig
from ..nn_models import eval_mode
from ...distributed import average_gradients


def _roll_obs(obs: torch.Tensor | dict[str, torch.Tensor], shifts: int = 1, dim: int = 0):
    if isinstance(obs, torch.Tensor):
        return torch.roll(obs, shifts=shifts, dims=dim)
    return tree_map(lambda x: torch.roll(x, shifts=shifts, dims=dim), obs)


class TldrDistAuxAgentTrainConfig(GcrRlDistAuxAgentTrainConfig):
    use_tldr_pretrain: bool = False
    tldr_pretrain_env_steps: int = 0
    dual_reg: bool = True
    dual_lam_init: float = 3000.0
    lr_dual_lam: float = 1e-3
    dual_slack: float = 1.0
    tldr_softplus_scale: float = 500.0
    tldr_softplus_beta: float = 0.01
    tldr_te_during_rl: bool = False
    tldr_reward_scale: float = 1.0
    relabel_ratio: float | None = 0.8
    goal_encoder_lr_schedule: Literal["none", "linear", "cosine"] = "none"
    goal_encoder_lr_schedule_steps: int = 0
    goal_encoder_lr_min: float = 1e-6


class TldrDistAuxAgentConfig(BaseConfig):
    name: Literal["TldrDistAuxAgent"] = "TldrDistAuxAgent"
    model: GcrRlDistAuxModelConfig = GcrRlDistAuxModelConfig()
    train: TldrDistAuxAgentTrainConfig = TldrDistAuxAgentTrainConfig()
    aux_rewards: list[str] = []
    aux_rewards_scaling: dict[str, float] = {}
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim: int) -> "TldrDistAuxAgent":
        return TldrDistAuxAgent(obs_space=obs_space, action_dim=action_dim, cfg=self)

    @property
    def object_class(self):
        return TldrDistAuxAgent


class TldrDistAuxAgent(GcrRlDistAuxAgent):
    config_class = TldrDistAuxAgentConfig

    def __init__(self, obs_space, action_dim: int, cfg: TldrDistAuxAgentConfig):
        super().__init__(obs_space=obs_space, action_dim=action_dim, cfg=cfg)
        self._base_goal_encoder_lr = float(cfg.train.lr_goal_encoder)

    def setup_training(self) -> None:
        super().setup_training()

        t = self.cfg.train
        dual_lam_log = torch.tensor(math.log(max(t.dual_lam_init, 1e-12)), dtype=torch.float32, device=self.device)
        self.dual_lam_log_param = torch.nn.Parameter(dual_lam_log)

        self.dual_lam_optimizer = torch.optim.Adam(
            [self.dual_lam_log_param],
            lr=t.lr_dual_lam,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=0.0,
        )

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers["dual_lam_optimizer"] = self.dual_lam_optimizer.state_dict()
        return optimizers

    def _apply_goal_encoder_lr_schedule(self, step: int) -> None:
        t = self.cfg.train
        if t.goal_encoder_lr_schedule == "none":
            return
        total_steps = int(t.goal_encoder_lr_schedule_steps)
        if total_steps <= 0:
            return

        progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
        min_lr = float(t.goal_encoder_lr_min)
        base_lr = self._base_goal_encoder_lr
        if t.goal_encoder_lr_schedule == "linear":
            new_lr = min_lr + (base_lr - min_lr) * (1.0 - progress)
        elif t.goal_encoder_lr_schedule == "cosine":
            new_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown goal_encoder_lr_schedule={t.goal_encoder_lr_schedule}")

        for group in self.goal_encoder_optimizer.param_groups:
            group["lr"] = float(new_lr)

    def _maybe_freeze_goal_encoder(self, step: int) -> None:
        if not self.cfg.train.freeze_goal_encoder_after_pretrain:
            return
        if not self.cfg.train.use_tldr_pretrain:
            return
        if self._goal_encoder_frozen:
            return
        if step < self.cfg.train.tldr_pretrain_env_steps:
            return
        for p in self._model._goal_encoder.parameters():
            p.requires_grad_(False)
        self._goal_encoder_frozen = True

    def _maybe_tldr_pretrain_only(self, replay_buffer: dict, step: int) -> Dict[str, torch.Tensor] | None:
        if not self.cfg.train.use_tldr_pretrain:
            return None
        if step >= self.cfg.train.tldr_pretrain_env_steps:
            return None
        if len(replay_buffer["train"]) == 0:
            return {"tldr_te_loss": torch.tensor(0.0, device=self.device)}

        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        train_obs_raw = tree_map(lambda x: x.to(self.device), train_batch["observation"])
        train_next_obs_raw = tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"])

        with torch.no_grad():
            _ = self._model._obs_normalizer(train_obs_raw)
            _ = self._model._obs_normalizer(train_next_obs_raw)

        torch.compiler.cudagraph_mark_step_begin()
        return self.update_tldr_te(train_obs_raw=train_obs_raw, train_next_obs_raw=train_next_obs_raw)

    def update_tldr_te(self, train_obs_raw: torch.Tensor | dict[str, torch.Tensor], train_next_obs_raw) -> Dict[str, torch.Tensor]:
        t = self.cfg.train
        train_goals_raw = _roll_obs(train_next_obs_raw, shifts=1, dim=0)

        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            phi_x = self._model.encode_goal(train_obs_raw)
            phi_y = self._model.encode_goal(train_next_obs_raw)
            phi_g = self._model.encode_goal(train_goals_raw)

        phi_x = phi_x.float()
        phi_y = phi_y.float()
        phi_g = phi_g.float()

        squared_dist = ((phi_x - phi_g) ** 2).sum(dim=-1)
        dist = torch.sqrt(torch.clamp(squared_dist, min=1e-6))
        cst_penalty = 1.0 - torch.square(phi_y - phi_x).mean(dim=1)
        cst_penalty = torch.clamp(cst_penalty, max=t.dual_slack)

        dual_lam = self.dual_lam_log_param.exp()
        soft_term = -F.softplus(t.tldr_softplus_scale - dist, beta=t.tldr_softplus_beta).mean()
        dual_term = (dual_lam.detach() * cst_penalty).mean()
        te_obj = soft_term + dual_term
        loss_te = -te_obj

        self.goal_encoder_optimizer.zero_grad(set_to_none=True)
        loss_te.backward()
        average_gradients(self._model._goal_encoder.parameters())
        self.goal_encoder_optimizer.step()

        loss_dual = torch.tensor(0.0, device=self.device)
        if t.dual_reg:
            loss_dual = self.dual_lam_log_param * (cst_penalty.detach()).mean()
            self.dual_lam_optimizer.zero_grad(set_to_none=True)
            loss_dual.backward()
            average_gradients([self.dual_lam_log_param])
            self.dual_lam_optimizer.step()

        return {
            "tldr_te_loss": loss_te.detach(),
            "tldr_te_obj_mean": te_obj.detach(),
            "tldr_dist_mean": dist.mean().detach(),
            "tldr_cst_penalty_mean": cst_penalty.mean().detach(),
            "tldr_dual_lam": dual_lam.detach(),
            "tldr_dual_loss": loss_dual.detach(),
            "tldr_soft_term": soft_term.detach(),
        }

    @torch.no_grad()
    def sample_mixed_z(
        self,
        train_goal: torch.Tensor | dict[str, torch.Tensor],
        expert_encodings: torch.Tensor,
        step: int | None = None,
    ) -> torch.Tensor:
        del step
        t = self.cfg.train
        b = self.cfg.train.batch_size

        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            z = self._model.sample_z(b, device=self._model.device)
            p_goal = float(t.train_goal_ratio)
            p_expert = float(t.expert_asm_ratio)
            p_random = 1.0 - p_goal - p_expert
            if p_goal < 0 or p_expert < 0 or p_random < 0:
                raise ValueError(
                    "Invalid z-mix probabilities: "
                    f"train_goal_ratio={p_goal}, expert_asm_ratio={p_expert}, random_ratio={p_random}. "
                    "Require all >= 0 and train_goal_ratio + expert_asm_ratio <= 1."
                )
            prob = torch.tensor([p_goal, p_expert, p_random], dtype=torch.float32, device=self.device)
            mix_idxs = torch.multinomial(prob, num_samples=b, replacement=True).reshape(-1, 1)

            rolled_goal = _roll_obs(train_goal, shifts=1, dim=0)
            goals = self._model.encode_goal(rolled_goal)
            z = torch.where(mix_idxs == 0, goals, z)

            if p_expert > 0:
                expert_roll = torch.roll(expert_encodings, shifts=1, dims=0)
                z = torch.where(mix_idxs == 1, expert_roll, z)

        return z

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        self._apply_goal_encoder_lr_schedule(step)
        pre = self._maybe_tldr_pretrain_only(replay_buffer, step)
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

        train_obs_raw = tree_map(lambda x: x.to(self.device), train_batch["observation"])
        train_action = train_batch["action"].to(self.device)
        train_next_obs_raw = tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"])
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)

        expert_obs_raw = tree_map(lambda x: x.to(self.device), expert_batch["observation"])
        expert_next_obs_raw = tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"])

        self._model._obs_normalizer(train_obs_raw)
        self._model._obs_normalizer(train_next_obs_raw)
        self._model._obs_normalizer(expert_obs_raw)
        self._model._obs_normalizer(expert_next_obs_raw)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs = self._model._obs_normalizer(train_obs_raw)
            train_next_obs = self._model._obs_normalizer(train_next_obs_raw)
            expert_obs = self._model._obs_normalizer(expert_obs_raw)
            expert_next_obs = self._model._obs_normalizer(expert_next_obs_raw)

        tldr_te_metrics = None
        if self.cfg.train.tldr_te_during_rl and (not self._goal_encoder_frozen) and step >= self.cfg.train.tldr_pretrain_env_steps:
            tldr_te_metrics = self.update_tldr_te(train_obs_raw=train_obs_raw, train_next_obs_raw=train_next_obs_raw)

        torch.compiler.cudagraph_mark_step_begin()

        expert_z = self.encode_expert(next_obs=expert_next_obs_raw)
        train_z = train_batch["z"].to(self.device)

        z_mixed = self.sample_mixed_z(train_goal=train_next_obs_raw, expert_encodings=expert_z, step=step).clone()
        self.z_buffer.add(z_mixed)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z_mixed, train_z)

        disc_reward = self._model._discriminator.compute_reward(obs=train_obs, z=train_z)

        with torch.no_grad():
            cur_z = self._model.project_z(self._model._goal_encoder(train_obs)).float()
            next_z = self._model.project_z(self._model._goal_encoder(train_next_obs)).float()
            goal_z = train_z.float()
            tldr_reward = torch.norm(goal_z - cur_z, dim=1) - torch.norm(goal_z - next_z, dim=1)
            if tldr_reward.dim() == 1:
                tldr_reward = tldr_reward.unsqueeze(-1)
            tldr_reward = tldr_reward * self.cfg.train.tldr_reward_scale

        total_reward = tldr_reward + self.cfg.train.disc_reward_coef * disc_reward

        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None,
        )
        if tldr_te_metrics is not None:
            metrics.update(tldr_te_metrics)

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
        metrics.update(
            self.update_actor(obs=train_obs, z=train_z, clip_grad_norm=clip_grad_norm)
        )

        with torch.no_grad():
            from ..nn_models import _soft_update_params

            _soft_update_params(
                self._critic_paramlist,
                self._target_critic_paramlist,
                self.cfg.train.critic_target_tau,
            )

        metrics["mean_disc_reward"] = disc_reward.mean().detach()
        metrics["mean_total_reward"] = total_reward.mean().detach()
        return metrics

    def maybe_update_rollout_context(
        self,
        z: torch.Tensor | None,
        step_count: torch.Tensor,
        replay_buffer: dict | None = None,
        global_env_step: int | None = None,
    ) -> torch.Tensor:
        del global_env_step
        return super().maybe_update_rollout_context(z=z, step_count=step_count, replay_buffer=replay_buffer)
