# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""GCR + goal-conditioned policy without FB networks."""

import json
import math
import pickle
from pathlib import Path
from typing import Dict, Literal, Tuple

import safetensors.torch
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..envs.utils.gym_spaces import json_to_space, space_to_json
from ..misc.zbuffer import ZBuffer
from ..nn_models import _soft_update_params, eval_mode, weight_init
from ...distributed import average_gradients
from .model import GcrRlModel, GcrRlModelConfig


def _poincare_pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, radius: float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    r2 = float(radius) ** 2
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    diff2 = torch.clamp(x2 + y2.transpose(0, 1) - 2.0 * torch.matmul(x, y.transpose(0, 1)), min=0.0)
    denom = torch.clamp((r2 - x2) * (r2 - y2).transpose(0, 1), min=eps)
    acosh_arg = 1.0 + 2.0 * diff2 / denom
    dist = float(radius) * torch.acosh(torch.clamp(acosh_arg, min=1.0 + eps))
    return dist


def _nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float,
    z_geometry: str = "hypersphere",
    poincare_radius: float = 1.0,
    poincare_eps: float = 1e-6,
) -> torch.Tensor:
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    if z_geometry == "poincare_ball":
        sim = -_poincare_pairwise_distance(z, z, radius=poincare_radius, eps=poincare_eps) / temperature
    else:
        z = F.normalize(z, dim=-1, eps=1e-8)
        sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    labels = torch.cat([torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)], dim=0)
    return F.cross_entropy(sim, labels)


class GcrRlAgentTrainConfig(BaseConfig):
    lr_goal_encoder: float = 1e-5
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    batch_size: int = 1024
    discount: float = 0.98
    stddev_clip: float = 0.3
    actor_pessimism_penalty: float = 0.5
    critic_pessimism_penalty: float = 0.5
    critic_target_tau: float = 0.005
    train_goal_ratio: float = 0.2
    expert_asm_ratio: float = 0.6
    relabel_ratio: float | None = 0.8
    use_mix_rollout: bool = True
    update_z_every_step: int = 100
    z_buffer_size: int = 8192
    rollout_expert_trajectories: bool = True
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.5
    use_gcr_pretrain: bool = False
    gcr_pretrain_env_steps: int = 0
    gcr_temperature: float = 0.07
    gcr_augment_noise_std: float = 0.02
    gcr_loss_coef: float = 10.0
    gcr_contrastive_during_rl: bool = False
    freeze_goal_encoder_after_pretrain: bool = True


class GcrRlAgentConfig(BaseConfig):
    name: Literal["GcrRlAgent"] = "GcrRlAgent"
    model: GcrRlModelConfig = GcrRlModelConfig()
    train: GcrRlAgentTrainConfig = GcrRlAgentTrainConfig()
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim):
        return GcrRlAgent(obs_space, action_dim, self)

    @property
    def object_class(self):
        return GcrRlAgent


class GcrRlAgent:
    config_class = GcrRlAgentConfig

    def __init__(self, obs_space, action_dim, cfg: GcrRlAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        seq_length = cfg.model.seq_length
        batch_size = cfg.train.batch_size
        assert (batch_size / seq_length) == (batch_size // seq_length), "Batch size should be divisible by seq_length"

        self._model: GcrRlModel = self.cfg.model.build(obs_space, action_dim)
        self.setup_training()
        self.setup_compile()
        self._model.to(self.device)
        self.env_idx_with_expert_rollout = None

    @classmethod
    def supported_evaluations(cls):
        return ["reward", "tracking"]

    @property
    def device(self):
        return self._model.device

    @property
    def optimizer_dict(self):
        return {
            "goal_encoder_optimizer": self.goal_encoder_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()

        t = self.cfg.train
        self.goal_encoder_optimizer = torch.optim.Adam(
            self._model._goal_encoder.parameters(),
            lr=t.lr_goal_encoder,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=t.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=t.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=t.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self._model._critic.parameters(),
            lr=t.lr_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=t.weight_decay,
        )

        self._critic_paramlist = tuple(x for x in self._model._critic.parameters())
        self._target_critic_paramlist = tuple(x for x in self._model._target_critic.parameters())
        self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self._model.device)
        self._goal_encoder_frozen = False

    def setup_compile(self):
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_critic = torch.compile(self.update_critic, mode=mode)
            self.update_actor = torch.compile(self.update_actor, mode=mode)
            self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)
        preds_uns2 = preds.unsqueeze(dim=dim + 1)
        preds_diffs = torch.abs(preds_uns - preds_uns2)
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = preds_diffs.sum(dim=(dim, dim + 1)) / num_parallel_scaling
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    def _augment_goal_obs(self, goal_obs: torch.Tensor | dict[str, torch.Tensor], noise_std: float):
        def _noise(x: torch.Tensor) -> torch.Tensor:
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                return x + noise_std * torch.randn_like(x)
            return x

        return tree_map(_noise, goal_obs)

    def _maybe_freeze_goal_encoder(self, step: int) -> None:
        if not self.cfg.train.freeze_goal_encoder_after_pretrain:
            return
        if not self.cfg.train.use_gcr_pretrain:
            return
        if self._goal_encoder_frozen:
            return
        if step < self.cfg.train.gcr_pretrain_env_steps:
            return
        for p in self._model._goal_encoder.parameters():
            p.requires_grad_(False)
        self._goal_encoder_frozen = True

    def update_gcr_contrastive(self, goal_obs: torch.Tensor | dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        aug = self._augment_goal_obs(goal_obs, self.cfg.train.gcr_augment_noise_std)
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            z1 = self._model.encode_goal(goal_obs)
            z2 = self._model.encode_goal(aug)
            raw_loss = _nt_xent_loss(
                z1,
                z2,
                self.cfg.train.gcr_temperature,
                z_geometry=self.cfg.model.archi.z_geometry,
                poincare_radius=self.cfg.model.archi.poincare_radius,
                poincare_eps=self.cfg.model.archi.poincare_eps,
            )
            loss = self.cfg.train.gcr_loss_coef * raw_loss
        self.goal_encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        average_gradients(self._model._goal_encoder.parameters())
        self.goal_encoder_optimizer.step()
        metrics = {
            "gcr_contrastive_loss_raw": raw_loss.detach(),
            "gcr_contrastive_loss": loss.detach(),
        }
        if self.cfg.model.archi.z_geometry == "poincare_ball":
            pos_dist = _poincare_pairwise_distance(
                z1,
                z2,
                radius=self.cfg.model.archi.poincare_radius,
                eps=self.cfg.model.archi.poincare_eps,
            )
            metrics["gcr_poincare_pos_dist"] = torch.diag(pos_dist).mean().detach()
        else:
            metrics["gcr_cos_pos"] = F.cosine_similarity(z1, z2, dim=-1).mean().detach()
        return metrics

    def _maybe_gcr_pretrain_only(self, replay_buffer: dict, step: int) -> Dict[str, torch.Tensor] | None:
        if not self.cfg.train.use_gcr_pretrain:
            return None
        if step >= self.cfg.train.gcr_pretrain_env_steps:
            return None
        if len(replay_buffer["train"]) == 0:
            return {"gcr_contrastive_loss": torch.tensor(0.0, device=self.device)}
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        train_next_obs = tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"])
        with torch.no_grad():
            _ = self._model._obs_normalizer(train_next_obs)
        torch.compiler.cudagraph_mark_step_begin()
        return self.update_gcr_contrastive(train_next_obs)

    @torch.no_grad()
    def sample_mixed_z(
        self,
        train_goal: torch.Tensor | dict[str, torch.Tensor],
        expert_encodings: torch.Tensor,
    ) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
            p_goal = self.cfg.train.train_goal_ratio
            p_expert_asm = self.cfg.train.expert_asm_ratio
            prob = torch.tensor(
                [p_goal, p_expert_asm, 1 - p_goal - p_expert_asm],
                dtype=torch.float32,
                device=self.device,
            )
            mix_idxs = torch.multinomial(prob, num_samples=self.cfg.train.batch_size, replacement=True).reshape(-1, 1)

            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            train_goal = tree_map(lambda x: x[perm], train_goal)
            goals = self._model.encode_goal(train_goal)
            z = torch.where(mix_idxs == 0, goals, z)

            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

        return z

    @torch.no_grad()
    def encode_expert(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            b_expert = self._model._goal_encoder(
                self._model._normalize(next_obs)  # type: ignore[arg-type]
            ).detach()
            b_expert = b_expert.view(
                self.cfg.train.batch_size // self.cfg.model.seq_length,
                self.cfg.model.seq_length,
                b_expert.shape[-1],
            )
            z_expert = b_expert.mean(dim=1)
            z_expert = self._model.project_z(z_expert)
            z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)
        return z_expert

    @torch.no_grad()
    def _sample_tracking_z(self, replay_buffer, batch_dim: int, traj_length: int) -> torch.Tensor:
        batch = replay_buffer["expert_slicer"].sample(batch_dim * traj_length, seq_length=traj_length)
        z = self._model.goal_inference(batch["next"]["observation"])
        z = z.view(batch_dim, traj_length, z.shape[-1])
        for step in range(traj_length):
            end_idx = min(step + self.cfg.model.seq_length, traj_length)
            z[:, step] = z[:, step:end_idx].mean(dim=1)
        return self._model.project_z(z)

    def maybe_update_rollout_context(
        self, z: torch.Tensor | None, step_count: torch.Tensor, replay_buffer: dict | None = None
    ) -> torch.Tensor:
        if z is not None:
            mask_reset_z = step_count % self.cfg.train.update_z_every_step == 0
            if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0], device=self._model.device)
            else:
                new_z = self._model.sample_z(z.shape[0], device=self._model.device)
            z = torch.where(mask_reset_z, new_z, z.to(self._model.device))
            if self.cfg.train.rollout_expert_trajectories and replay_buffer is not None:
                idxs = step_count % self.cfg.train.rollout_expert_trajectories_length
                if torch.any(idxs == 0):
                    n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage * step_count.shape[0])
                    self.env_idx_with_expert_rollout = torch.randint(0, step_count.shape[0], size=(n_elem,), device=self._model.device)
                    self.tracking_z = self._sample_tracking_z(replay_buffer, n_elem, self.cfg.train.rollout_expert_trajectories_length)
                if self.env_idx_with_expert_rollout is not None:
                    mod_time = idxs[self.env_idx_with_expert_rollout].ravel()
                    z[self.env_idx_with_expert_rollout] = self.tracking_z[
                        torch.arange(len(self.env_idx_with_expert_rollout), device=self._model.device), mod_time
                    ]
        else:
            z = self._model.sample_z(step_count.shape[0], device=self._model.device)
            if self.cfg.train.rollout_expert_trajectories and replay_buffer is not None:
                n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage * step_count.shape[0])
                self.env_idx_with_expert_rollout = torch.randint(0, step_count.shape[0], size=(n_elem,), device=self._model.device)
                self.tracking_z = self._sample_tracking_z(replay_buffer, n_elem, self.cfg.train.rollout_expert_trajectories_length)
                z[self.env_idx_with_expert_rollout] = self.tracking_z[:, 0]
        return z

    def update_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        reward: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            with torch.no_grad():
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_qs = self._model._target_critic(next_obs, z, next_action)
                q_mean, q_unc, next_v = self.get_targets_uncertainty(next_qs, self.cfg.train.critic_pessimism_penalty)
                target_q = reward + discount * next_v
                expanded_targets = target_q.expand(num_parallel, -1, -1)

            qs = self._model._critic(obs, z, action)
            critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded_targets)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        average_gradients(self._model._critic.parameters())
        self.critic_optimizer.step()

        with torch.no_grad():
            return {
                "target_Q": target_q.mean().detach(),
                "Q1": qs.mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_next_Q": q_mean.mean().detach(),
                "unc_Q": q_unc.mean().detach(),
                "mean_reward_train": reward.mean().detach(),
            }

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
            actor_loss = -q.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        average_gradients(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "Q_actor": q.mean().detach()}

    def update(self, replay_buffer: dict, step: int) -> Dict[str, torch.Tensor]:
        pre = self._maybe_gcr_pretrain_only(replay_buffer, step)
        if pre is not None:
            return pre

        self._maybe_freeze_goal_encoder(step)

        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs = tree_map(lambda x: x.to(self.device), train_batch["observation"])
        train_action = train_batch["action"].to(self.device)
        train_next_obs = tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"])
        reward = train_batch["reward"].to(self.device)
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)

        expert_obs = tree_map(lambda x: x.to(self.device), expert_batch["observation"])
        expert_next_obs = tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"])

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs = self._model._obs_normalizer(train_obs)
            train_next_obs = self._model._obs_normalizer(train_next_obs)
            expert_obs = self._model._obs_normalizer(expert_obs)
            expert_next_obs = self._model._obs_normalizer(expert_next_obs)

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device)

        z_mixed = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z_mixed)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z_mixed, train_z)

        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics: Dict[str, torch.Tensor] = {}
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
                reward=reward,
            )
        )
        metrics.update(self.update_actor(obs=train_obs, z=train_z, clip_grad_norm=clip_grad_norm))

        with torch.no_grad():
            _soft_update_params(
                self._critic_paramlist,
                self._target_critic_paramlist,
                self.cfg.train.critic_target_tau,
            )

        return metrics

    @classmethod
    def load(cls, path: str, device: str | None = None):
        path = Path(path)
        with (path / "config.json").open() as f:
            loaded_config = json.load(f)
        if device is not None:
            loaded_config["model"]["device"] = device

        if (path / "init_kwargs.pkl").exists():
            with (path / "init_kwargs.pkl").open("rb") as f:
                args = pickle.load(f)
            obs_space = args["obs_space"]
            action_dim = args["action_dim"]
        else:
            with (path / "init_kwargs.json").open("r") as f:
                args = json.load(f)
            obs_space = json_to_space(args["obs_space"])
            action_dim = args["action_dim"]

        config = cls.config_class(**loaded_config)
        agent = config.build(obs_space, action_dim)
        optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True, map_location=device)
        for k, v in optimizers.items():
            getattr(agent, k).load_state_dict(v)
        safetensors.torch.load_model(agent._model, str(path / "model/model.safetensors"), device=device, strict=False)
        agent._model.train()
        agent._model.requires_grad_(True)
        return agent

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        json_dump = self.cfg.model_dump()
        with (output_folder / "config.json").open("w+") as f:
            json.dump(json_dump, f, indent=4)
        torch.save(self.optimizer_dict, output_folder / "optimizers.pth")
        model_folder = output_folder / "model"
        model_folder.mkdir(exist_ok=True)
        self._model.save(output_folder=str(model_folder))
        init_kwargs = {"obs_space": space_to_json(self.obs_space), "action_dim": self.action_dim}
        with (output_folder / "init_kwargs.json").open("w") as f:
            json.dump(init_kwargs, f, indent=4)
