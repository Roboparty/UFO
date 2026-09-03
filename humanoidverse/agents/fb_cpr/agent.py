# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Literal

import torch
import torch.nn.functional as F
from torch import autograd
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..fb.agent import FBAgent, FBAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode
from ..pytree_utils import tree_get_batch_size
from .model import FBcprModel, FBcprModelConfig


class FBcprAgentTrainConfig(FBAgentTrainConfig):
    behavior_prior_enabled: bool = True
    lr_discriminator: float = 1e-4
    lr_critic: float = 1e-4
    critic_target_tau: float = 0.005
    critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 1
    scale_reg: bool = True
    # the z distribution for rollouts (when agent.use_mix_rollout=1) and for the mini-batches used in the network updates is:
    # - a fraction of 'expert_asm_ratio' zs from expert trajectory encoding
    # - a fraction of 'train_goal_ratio' zs from goal encoding (goals sampled from the train buffer)
    # - the remaining fraction from the uniform distribution
    expert_asm_ratio: float = 0
    # a fraction of 'relabel_ratio' transitions in each mini-batch are relabeled with a z sampled from the above distribution
    relabel_ratio: float | None = 1
    grad_penalty_discriminator: float = 10.0
    discriminator_loss: Literal["bce", "lsgan"] = "bce"
    discriminator_reward: Literal["log_odds", "amp"] = "log_odds"
    # Commented out as example but not useful
    # grad_penalty_obs_weight: float | None = None  # must be in (0,1)
    weight_decay_discriminator: float = 0.0


class FBcprAgentConfig(BaseConfig):
    name: Literal["FBcprAgent"] = "FBcprAgent"
    model: FBcprModelConfig = FBcprModelConfig()
    train: FBcprAgentTrainConfig = FBcprAgentTrainConfig()
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim):
        return FBcprAgent(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBcprAgent


class FBcprAgent(FBAgent):
    config_class = FBcprAgentConfig

    def __init__(self, obs_space, action_dim, cfg: FBcprAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        # make sure batch size is a multiple of seq_length
        seq_length = cfg.model.seq_length
        batch_size = cfg.train.batch_size
        assert (batch_size / seq_length) == (batch_size // seq_length), "Batch size should be divisable by seq_length"
        del seq_length, batch_size

        self._model: FBcprModel = self.cfg.model.build(obs_space, action_dim)
        # This is just to be sure? I think it should not change since build
        self._model.to(self.device)
        self.setup_training()
        self.setup_compile()
        self.env_idx_with_expert_rollout = None

    @classmethod
    def supported_evaluations(cls):
        return ["reward", "tracking"]

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers.update(
            {
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            }
        )
        return optimizers

    def setup_training(self) -> None:
        super().setup_training()

        # prepare parameter list
        self._critic_map_paramlist = tuple(x for x in self._model._critic.parameters())
        self._target_critic_map_paramlist = tuple(x for x in self._model._target_critic.parameters())

        self.critic_optimizer = torch.optim.Adam(
            self._model._critic.parameters(),
            lr=self.cfg.train.lr_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator.parameters(),
            lr=self.cfg.train.lr_discriminator,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay_discriminator,
        )

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_critic = torch.compile(self.update_critic, mode=mode)
            self.update_discriminator = torch.compile(self.update_discriminator, mode=mode)
            self.encode_expert = torch.compile(self.encode_expert, mode=mode, fullgraph=True)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_critic = CudaGraphModule(self.update_critic, warmup=5)
            self.update_discriminator = CudaGraphModule(self.update_discriminator, warmup=5)
            self.encode_expert = CudaGraphModule(self.encode_expert, warmup=5)

    @torch.no_grad()
    def sample_mixed_context_components(
        self,
        train_goal: torch.Tensor,
        expert_encodings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample z and retain the source/permutation needed by companion context."""

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

            # zs obtained by encoding train goals
            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            train_goal = tree_map(lambda x: x[perm], train_goal)
            goals = self._model._backward_map(train_goal)
            goals = self._model.project_z(goals)
            z = torch.where(mix_idxs == 0, goals, z)

            # zs obtained by encoding expert trajectories
            expert_perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            z = torch.where(mix_idxs == 1, expert_encodings[expert_perm], z)

        return z, mix_idxs, expert_perm

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor, expert_encodings: torch.Tensor, *args, **kwargs):
        z, _source_type, _expert_perm = self.sample_mixed_context_components(train_goal, expert_encodings)
        return z

    @torch.no_grad()
    def encode_expert(
        self,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
    ):
        # encode expert trajectories through B
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            B_expert = self._model._backward_map(next_obs).detach()  # batch x d
            batch_size = tree_get_batch_size(next_obs)
            if batch_size % self.cfg.model.seq_length != 0:
                raise ValueError(
                    "Expert encoding batch must be divisible by model seq_length: "
                    f"batch_size={batch_size}, seq_length={self.cfg.model.seq_length}"
                )
            B_expert = B_expert.view(
                batch_size // self.cfg.model.seq_length,
                self.cfg.model.seq_length,
                B_expert.shape[-1],
            )  # N x L x d
            z_expert = B_expert.mean(dim=1)  # N x d
            z_expert = self._model.project_z(z_expert)
            z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)  # batch x d
        return z_expert

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs, train_action, train_next_obs = (
            tree_map(lambda x: x.to(self.device), train_batch["observation"]),
            train_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"]),
        )
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

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device)

        # train the discriminator
        grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=grad_penalty,
        )

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                goal=train_next_obs,
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        metrics.update(
            self._run_actor_update(
                obs=train_obs,
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
            )
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

        return metrics

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

        # TODO does not work for nested dicts
        interpolated_obs = {}
        interpolated_obs_list = []
        if isinstance(real_obs, torch.Tensor):
            if real_obs.shape != fake_obs.shape:
                raise ValueError(f"Shape mismatch: {real_obs.shape} vs {fake_obs.shape}")
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
            # obs might contain entries that are not used by discriminator, in which case autograd.grad would complain about unused variables
            # We need to explicitely allow unused variables
            # TODO maybe remove these variables from the inputs completely instead? (we could read self._model._discriminator.filtered_space.keys, but this creates a dependency on the discriminator)
            allow_unused=True,
        )

        # Filter out None's from gradients: if any input is not used by the discriminator, autograd.grad will return None for its gradient
        gradients = [g for g in gradients if g is not None]
        cat_gradients = torch.cat(gradients, dim=1)
        gradient_penalty = ((cat_gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Example of code with grad_penalty_obs_weight
        # obs_dim = real_obs.shape[-1]
        # grad_obs = gradients[:, :obs_dim]
        # grad_z = gradients[:, obs_dim:]
        # w_obs = self.cfg.train.grad_penalty_obs_weight
        # w_z = 1 - w_obs
        # gradient_penalty_obs = ((grad_obs.norm(2, dim=1) - w_obs) ** 2).mean()
        # gradient_penalty_z = ((grad_z.norm(2, dim=1) - w_z) ** 2).mean()
        # gradient_penalty = 0.5 * (gradient_penalty_obs + gradient_penalty_z)
        return gradient_penalty

    def update_discriminator(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        train_obs: torch.Tensor | dict[str, torch.Tensor],
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            discriminator_stage = self._training_stage("discriminator")
            if discriminator_stage is None:
                expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
                unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)
            else:
                expert_logits = discriminator_stage(obs=expert_obs, z=expert_z)
                unlabeled_logits = discriminator_stage(obs=train_obs, z=train_z)
            if self.cfg.train.discriminator_loss == "bce":
                # Raw-logit binary cross entropy, retained for the original
                # UFO/MetaMotivo agents and controlled ablations.
                expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
                unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
                data_loss = torch.mean(expert_loss + unlabeled_loss)
            elif self.cfg.train.discriminator_loss == "lsgan":
                # AMP-style least-squares targets: expert=+1, policy=-1.
                expert_loss = 0.5 * (expert_logits - 1.0).square()
                unlabeled_loss = 0.5 * (unlabeled_logits + 1.0).square()
                data_loss = expert_loss.mean() + unlabeled_loss.mean()
            else:  # pragma: no cover - protected by the pydantic Literal
                raise ValueError(
                    f"Unsupported discriminator loss: {self.cfg.train.discriminator_loss!r}"
                )

            weighted_gp = torch.zeros((), device=data_loss.device, dtype=data_loss.dtype)
            loss = data_loss

            if grad_penalty is not None:
                wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
                weighted_gp = float(grad_penalty) * wgan_gp
                loss = loss + weighted_gp

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._sync_gradients_if_manual(self._model._discriminator.parameters())
        self.discriminator_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean().detach(),
                "disc_train_loss": unlabeled_loss.detach().mean().detach(),
                "disc/data_loss": data_loss.detach(),
                "disc/gp_weighted": weighted_gp.detach(),
                "disc/total_loss": loss.detach(),
                "disc/gp_fraction": (
                    weighted_gp / (data_loss.detach().abs() + weighted_gp.abs()).clamp_min(1.0e-8)
                ).detach(),
            }
            if self.cfg.train.discriminator_loss == "lsgan":
                output_metrics["disc/expert_lsgan"] = expert_loss.detach().mean()
                output_metrics["disc/policy_lsgan"] = unlabeled_loss.detach().mean()
            if grad_penalty is not None:
                output_metrics["disc_wgan_gp_loss"] = wgan_gp.detach()
                output_metrics["disc/gp_raw"] = wgan_gp.detach()
            else:
                output_metrics["disc/gp_raw"] = torch.zeros_like(loss.detach())
        return output_metrics

    def discriminator_reward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the configured behavior-prior reward.

        ``log_odds`` preserves the historical implementation.  ``amp`` uses
        the bounded least-squares reward from AMP, which is exactly in [0, 1]
        and therefore prevents discriminator-logit drift from changing the
        Q_D return scale.
        """

        logits = self._model._discriminator.compute_logits(obs=obs, z=z)
        return self.discriminator_reward_from_logits(logits)

    def discriminator_reward_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Map raw discriminator outputs to the configured scalar reward."""

        reward_mode = getattr(
            getattr(self.cfg, "train", None),
            "discriminator_reward",
            "log_odds",
        )
        if reward_mode == "log_odds":
            probability = torch.sigmoid(logits).clamp(1.0e-6, 1.0 - 1.0e-6)
            return torch.log(probability) - torch.log1p(-probability)
        if reward_mode == "amp":
            return (1.0 - 0.25 * (logits - 1.0).square()).clamp_min(0.0)
        raise ValueError(  # pragma: no cover - protected by the pydantic Literal
            f"Unsupported discriminator reward: {reward_mode!r}"
        )

    def update_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            # compute target critic
            with torch.no_grad():
                reward = self.discriminator_reward(obs=obs, z=z)
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_critic(next_obs, z, next_action)  # num_parallel x batch x 1
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.critic_pessimism_penalty)
                target_Q = reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            # compute critic loss
            critic_stage = self._training_stage("critic")
            critic = self._model._critic if critic_stage is None else critic_stage
            Qs = critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._sync_gradients_if_manual(self._model._critic.parameters())
        self.critic_optimizer.step()

        with torch.no_grad():
            online_Q_mean, online_Q_unc, _ = self.get_targets_uncertainty(Qs, 0.0)
            target_Q_abs_scale = Q_mean.abs().mean()
            output_metrics = {
                "target_Q": target_Q.mean().detach(),
                "Q1": Qs.mean().detach(),
                "mean_next_Q": Q_mean.mean().detach(),
                "unc_Q": Q_unc.mean().detach(),
                # ``unc_Q`` is target-critic disagreement, not a loss.  Keep
                # the historical key and additionally expose its scale-aware
                # form plus the individual ensemble heads.  The old ``Q1``
                # key is actually an all-head mean and is retained only for
                # dashboard compatibility.
                "qd/target_abs_scale": target_Q_abs_scale.detach(),
                "qd/relative_uncertainty": (
                    Q_unc.mean() / target_Q_abs_scale.clamp_min(1.0e-8)
                ).detach(),
                "qd/online_uncertainty": online_Q_unc.mean().detach(),
                "qd/online_abs_scale": online_Q_mean.abs().mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_disc_reward": reward.mean().detach(),
            }
            # The production configuration has two heads.  Logging them
            # separately is essential: the legacy ``Q1`` metric cannot reveal
            # head divergence because it averages the ensemble first.
            if num_parallel >= 2:
                output_metrics.update(
                    {
                        "qd/online_head0_mean": Qs[0].mean().detach(),
                        "qd/online_head1_mean": Qs[1].mean().detach(),
                        "qd/target_head0_mean": next_Qs[0].mean().detach(),
                        "qd/target_head1_mean": next_Qs[1].mean().detach(),
                    }
                )
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        actor_stage = self._training_stage("actor")
        actor = self._model._actor if actor_stage is None else actor_stage
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)

            # compute discriminator reward loss
            Qs_discriminator = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.cfg.train.actor_pessimism_penalty)  # batch

            # compute fb reward loss
            Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
            Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

            weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            actor_loss = -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight - Q_fb.mean()

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
                "Q_fb": Q_fb.mean().detach(),
            }
        return output_metrics
