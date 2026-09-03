# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import json
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import safetensors
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ...distributed import average_gradients, wrap_distributed_stage
from ..base import BaseConfig
from ..behavior_context import (
    HEADING_SOURCE_EXACT_TRACKING,
    HEADING_SOURCE_INVALID,
    align_heading_sequence,
    normalize_heading_xy,
)
from ..envs.utils.gym_spaces import json_to_space, space_to_json
from ..misc.zbuffer import ZBuffer
from ..nn_models import _soft_update_params, eval_mode, weight_init
from .model import FBModel, FBModelConfig


@dataclass
class RolloutContextState:
    """Collector-local latent and expert-tracking rollout state.

    Training may drive more than one environment collector with the same
    agent. The temporal bookkeeping therefore belongs to the collector, not
    to the shared agent/model.
    """

    z: torch.Tensor | None = None
    heading_target_xy: torch.Tensor | None = None
    heading_valid: torch.Tensor | None = None
    context_id: torch.Tensor | None = None
    source_type: torch.Tensor | None = None
    expert_env_ids: torch.Tensor | None = None
    tracking_z: torch.Tensor | None = None
    tracking_heading_target_xy: torch.Tensor | None = None
    motion_id: torch.Tensor | None = None
    tracking_motion_id: torch.Tensor | None = None
    reference_index: torch.Tensor | None = None
    tracking_reference_index: torch.Tensor | None = None
    z_encoder_version: torch.Tensor | None = None
    tracking_z_encoder_version: torch.Tensor | None = None


class _FBTrainingStage(torch.nn.Module):
    def __init__(self, forward_map: torch.nn.Module, backward_map: torch.nn.Module) -> None:
        super().__init__()
        self.forward_map = forward_map
        self.backward_map = backward_map

    def forward(self, obs, z, action, goal):
        return self.forward_map(obs, z, action), self.backward_map(goal)


class _MethodTrainingStage(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, method_name: str = "forward") -> None:
        super().__init__()
        self.wrapped_module = module
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        return getattr(self.wrapped_module, self.method_name)(*args, **kwargs)


@contextmanager
def _without_parameter_gradients(*modules: torch.nn.Module):
    parameters = tuple(parameter for module in modules for parameter in module.parameters())
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, requires_grad):
            parameter.requires_grad_(enabled)


class FBAgentTrainConfig(BaseConfig):
    lr_f: float = 1e-4
    lr_b: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    fb_target_tau: float = 0.01
    ortho_coef: float = 1.0
    train_goal_ratio: float = 0.5
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    stddev_clip: float = 0.3
    q_loss_coef: float = 0.0
    batch_size: int = 1024
    discount: float = 0.99
    use_mix_rollout: bool = False
    update_z_every_step: int = 150
    z_buffer_size: int = 10000
    rollout_expert_trajectories: bool = False
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.25


class FBAgentConfig(BaseConfig):
    name: Literal["FBAgent"] = "FBAgent"
    model: FBModelConfig
    train: FBAgentTrainConfig
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim):
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBAgent


class FBAgent:
    config_class = FBAgentConfig

    def __init__(self, obs_space, action_dim, cfg: FBAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        self.fb_target_tau = float(min(max(self.cfg.train.fb_target_tau, 0), 1))
        self._model: FBModel = self.cfg.model.build(obs_space, action_dim)
        self.setup_training()
        self.setup_compile()
        # This is just to be sure? I think it should not change since build
        self._model.to(self.device)

        self.env_idx_with_expert_rollout = None
        self._distributed_training_stages: dict[str, torch.nn.Module] = {}

    def enable_distributed_gradient_sync(self, *, bucket_cap_mb: float = 25.0) -> None:
        """Enable DDP reducer hooks separately for every optimizer stage."""
        if getattr(self, "_distributed_training_stages", None):
            return
        stages: dict[str, torch.nn.Module] = {
            "fb": _FBTrainingStage(self._model._forward_map, self._model._backward_map),
            "actor": _MethodTrainingStage(self._model._actor),
        }
        if hasattr(self._model, "_discriminator"):
            stages["discriminator"] = _MethodTrainingStage(self._model._discriminator, "compute_logits")
        if hasattr(self._model, "_critic"):
            stages["critic"] = _MethodTrainingStage(self._model._critic)
        if hasattr(self._model, "_aux_critic"):
            stages["aux_critic"] = _MethodTrainingStage(self._model._aux_critic)
        if hasattr(self._model, "_heading_critic"):
            stages["heading_critic"] = _MethodTrainingStage(self._model._heading_critic)
        self._distributed_training_stages = {
            name: wrap_distributed_stage(stage, bucket_cap_mb=bucket_cap_mb) for name, stage in stages.items()
        }

    def _training_stage(self, name: str) -> torch.nn.Module | None:
        return getattr(self, "_distributed_training_stages", {}).get(name)

    def _sync_gradients_if_manual(self, parameters) -> None:
        if not getattr(self, "_distributed_training_stages", None):
            average_gradients(parameters)

    def _run_actor_update(self, *args, **kwargs):
        if not getattr(self, "_distributed_training_stages", None):
            return self.update_actor(*args, **kwargs)
        dependencies = [self._model._forward_map]
        if hasattr(self._model, "_critic"):
            dependencies.append(self._model._critic)
        if hasattr(self._model, "_aux_critic"):
            dependencies.append(self._model._aux_critic)
        if hasattr(self._model, "_heading_critic"):
            dependencies.append(self._model._heading_critic)
        with _without_parameter_gradients(*dependencies):
            return self.update_actor(*args, **kwargs)

    @property
    def device(self):
        return self._model.device

    @property
    def optimizer_dict(self):
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
        }

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.backward_optimizer = torch.optim.Adam(
            self._model._backward_map.parameters(),
            lr=self.cfg.train.lr_b,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            self._model._forward_map.parameters(),
            lr=self.cfg.train.lr_f,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

        # prepare parameter list
        self._forward_map_paramlist = tuple(x for x in self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

        self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self._model.device)

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_fb = torch.compile(self.update_fb, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.update_actor = torch.compile(self.update_actor, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)

        print(f"cudagraphs {self.cfg.cudagraphs}")
        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_fb = CudaGraphModule(self.update_fb, warmup=5)
            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor | dict[str, torch.Tensor] | None = None, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)

            if train_goal is not None:
                perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
                train_goal = tree_map(lambda x: x[perm], train_goal)
                goals = self._model._backward_map(train_goal)
                goals = self._model.project_z(goals)
                mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) < self.cfg.train.train_goal_ratio
                z = torch.where(mask, goals, z)
        return z

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        obs, action, next_obs, terminated = (
            batch["observation"],
            batch["action"],
            batch["next"]["observation"],
            batch["next"]["terminated"],
        )
        discount = self.cfg.train.discount * ~terminated

        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs = self._model._obs_normalizer(obs), self._model._obs_normalizer(next_obs)

        torch.compiler.cudagraph_mark_step_begin()
        z = self.sample_mixed_z(train_goal=next_obs).clone()
        self.z_buffer.add(z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        torch.compiler.cudagraph_mark_step_begin()
        metrics = self.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            goal=next_obs,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )
        metrics.update(
            self._run_actor_update(
                obs=obs,
                action=action,
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.fb_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.fb_target_tau)

        return metrics

    def sample_action_from_norm_obs(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)
        return action

    def update_fb(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                # dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                # next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_action = self.sample_action_from_norm_obs(next_obs, z)
                target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # num_parallel x batch x z_dim
                target_B = self._model._target_backward_map(goal)  # batch x z_dim
                target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
                _, _, target_M = self.get_targets_uncertainty(target_Ms, self.cfg.train.fb_pessimism_penalty)  # batch x batch

            # compute FB loss
            fb_stage = self._training_stage("fb")
            if fb_stage is None:
                Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
                B = self._model._backward_map(goal)  # batch x z_dim
            else:
                Fs, B = fb_stage(obs, z, action, goal)
            Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

            diff = Ms - discount * target_M  # num_parallel x batch x batch
            fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
            fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            fb_loss = fb_offdiag + fb_diag

            # compute orthonormality loss for backward embedding
            Cov = torch.matmul(B, B.T)
            orth_loss_diag = -Cov.diag().mean()
            orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
            orth_loss = orth_loss_offdiag + orth_loss_diag
            fb_loss += self.cfg.train.ortho_coef * orth_loss

            q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
            if q_loss_coef is not None:
                with torch.no_grad():
                    next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                    _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.fb_pessimism_penalty)  # batch
                    # TODO: we disable autocast here to make sure B and cov have the same dtype (otherwise torch.linalg.solve fails)
                    with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=False):
                        cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                    # inv_cov = torch.inverse(cov)  # z_dim x z_dim
                    B_inv_conv = torch.linalg.solve(cov, B, left=False)
                    implicit_reward = (B_inv_conv * z).sum(dim=-1)  # batch
                    target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                    expanded_targets = target_Q.expand(Fs.shape[0], -1)
                Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
                q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
                fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        self._sync_gradients_if_manual((*self._model._forward_map.parameters(), *self._model._backward_map.parameters()))
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        return self.update_td3_actor(obs=obs, z=z, clip_grad_norm=clip_grad_norm)

    def update_td3_actor(
        self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, clip_grad_norm: float | None
    ) -> Dict[str, torch.Tensor]:
        actor_stage = self._training_stage("actor")
        actor = self._model._actor if actor_stage is None else actor_stage
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)
            Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
            Qs = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q = self.get_targets_uncertainty(Qs, self.cfg.train.actor_pessimism_penalty)  # batch
            actor_loss = -Q.mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._sync_gradients_if_manual(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "q": Q.mean().detach()}

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    def _sample_tracking_context(self, replay_buffer, batch_dim, traj_length):
        batch, sampled_idxs = replay_buffer["expert_slicer"].sample(
            batch_dim * traj_length,
            seq_length=traj_length,
            return_indices=True,
        )  # N*T x obs_dim
        z = self._model.backward_map(batch["next"]["observation"])  # NT x z_dim
        z = z.view(batch_dim, traj_length, z.shape[-1])  # N x T x z_dim
        for step in range(traj_length):
            end_idx = min(step + self.cfg.model.seq_length, traj_length)
            z[:, step] = z[:, step:end_idx].mean(dim=1)
        if "heading_forward_xy" not in batch:
            raise RuntimeError(
                "Exact tracking heading contexts require expert heading_forward_xy metadata; rebuild the expert buffer cache"
            )
        heading_reference_xy = normalize_heading_xy(batch["heading_forward_xy"].to(device=self.device, dtype=torch.float32)).view(
            batch_dim, traj_length, 2
        )
        if "motion_id" not in batch:
            raise RuntimeError("Exact tracking contexts require expert motion_id metadata")
        motion_id = batch["motion_id"].to(device=self.device, dtype=torch.long).view(batch_dim, traj_length, -1)[..., :1]
        reference_index = sampled_idxs[0].to(device=self.device, dtype=torch.long).view(batch_dim, traj_length, 1)
        selective_state = getattr(self, "_selective_prior_state", None)
        encoder_version = int(getattr(selective_state, "behavior_encoder_version", 0))
        encoder_version = torch.full_like(reference_index, encoder_version)
        return self._model.project_z(z), heading_reference_xy, motion_id, reference_index, encoder_version

    def advance_rollout_context(
        self,
        state: RolloutContextState,
        step_count: torch.Tensor,
        replay_buffer: dict | None = None,
        current_heading_xy: torch.Tensor | None = None,
    ) -> RolloutContextState:
        """Advance one collector without mutating agent-global rollout state.

        Expert-tracking environment identities remain collector-local and
        stable. Each such environment refreshes its own tracking trajectory at
        the trajectory boundary. This avoids one environment reset silently
        replacing every other environment's behavior context.
        """

        device = self._model.device
        counts = step_count.to(device=device, dtype=torch.long).reshape(-1)
        num_envs = counts.shape[0]
        heading_enabled = bool(getattr(getattr(self.cfg, "model", None), "heading_context_enabled", False))
        if current_heading_xy is None:
            if heading_enabled:
                raise ValueError("Heading-enabled rollout context requires current_heading_xy")
            current_heading_xy = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
            current_heading_xy[:, 0] = 1.0
        else:
            current_heading_xy = normalize_heading_xy(current_heading_xy.to(device=device, dtype=torch.float32))

        z = None if state.z is None else state.z.to(device)
        expert_env_ids = state.expert_env_ids
        tracking_z = state.tracking_z
        tracking_heading_target_xy = state.tracking_heading_target_xy
        motion_id = state.motion_id
        tracking_motion_id = state.tracking_motion_id
        reference_index = state.reference_index
        tracking_reference_index = state.tracking_reference_index
        z_encoder_version = state.z_encoder_version
        tracking_z_encoder_version = state.tracking_z_encoder_version

        if z is None:
            z = self._model.sample_z(num_envs, device=device)
            heading_target_xy = torch.zeros((num_envs, 2), device=device)
            heading_valid = torch.zeros((num_envs, 1), device=device, dtype=torch.bool)
            context_id = (
                state.context_id.to(device=device, dtype=torch.long).clone()
                if state.context_id is not None
                else torch.ones((num_envs, 1), device=device, dtype=torch.long)
            )
            if context_id.shape != (num_envs, 1):
                raise ValueError(
                    "Initial rollout context_id must have shape "
                    f"({num_envs}, 1), got {tuple(context_id.shape)}"
                )
            source_type = torch.full((num_envs, 1), HEADING_SOURCE_INVALID, device=device, dtype=torch.long)
            motion_id = torch.full((num_envs, 1), -1, device=device, dtype=torch.long)
            reference_index = torch.full((num_envs, 1), -1, device=device, dtype=torch.long)
            z_encoder_version = torch.full((num_envs, 1), -1, device=device, dtype=torch.long)
            if self.cfg.train.rollout_expert_trajectories:
                if replay_buffer is None:
                    raise ValueError("Expert rollout contexts require an expert replay buffer")
                n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage * num_envs)
                expert_env_ids = torch.randperm(num_envs, device=device)[:n_elem]
                (
                    tracking_z,
                    tracking_heading_reference_xy,
                    tracking_motion_id,
                    tracking_reference_index,
                    tracking_z_encoder_version,
                ) = self._sample_tracking_context(
                    replay_buffer,
                    n_elem,
                    self.cfg.train.rollout_expert_trajectories_length,
                )
                mod_time = counts[expert_env_ids] % self.cfg.train.rollout_expert_trajectories_length
                tracking_heading_target_xy = align_heading_sequence(
                    tracking_heading_reference_xy,
                    current_heading_xy[expert_env_ids],
                    mod_time,
                )
        else:
            heading_target_xy = (
                state.heading_target_xy.to(device).clone()
                if state.heading_target_xy is not None
                else torch.zeros((num_envs, 2), device=device)
            )
            heading_valid = (
                state.heading_valid.to(device).clone()
                if state.heading_valid is not None
                else torch.zeros((num_envs, 1), device=device, dtype=torch.bool)
            )
            context_id = (
                state.context_id.to(device).clone()
                if state.context_id is not None
                else torch.zeros((num_envs, 1), device=device, dtype=torch.long)
            )
            source_type = (
                state.source_type.to(device).clone()
                if state.source_type is not None
                else torch.full((num_envs, 1), HEADING_SOURCE_INVALID, device=device, dtype=torch.long)
            )
            motion_id = (
                state.motion_id.to(device).clone()
                if state.motion_id is not None
                else torch.full((num_envs, 1), -1, device=device, dtype=torch.long)
            )
            reference_index = (
                state.reference_index.to(device).clone()
                if state.reference_index is not None
                else torch.full((num_envs, 1), -1, device=device, dtype=torch.long)
            )
            z_encoder_version = (
                state.z_encoder_version.to(device).clone()
                if state.z_encoder_version is not None
                else torch.full((num_envs, 1), -1, device=device, dtype=torch.long)
            )

        expert_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)
        if expert_env_ids is not None:
            expert_env_ids = expert_env_ids.to(device=device, dtype=torch.long)
            expert_mask[expert_env_ids] = True

        # Non-tracking contexts follow the existing random/z-buffer rollout
        # schedule and intentionally have no invented heading semantics.
        reset_random = (counts % self.cfg.train.update_z_every_step == 0) & ~expert_mask
        if bool(reset_random.any()):
            if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(num_envs, device=device)
            else:
                new_z = self._model.sample_z(num_envs, device=device)
            z[reset_random] = new_z[reset_random]
            heading_target_xy[reset_random] = 0.0
            heading_valid[reset_random] = False
            source_type[reset_random] = HEADING_SOURCE_INVALID
            motion_id[reset_random] = -1
            reference_index[reset_random] = -1
            z_encoder_version[reset_random] = -1
            context_id[reset_random] += 1

        if self.cfg.train.rollout_expert_trajectories:
            if replay_buffer is None:
                raise ValueError("Expert rollout contexts require an expert replay buffer")
            if (
                expert_env_ids is None
                or tracking_z is None
                or tracking_heading_target_xy is None
                or tracking_motion_id is None
                or tracking_reference_index is None
                or tracking_z_encoder_version is None
            ):
                raise RuntimeError("Expert rollout context was not initialized")
            trajectory_length = self.cfg.train.rollout_expert_trajectories_length
            mod_time = counts[expert_env_ids] % trajectory_length
            refresh_rows = torch.nonzero(mod_time == 0, as_tuple=False).reshape(-1)
            # Initialization already sampled every row above. On later calls,
            # refresh only expert environments whose own context expired.
            if state.z is not None and refresh_rows.numel() > 0:
                (
                    refreshed_z,
                    refreshed_reference_xy,
                    refreshed_motion_id,
                    refreshed_reference_index,
                    refreshed_encoder_version,
                ) = self._sample_tracking_context(
                    replay_buffer,
                    int(refresh_rows.numel()),
                    trajectory_length,
                )
                refreshed_env_ids = expert_env_ids[refresh_rows]
                refreshed_targets = align_heading_sequence(
                    refreshed_reference_xy,
                    current_heading_xy[refreshed_env_ids],
                    mod_time[refresh_rows],
                )
                tracking_z = tracking_z.clone()
                tracking_heading_target_xy = tracking_heading_target_xy.clone()
                tracking_z[refresh_rows] = refreshed_z
                tracking_heading_target_xy[refresh_rows] = refreshed_targets
                tracking_motion_id = tracking_motion_id.clone()
                tracking_motion_id[refresh_rows] = refreshed_motion_id
                tracking_reference_index = tracking_reference_index.clone()
                tracking_reference_index[refresh_rows] = refreshed_reference_index
                tracking_z_encoder_version = tracking_z_encoder_version.clone()
                tracking_z_encoder_version[refresh_rows] = refreshed_encoder_version
                context_id[refreshed_env_ids] += 1

            rows = torch.arange(expert_env_ids.numel(), device=device)
            z[expert_env_ids] = tracking_z[rows, mod_time]
            heading_target_xy[expert_env_ids] = tracking_heading_target_xy[rows, mod_time]
            heading_valid[expert_env_ids] = heading_enabled
            source_type[expert_env_ids] = HEADING_SOURCE_EXACT_TRACKING
            motion_id[expert_env_ids] = tracking_motion_id[rows, mod_time]
            reference_index[expert_env_ids] = tracking_reference_index[rows, mod_time]
            z_encoder_version[expert_env_ids] = tracking_z_encoder_version[rows, mod_time]

        return RolloutContextState(
            z=z,
            heading_target_xy=heading_target_xy,
            heading_valid=heading_valid,
            context_id=context_id,
            source_type=source_type,
            expert_env_ids=expert_env_ids,
            tracking_z=tracking_z,
            tracking_heading_target_xy=tracking_heading_target_xy,
            motion_id=motion_id,
            tracking_motion_id=tracking_motion_id,
            reference_index=reference_index,
            tracking_reference_index=tracking_reference_index,
            z_encoder_version=z_encoder_version,
            tracking_z_encoder_version=tracking_z_encoder_version,
        )

    def next_rollout_heading_target(
        self,
        state: RolloutContextState,
        step_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current context's target evaluated at the resulting state."""

        if state.heading_target_xy is None or state.heading_valid is None:
            raise RuntimeError("Rollout heading context is not initialized")
        target = state.heading_target_xy.clone()
        valid = state.heading_valid.clone()
        if state.expert_env_ids is not None and state.tracking_heading_target_xy is not None and state.expert_env_ids.numel() > 0:
            ids = state.expert_env_ids.to(device=self.device, dtype=torch.long)
            counts = step_count.to(device=self.device, dtype=torch.long).reshape(-1)
            length = self.cfg.train.rollout_expert_trajectories_length
            current_index = counts[ids] % length
            next_index = torch.clamp(current_index + 1, max=length - 1)
            rows = torch.arange(ids.numel(), device=self.device)
            target[ids] = state.tracking_heading_target_xy[rows, next_index]
        return target, valid

    def next_rollout_z(
        self,
        state: RolloutContextState,
        step_count: torch.Tensor,
    ) -> torch.Tensor:
        """Return the current context's latent at the resulting state.

        Exact tracking contexts are time-varying even though their
        ``context_id`` remains constant, so Q_H must bootstrap with z_{t+1}
        rather than silently reusing z_t.
        """

        if state.z is None:
            raise RuntimeError("Rollout behavior context is not initialized")
        next_z = state.z.clone()
        if state.expert_env_ids is not None and state.tracking_z is not None and state.expert_env_ids.numel() > 0:
            ids = state.expert_env_ids.to(device=self.device, dtype=torch.long)
            counts = step_count.to(device=self.device, dtype=torch.long).reshape(-1)
            length = self.cfg.train.rollout_expert_trajectories_length
            current_index = counts[ids] % length
            next_index = torch.clamp(current_index + 1, max=length - 1)
            rows = torch.arange(ids.numel(), device=self.device)
            next_z[ids] = state.tracking_z[rows, next_index]
        return next_z

    def rollout_heading_context_continues(
        self,
        state: RolloutContextState,
        next_step_count: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """Return whether Q_H may bootstrap under the same valid context."""

        if state.heading_valid is None:
            raise RuntimeError("Rollout heading context is not initialized")
        counts = next_step_count.to(device=self.device, dtype=torch.long).reshape(-1)
        continues = state.heading_valid.reshape(-1).clone()
        expert_mask = torch.zeros_like(continues)
        if state.expert_env_ids is not None:
            ids = state.expert_env_ids.to(device=self.device, dtype=torch.long)
            expert_mask[ids] = True
            continues[ids] &= counts[ids] % self.cfg.train.rollout_expert_trajectories_length != 0
        continues[~expert_mask] &= counts[~expert_mask] % self.cfg.train.update_z_every_step != 0
        continues &= ~done.to(device=self.device, dtype=torch.bool).reshape(-1)
        return continues.unsqueeze(-1)

    def maybe_update_rollout_context(self, z: torch.Tensor | None, step_count: torch.Tensor, replay_buffer: None = None) -> torch.Tensor:
        # Backward-compatible single-collector entrypoint. New multi-stream
        # training owns RolloutContextState instances in the workspace.
        state = self.advance_rollout_context(
            RolloutContextState(
                z=z,
                heading_target_xy=getattr(self, "heading_target_xy", None),
                heading_valid=getattr(self, "heading_valid", None),
                context_id=getattr(self, "heading_context_id", None),
                source_type=getattr(self, "heading_source_type", None),
                expert_env_ids=self.env_idx_with_expert_rollout,
                tracking_z=getattr(self, "tracking_z", None),
                tracking_heading_target_xy=getattr(self, "tracking_heading_target_xy", None),
                motion_id=getattr(self, "rollout_motion_id", None),
                tracking_motion_id=getattr(self, "tracking_motion_id", None),
                reference_index=getattr(self, "rollout_reference_index", None),
                tracking_reference_index=getattr(self, "tracking_reference_index", None),
                z_encoder_version=getattr(self, "rollout_z_encoder_version", None),
                tracking_z_encoder_version=getattr(self, "tracking_z_encoder_version", None),
            ),
            step_count,
            replay_buffer,
        )
        self.env_idx_with_expert_rollout = state.expert_env_ids
        self.tracking_z = state.tracking_z
        self.heading_target_xy = state.heading_target_xy
        self.heading_valid = state.heading_valid
        self.heading_context_id = state.context_id
        self.heading_source_type = state.source_type
        self.tracking_heading_target_xy = state.tracking_heading_target_xy
        self.rollout_motion_id = state.motion_id
        self.tracking_motion_id = state.tracking_motion_id
        self.rollout_reference_index = state.reference_index
        self.tracking_reference_index = state.tracking_reference_index
        self.rollout_z_encoder_version = state.z_encoder_version
        self.tracking_z_encoder_version = state.tracking_z_encoder_version
        assert state.z is not None
        return state.z

    @classmethod
    def load(cls, path: str, device: str | None = None):
        path = Path(path)
        with (path / "config.json").open() as f:
            loaded_config = json.load(f)
        if device is not None:
            loaded_config["model"]["device"] = device

        if (path / "init_kwargs.pkl").exists():
            # Load arguments from a pickle file
            with (path / "init_kwargs.pkl").open("rb") as f:
                args = pickle.load(f)
            obs_space = args["obs_space"]
            action_dim = args["action_dim"]
        else:
            # load argeuments from a json file
            with (path / "init_kwargs.json").open("r") as f:
                args = json.load(f)
            obs_space = json_to_space(args["obs_space"])
            action_dim = args["action_dim"]

        # JSON has no tuple type, so strict construction cannot reload fields
        # such as direct_depth.proprio_keys after a checkpoint save. Keep
        # strict validation for newly-built configs and coerce JSON containers
        # only at this serialization boundary.
        config = cls.config_class.model_validate(loaded_config, strict=False)
        agent = config.build(obs_space, action_dim)
        optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True, map_location=device)
        for k, v in optimizers.items():
            getattr(agent, k).load_state_dict(v)
        model_state = safetensors.torch.load_file(path / "model/model.safetensors", device=device or config.model.device)
        agent._model.load_state_dict(model_state, strict=False)
        del model_state
        extra_state_path = path / "training_state.pth"
        if extra_state_path.exists() and hasattr(agent, "load_extra_training_state_dict"):
            extra_state = torch.load(extra_state_path, weights_only=True, map_location=device)
            agent.load_extra_training_state_dict(extra_state)
        agent._model.train()
        agent._model.requires_grad_(True)
        return agent

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        json_dump = self.cfg.model_dump()
        with (output_folder / "config.json").open("w+") as f:
            json.dump(json_dump, f, indent=4)
        # save optimizer
        torch.save(
            self.optimizer_dict,
            output_folder / "optimizers.pth",
        )
        if hasattr(self, "extra_training_state_dict"):
            torch.save(self.extra_training_state_dict(), output_folder / "training_state.pth")
        # save model
        model_folder = output_folder / "model"
        model_folder.mkdir(exist_ok=True)
        self._model.save(output_folder=str(model_folder))

        # Save the arguments required to create this agent (in addition to the config)
        init_kwargs = {
            "obs_space": space_to_json(self.obs_space),
            "action_dim": self.action_dim,
        }
        with (output_folder / "init_kwargs.json").open("w") as f:
            json.dump(init_kwargs, f, indent=4)
