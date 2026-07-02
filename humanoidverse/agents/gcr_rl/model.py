# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Goal-conditioned representation (GCR) + policy pi(a|s,z_g)."""

import copy
import math
import typing as tp

import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast

from ..base import BaseConfig
from ..base_model import BaseModel, BaseModelConfig
from ..nn_models import ActorArchiConfig, BackwardArchiConfig, ForwardArchiConfig, eval_mode
from ..normalizers import ObsNormalizerConfig


class GcrRlModelArchiConfig(BaseConfig):
    z_dim: int = 256
    norm_z: bool = True
    z_geometry: tp.Literal["hypersphere", "poincare_ball"] = "hypersphere"
    poincare_radius: float = 1.0
    poincare_eps: float = 1e-6
    goal_encoder: BackwardArchiConfig = pydantic.Field(BackwardArchiConfig(), discriminator="name")
    actor: ActorArchiConfig = pydantic.Field(ActorArchiConfig(), discriminator="name")
    critic: ForwardArchiConfig = pydantic.Field(ForwardArchiConfig(), discriminator="name")


class GcrRlModelConfig(BaseModelConfig):
    name: tp.Literal["GcrRlModel"] = "GcrRlModel"
    archi: GcrRlModelArchiConfig = GcrRlModelArchiConfig()
    obs_normalizer: ObsNormalizerConfig = ObsNormalizerConfig()
    inference_batch_size: int = 500_000
    seq_length: int = 8
    actor_std: float = 0.05
    amp: bool = False

    def build(self, obs_space, action_dim) -> "GcrRlModel":
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return GcrRlModel


class GcrRlModel(BaseModel):
    config_class = GcrRlModelConfig

    def __init__(self, obs_space, action_dim, cfg: GcrRlModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.cfg: GcrRlModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device
        self.amp_dtype = torch.bfloat16

        self._goal_encoder = arch.goal_encoder.build(obs_space, arch.z_dim)
        self._actor = arch.actor.build(obs_space, arch.z_dim, action_dim)
        self._critic = arch.critic.build(obs_space, arch.z_dim, action_dim, output_dim=1)
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)

        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        self._target_critic = copy.deepcopy(self._critic)

    def _normalize(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.archi.z_geometry == "poincare_ball":
            radius = float(self.cfg.archi.poincare_radius)
            eps = float(self.cfg.archi.poincare_eps)
            z_norm = torch.linalg.norm(z, dim=-1, keepdim=True).clamp_min(eps)
            return radius * torch.tanh(z_norm / radius) * (z / z_norm)
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def encode_goal(self, goal_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            z = self._goal_encoder(self._normalize(goal_obs))
        return self.project_z(z)

    def actor(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, std: float):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._actor(self._normalize(obs), z, std)

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean.float()
        return dist.sample().float()

    def goal_inference(self, goal_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z_raw = self.backward_map(goal_obs)
        return self.project_z(z_raw)

    def reward_wr_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor) -> torch.Tensor:
        del reward
        return self.goal_inference(next_obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._goal_encoder(self._normalize(obs))

    def tracking_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)
