# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pydantic
import torch
from torch.amp import autocast

from ..gcr_rl.model import GcrRlModel, GcrRlModelArchiConfig, GcrRlModelConfig
from ..nn_filter_models import DiscriminatorFilterArchiConfig
from ..nn_models import DiscriminatorArchiConfig


class GcrRlDistModelArchiConfig(GcrRlModelArchiConfig):
    discriminator: DiscriminatorArchiConfig | DiscriminatorFilterArchiConfig = pydantic.Field(
        DiscriminatorArchiConfig(), discriminator="name"
    )


class GcrRlDistModelConfig(GcrRlModelConfig):
    name: tp.Literal["GcrRlDistModel"] = "GcrRlDistModel"
    archi: GcrRlDistModelArchiConfig = GcrRlDistModelArchiConfig()

    @property
    def object_class(self):
        return GcrRlDistModel


class GcrRlDistModel(GcrRlModel):
    config_class = GcrRlDistModelConfig

    def __init__(self, obs_space, action_dim, cfg: GcrRlDistModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.cfg: GcrRlDistModelConfig = cfg
        self._discriminator = cfg.archi.discriminator.build(obs_space, cfg.archi.z_dim)

        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @torch.no_grad()
    def discriminator(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._discriminator(self._normalize(obs), z)
