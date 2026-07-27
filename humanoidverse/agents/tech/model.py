"""TeCH model public aliases."""

from humanoidverse.agents.gcr_rl_dist_aux.model import (
    GcrRlDistAuxModel,
    GcrRlDistAuxModelArchiConfig,
    GcrRlDistAuxModelConfig,
)

TeCHModel = GcrRlDistAuxModel
TeCHModelArchiConfig = GcrRlDistAuxModelArchiConfig
TeCHModelConfig = GcrRlDistAuxModelConfig

__all__ = [
    "TeCHModel",
    "TeCHModelArchiConfig",
    "TeCHModelConfig",
    "GcrRlDistAuxModel",
    "GcrRlDistAuxModelArchiConfig",
    "GcrRlDistAuxModelConfig",
]
