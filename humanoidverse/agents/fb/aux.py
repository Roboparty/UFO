"""Auxiliary-critic FB agent public entrypoints.

The implementation lives in ``fb_cpr_aux`` to preserve legacy config and
checkpoint names. New code should import through this module.
"""

from humanoidverse.agents.fb_cpr_aux.agent import (
    FBcprAuxAgent,
    FBcprAuxAgentConfig,
    FBcprAuxAgentTrainConfig,
)
from humanoidverse.agents.fb_cpr_aux.model import (
    FBcprAuxModel,
    FBcprAuxModelArchiConfig,
    FBcprAuxModelConfig,
)

__all__ = [
    "FBcprAuxAgent",
    "FBcprAuxAgentConfig",
    "FBcprAuxAgentTrainConfig",
    "FBcprAuxModel",
    "FBcprAuxModelArchiConfig",
    "FBcprAuxModelConfig",
]
