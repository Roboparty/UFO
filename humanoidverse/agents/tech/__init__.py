"""TeCH agent family public entrypoints.

Historical TLDR/GCR module names remain in place for checkpoint and config
compatibility. New code should import TeCH through this package.
"""

from humanoidverse.agents.tech.agent import (
    TeCHAgent,
    TeCHAgentConfig,
    TeCHAgentTrainConfig,
    TldrDistAuxAgent,
    TldrDistAuxAgentConfig,
    TldrDistAuxAgentTrainConfig,
)
from humanoidverse.agents.tech.model import (
    GcrRlDistAuxModel,
    GcrRlDistAuxModelArchiConfig,
    GcrRlDistAuxModelConfig,
    TeCHModel,
    TeCHModelArchiConfig,
    TeCHModelConfig,
)
from humanoidverse.agents.tech.preset import build_tech_agent

__all__ = [
    "TeCHAgent",
    "TeCHAgentConfig",
    "TeCHAgentTrainConfig",
    "TeCHModel",
    "TeCHModelArchiConfig",
    "TeCHModelConfig",
    "TldrDistAuxAgent",
    "TldrDistAuxAgentConfig",
    "TldrDistAuxAgentTrainConfig",
    "GcrRlDistAuxModel",
    "GcrRlDistAuxModelArchiConfig",
    "GcrRlDistAuxModelConfig",
    "build_tech_agent",
]
