"""TeCH agent public aliases."""

from humanoidverse.agents.tldr_dist_aux.agent import (
    TldrDistAuxAgent,
    TldrDistAuxAgentConfig,
    TldrDistAuxAgentTrainConfig,
)

TeCHAgent = TldrDistAuxAgent
TeCHAgentConfig = TldrDistAuxAgentConfig
TeCHAgentTrainConfig = TldrDistAuxAgentTrainConfig

__all__ = [
    "TeCHAgent",
    "TeCHAgentConfig",
    "TeCHAgentTrainConfig",
    "TldrDistAuxAgent",
    "TldrDistAuxAgentConfig",
    "TldrDistAuxAgentTrainConfig",
]
