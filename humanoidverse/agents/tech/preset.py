"""TeCH training preset public entrypoint."""

from humanoidverse.agents.presets.tldr import TRAIN_RUNTIME
from humanoidverse.agents.presets.tldr import build_tldr_agent as build_tech_agent

__all__ = ["TRAIN_RUNTIME", "build_tech_agent"]
