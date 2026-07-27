# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Forward-backward agent family public entrypoints."""

from humanoidverse.agents.fb.agent import FBAgent, FBAgentConfig, FBAgentTrainConfig
from humanoidverse.agents.fb.model import FBModel, FBModelArchiConfig, FBModelConfig

__all__ = [
    "FBAgent",
    "FBAgentConfig",
    "FBAgentTrainConfig",
    "FBModel",
    "FBModelArchiConfig",
    "FBModelConfig",
]
