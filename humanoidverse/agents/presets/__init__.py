"""Agent presets used by the UFO training entrypoint."""

from __future__ import annotations

from typing import Any

from humanoidverse.agents.presets.fb import TRAIN_RUNTIME as FB_TRAIN_RUNTIME
from humanoidverse.agents.presets.fb import build_fb_agent
from humanoidverse.agents.presets.tldr import TRAIN_RUNTIME as TECH_TRAIN_RUNTIME
from humanoidverse.agents.presets.tldr import build_tldr_agent
from humanoidverse.agents.presets.tldr import build_tldr_agent as build_tech_agent


def build_agent_preset(
    *,
    agent: str,
    device: str,
    compile: bool,
    update_z_every_step: int,
    lr_scale: float,
    clip_grad_norm: float,
    cartwheel_aux_safe: bool,
    wandb_project: str,
) -> dict[str, Any]:
    if agent == "fb":
        return {
            "agent_cfg": build_fb_agent(
                device=device,
                compile=compile,
                update_z_every_step=update_z_every_step,
                lr_scale=lr_scale,
                clip_grad_norm=clip_grad_norm,
                cartwheel_aux_safe=cartwheel_aux_safe,
            ),
            "wandb_group": "ufo_fb",
            "wandb_project": wandb_project,
            "train_runtime": dict(FB_TRAIN_RUNTIME),
        }
    if agent in {"tech", "tldr"}:
        return {
            "agent_cfg": build_tech_agent(
                device=device,
                compile=compile,
                update_z_every_step=update_z_every_step,
            ),
            "wandb_group": "ufo_tech",
            "wandb_project": wandb_project,
            "train_runtime": dict(TECH_TRAIN_RUNTIME),
        }
    raise ValueError(f"Unsupported agent preset: {agent}")


__all__ = ["build_agent_preset", "build_fb_agent", "build_tech_agent", "build_tldr_agent"]
