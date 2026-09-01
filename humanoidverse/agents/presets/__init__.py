"""Agent presets used by the UFO training entrypoint."""

from __future__ import annotations

from typing import Any

from humanoidverse.agents.presets.fb import TRAIN_RUNTIME as FB_TRAIN_RUNTIME
from humanoidverse.agents.presets.fb import build_fb_agent
from humanoidverse.agents.presets.fb_depth import build_fb_depth_agent
from humanoidverse.agents.presets.fb_terrain import build_fb_terrain_agent
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
    heading_context: bool = True,
    heading_reg_coeff: float = 0.0,
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
    if agent == "fb_terrain":
        return {
            "agent_cfg": build_fb_terrain_agent(
                device=device,
                compile=compile,
                update_z_every_step=update_z_every_step,
                lr_scale=lr_scale,
                clip_grad_norm=clip_grad_norm,
                cartwheel_aux_safe=cartwheel_aux_safe,
            ),
            "wandb_group": "ufo_fb_terrain_v0",
            "wandb_project": wandb_project,
            "train_runtime": dict(FB_TRAIN_RUNTIME),
        }
    if agent == "fb_depth":
        return {
            "agent_cfg": build_fb_depth_agent(
                device=device,
                compile=compile,
                update_z_every_step=update_z_every_step,
                lr_scale=lr_scale,
                clip_grad_norm=clip_grad_norm,
                cartwheel_aux_safe=cartwheel_aux_safe,
                heading_context=heading_context,
                heading_reg_coeff=heading_reg_coeff,
            ),
            "wandb_group": "pbfm_direct_depth_v0",
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


__all__ = [
    "build_agent_preset",
    "build_fb_agent",
    "build_fb_depth_agent",
    "build_fb_terrain_agent",
    "build_tech_agent",
    "build_tldr_agent",
]
