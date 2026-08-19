"""Opt-in terrain-conditioned FB preset.

The actor receives pelvis-to-terrain clearances through ``terrain_actor``.
F, critic, and auxiliary critic receive ground-relative geometry through
``terrain_priv``. Backward and discriminator remain terrain-agnostic.
"""

from __future__ import annotations

from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig
from humanoidverse.agents.presets.fb import TRAIN_RUNTIME, build_fb_agent

TERRAIN_CONTEXT_KEYS = ["state", "privileged_state", "last_action", "history_actor", "terrain_priv"]
TERRAIN_ACTOR_KEYS = ["state", "last_action", "history_actor", "terrain_actor"]


def build_fb_terrain_agent(**kwargs):
    base = build_fb_agent(**kwargs)
    archi = base.model.archi

    def with_filter(component, keys: list[str]):
        return component.model_copy(
            update={"input_filter": DictInputFilterConfig(name="DictInputFilterConfig", key=keys)}
        )

    terrain_archi = archi.model_copy(
        update={
            "f": with_filter(archi.f, TERRAIN_CONTEXT_KEYS),
            "actor": with_filter(archi.actor, TERRAIN_ACTOR_KEYS),
            "critic": with_filter(archi.critic, TERRAIN_CONTEXT_KEYS),
            "aux_critic": with_filter(archi.aux_critic, TERRAIN_CONTEXT_KEYS),
        }
    )
    normalizers = dict(base.model.obs_normalizer.normalizers)
    normalizers["terrain_actor"] = BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01)
    normalizers["terrain_priv"] = BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01)
    terrain_normalizer = base.model.obs_normalizer.model_copy(update={"normalizers": normalizers})
    terrain_model = base.model.model_copy(update={"archi": terrain_archi, "obs_normalizer": terrain_normalizer})
    update = {"model": terrain_model}
    if not kwargs.get("cartwheel_aux_safe", False):
        update.update(
            {
                "aux_rewards": [
                    "penalty_action_rate",
                    "limits_dof_pos",
                    "penalty_body_impact",
                    "penalty_slippage",
                    "penalty_ankle_roll",
                ],
                "aux_rewards_scaling": {
                    "penalty_action_rate": -0.1,
                    "limits_dof_pos": -10.0,
                    "penalty_body_impact": -1.0,
                    "penalty_slippage": -1.0,
                    "penalty_ankle_roll": -1.0,
                },
            }
        )
    return base.model_copy(update=update)


__all__ = ["TRAIN_RUNTIME", "build_fb_terrain_agent"]
