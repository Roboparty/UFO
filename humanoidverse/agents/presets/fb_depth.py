"""PBFM direct-depth preset with the UFO-rp1 visual branch."""

from __future__ import annotations

from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import DirectDepthActorArchiConfig
from humanoidverse.agents.normalizers import IdentityNormalizerConfig
from humanoidverse.agents.presets.fb import TRAIN_RUNTIME
from humanoidverse.agents.presets.fb_terrain import build_fb_terrain_agent


def build_fb_depth_agent(**kwargs):
    base = build_fb_terrain_agent(**kwargs)
    archi = base.model.archi
    source_actor = archi.actor
    depth_actor = DirectDepthActorArchiConfig(
        name="direct_depth",
        model="residual",
        hidden_dim=source_actor.hidden_dim,
        hidden_layers=source_actor.hidden_layers,
        embedding_layers=source_actor.embedding_layers,
        input_filter=DictInputFilterConfig(
            name="DictInputFilterConfig",
            key=["state", "last_action", "history_actor"],
        ),
        depth_key="depth_image",
        depth_channels=8,
        depth_height=36,
        depth_width=32,
        depth_latent_dim=256,
    )
    depth_archi = archi.model_copy(update={"actor": depth_actor})
    normalizers = dict(base.model.obs_normalizer.normalizers)
    normalizers.pop("terrain_actor", None)
    normalizers["depth_image"] = IdentityNormalizerConfig(name="IdentityNormalizerConfig")
    depth_normalizer = base.model.obs_normalizer.model_copy(update={"normalizers": normalizers})
    depth_model = base.model.model_copy(update={"archi": depth_archi, "obs_normalizer": depth_normalizer})
    return base.model_copy(update={"model": depth_model})


__all__ = ["TRAIN_RUNTIME", "build_fb_depth_agent"]
