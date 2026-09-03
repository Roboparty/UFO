"""PBFM direct-depth preset with the UFO-rp1 visual branch."""

from __future__ import annotations

from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import DirectDepthActorArchiConfig
from humanoidverse.agents.normalizers import IdentityNormalizerConfig
from humanoidverse.agents.presets.fb import TRAIN_RUNTIME
from humanoidverse.agents.presets.fb_terrain import build_fb_terrain_agent


def build_fb_depth_agent(
    *,
    heading_context: bool = True,
    heading_reg_coeff: float = 0.0,
    behavior_prior: bool = True,
    selective_prior: bool = False,
    **kwargs,
):
    if heading_reg_coeff < 0.0:
        raise ValueError("heading_reg_coeff must be non-negative")
    if heading_reg_coeff > 0.0 and not heading_context:
        raise ValueError("heading_reg_coeff requires heading_context=True")
    if selective_prior and not behavior_prior:
        raise ValueError("selective_prior requires behavior_prior=True")
    if selective_prior and not heading_context:
        raise ValueError("selective_prior requires BehaviorContext heading metadata")
    base = build_fb_terrain_agent(**kwargs)
    archi = base.model.archi
    source_actor = archi.actor
    actor_keys = ["state", "last_action", "history_actor"]
    context_keys = ["state", "privileged_state", "last_action", "history_actor", "terrain_priv"]
    if heading_context:
        actor_keys.append("heading")
        context_keys.append("heading")
    depth_actor = DirectDepthActorArchiConfig(
        name="direct_depth",
        model="residual",
        hidden_dim=source_actor.hidden_dim,
        hidden_layers=source_actor.hidden_layers,
        embedding_layers=source_actor.embedding_layers,
        input_filter=DictInputFilterConfig(
            name="DictInputFilterConfig",
            key=actor_keys,
        ),
        depth_key="depth_image",
        depth_channels=8,
        depth_height=36,
        depth_width=32,
        depth_latent_dim=256,
    )

    def with_context_filter(component):
        return component.model_copy(update={"input_filter": DictInputFilterConfig(name="DictInputFilterConfig", key=context_keys)})

    # The scalar behavior-prior and auxiliary value functions do not need to
    # duplicate the full 2048x6 successor-feature capacity.  Preserve their
    # twin ensembles and target networks while reducing only critic capacity.
    discriminator_critic = archi.critic.model_copy(update={"hidden_dim": 1024, "hidden_layers": 4})
    aux_critic = archi.aux_critic.model_copy(update={"hidden_layers": 4})
    archi_update = {
        "actor": depth_actor,
        "critic": discriminator_critic,
        "aux_critic": aux_critic,
    }
    if heading_context:
        # Heading value prediction is a much simpler scalar objective than
        # FB or the heterogeneous auxiliary return.  Keep the twin ensemble
        # and target-network semantics, but do not duplicate a full 2048x6
        # critic solely for Q_H.
        heading_critic = with_context_filter(archi.aux_critic).model_copy(update={"hidden_dim": 1024, "hidden_layers": 3})
        archi_update.update(
            {
                "f": with_context_filter(archi.f),
                "critic": with_context_filter(discriminator_critic),
                "aux_critic": with_context_filter(aux_critic),
                "heading_critic": heading_critic,
            }
        )
    depth_archi = archi.model_copy(update=archi_update)
    normalizers = dict(base.model.obs_normalizer.normalizers)
    normalizers.pop("terrain_actor", None)
    normalizers["depth_image"] = IdentityNormalizerConfig(name="IdentityNormalizerConfig")
    if heading_context:
        # Invalid and valid zero-error contexts must both remain exactly
        # [0,0]; BatchNorm would turn them into a hidden source command.
        normalizers["heading"] = IdentityNormalizerConfig(name="IdentityNormalizerConfig")
    depth_normalizer = base.model.obs_normalizer.model_copy(update={"normalizers": normalizers})
    depth_model = base.model.model_copy(
        update={
            "archi": depth_archi,
            "obs_normalizer": depth_normalizer,
            "heading_context_enabled": bool(heading_context),
            # Observation-only ablation (lambda_H=0) does not even allocate
            # Q_H, so it has no optimizer/DDP/VRAM side effects.
            "heading_critic_enabled": bool(heading_context and heading_reg_coeff > 0.0),
        }
    )
    depth_train = base.train.model_copy(
        update={
            "reg_coeff_heading": float(heading_reg_coeff),
            "behavior_prior_enabled": bool(behavior_prior),
            "selective_prior_enabled": bool(selective_prior),
            # Direct-depth formal runs use the original main-policy behavior
            # prior topology with an AMP-style bounded LSGAN reward. Other FB
            # presets retain their historical BCE/log-odds defaults.
            "discriminator_loss": "lsgan",
            "discriminator_reward": "amp",
            # A disabled branch must have zero Actor weight even if an old
            # config is later inspected or resumed through generic tooling.
            "reg_coeff": base.train.reg_coeff if behavior_prior else 0.0,
        }
    )
    return base.model_copy(update={"model": depth_model, "train": depth_train})


__all__ = ["TRAIN_RUNTIME", "build_fb_depth_agent"]
