"""Diagnostic-only decomposition of the terrain Actor auxiliary gradient.

This command never updates the training run.  It freezes a checkpoint, fits
five reward-return heads plus a critic-pessimism return head, and attributes
the effective Actor gradient to each stored auxiliary reward.  The sixth
component combines the critic-pessimism return with actor-time ensemble
uncertainty.  The six components must reconstruct the frozen scalar Q_aux
gradient on held-out policy actions before the result is considered valid.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.direct_depth_actor_diagnostics import (
    MemoryMappedTrajectoryReplay,
    _actor_parameter_groups,
    _encode_expert,
    _find_expert_cache,
    _gradient_pair_metrics,
    _gradient_vector,
    _load_expert_buffer,
    _load_json,
    _sample_mixed_z,
    _summarize_reports,
    _to_device,
)
from humanoidverse.mjlab_inference_utils import checkpoint_load_device

UNCERTAINTY_COMPONENT = "uncertainty_penalty"
ACTOR_UNCERTAINTY_SUBCOMPONENT = "actor_ensemble_uncertainty"
CRITIC_RESIDUAL_SUBCOMPONENT = "critic_bootstrap_uncertainty_return"


def ensemble_mean_uncertainty(predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the training code's ensemble mean and pairwise uncertainty."""
    if predictions.shape[0] < 2:
        raise ValueError("Aux decomposition requires an ensemble with at least two critics")
    mean = predictions.mean(dim=0)
    left = predictions.unsqueeze(0)
    right = predictions.unsqueeze(1)
    scale = predictions.shape[0] ** 2 - predictions.shape[0]
    uncertainty = (left - right).abs().sum(dim=(0, 1)) / scale
    return mean, uncertainty


def _component_rewards(
    batch: Mapping[str, Any],
    *,
    reward_names: Sequence[str],
    reward_scales: Mapping[str, float],
    reward_std: torch.Tensor,
    device: str,
) -> torch.Tensor:
    components = []
    for name in reward_names:
        if name not in batch["aux_rewards"]:
            raise KeyError(f"Replay does not contain auxiliary reward {name!r}")
        value = batch["aux_rewards"][name].to(device=device, dtype=torch.float32)
        components.append(value * float(reward_scales[name]) / reward_std)
    return torch.cat(components, dim=-1)


def _normalizer_std(model, agent_config: Mapping[str, Any]) -> torch.Tensor:
    config = agent_config["model"]["norm_aux_reward"]
    if bool(config.get("translate", False)):
        raise ValueError("The diagnostic currently requires a non-translating total Aux normalizer")
    if not bool(config.get("scale", False)):
        return torch.ones((1,), device=model.device, dtype=torch.float32)
    return model._aux_reward_normalizer.S.detach().sqrt().to(device=model.device, dtype=torch.float32)


def _expanded_component_state(
    scalar_module: torch.nn.Module,
    component_module: torch.nn.Module,
    fractions: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    scalar_state = scalar_module.state_dict()
    output = component_module.state_dict()
    head_names: list[str] = []
    components = int(fractions.numel()) + 1
    fraction_shape: tuple[int, ...]
    for name, component_value in output.items():
        scalar_value = scalar_state[name]
        if component_value.shape == scalar_value.shape:
            output[name] = scalar_value.detach().clone()
            continue
        if (
            component_value.ndim == scalar_value.ndim
            and component_value.shape[:-1] == scalar_value.shape[:-1]
            and component_value.shape[-1] == components
            and scalar_value.shape[-1] == 1
        ):
            fraction_shape = (1,) * (scalar_value.ndim - 1) + (fractions.numel(),)
            reward_values = scalar_value * fractions.reshape(fraction_shape).to(scalar_value)
            residual = scalar_value - reward_values.sum(dim=-1, keepdim=True)
            output[name] = torch.cat((reward_values, residual), dim=-1)
            head_names.append(name)
            continue
        raise ValueError(
            f"Unexpected component-critic shape mismatch for {name}: "
            f"scalar={tuple(scalar_value.shape)}, component={tuple(component_value.shape)}"
        )
    if len(head_names) != 2:
        raise ValueError(f"Expected output weight and bias tensors, found {head_names}")
    return output, tuple(head_names)


def build_component_critic(
    model,
    scalar_module: torch.nn.Module,
    *,
    reward_fractions: torch.Tensor,
) -> tuple[torch.nn.Module, tuple[str, ...]]:
    output_dim = int(reward_fractions.numel()) + 1
    module = model.cfg.archi.aux_critic.build(
        model.obs_space,
        model.cfg.archi.z_dim,
        model.action_dim,
        output_dim=output_dim,
    ).to(model.device)
    state, head_names = _expanded_component_state(scalar_module, module, reward_fractions)
    module.load_state_dict(state, strict=True)
    module.requires_grad_(False)
    named = dict(module.named_parameters())
    for name in head_names:
        named[name].requires_grad_(True)
    return module, head_names


@torch.no_grad()
def project_residual_head(
    component_module: torch.nn.Module,
    scalar_module: torch.nn.Module,
    head_names: Sequence[str],
) -> None:
    component_parameters = dict(component_module.named_parameters())
    scalar_parameters = dict(scalar_module.named_parameters())
    for name in head_names:
        component = component_parameters[name]
        scalar = scalar_parameters[name]
        component[..., -1].copy_(scalar[..., 0] - component[..., :-1].sum(dim=-1))


@torch.no_grad()
def soft_update_reward_heads(
    current: torch.nn.Module,
    target: torch.nn.Module,
    *,
    head_names: Sequence[str],
    reward_count: int,
    tau: float,
) -> None:
    current_parameters = dict(current.named_parameters())
    target_parameters = dict(target.named_parameters())
    for name in head_names:
        target_parameters[name][..., :reward_count].lerp_(
            current_parameters[name][..., :reward_count],
            float(tau),
        )


@torch.no_grad()
def soft_update_module(current: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for current_parameter, target_parameter in zip(current.parameters(), target.parameters()):
        target_parameter.lerp_(current_parameter, float(tau))


def component_sum_error(
    scalar_module: torch.nn.Module,
    component_module: torch.nn.Module,
    obs: Mapping[str, torch.Tensor],
    z: torch.Tensor,
    action: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        scalar = scalar_module(obs, z, action)
        reconstructed = component_module(obs, z, action).sum(dim=-1, keepdim=True)
        difference = reconstructed - scalar
        return {
            "max_abs": float(difference.abs().max().item()),
            "mean_abs": float(difference.abs().mean().item()),
            "relative_l2": float(
                (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(scalar).clamp_min(1.0e-12)).item()
            ),
        }


def _estimate_reward_fractions(
    *,
    replay: MemoryMappedTrajectoryReplay,
    reward_names: Sequence[str],
    reward_scales: Mapping[str, float],
    samples: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = replay.sample(samples)
    magnitudes = torch.tensor(
        [
            float((batch["aux_rewards"][name].float() * float(reward_scales[name])).abs().mean().item())
            for name in reward_names
        ],
        dtype=torch.float32,
    )
    fractions = magnitudes / magnitudes.sum().clamp_min(1.0e-12)
    return fractions, {name: float(value) for name, value in zip(reward_names, fractions.tolist())}


@torch.no_grad()
def _sample_training_batch(
    *,
    model,
    agent_config: Mapping[str, Any],
    replay: MemoryMappedTrajectoryReplay,
    expert_buffer,
    batch_size: int,
    device: str,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], torch.Tensor]:
    train = agent_config["train"]
    batch = replay.sample(batch_size)
    expert = expert_buffer.sample(batch_size)
    obs = model._normalize(_to_device(batch["observation"], device))
    next_obs = model._normalize(_to_device(batch["next"]["observation"], device))
    expert_next = model._normalize(_to_device(expert["next"]["observation"], device))
    expert_z = _encode_expert(model, expert_next)
    sampled_z = _sample_mixed_z(
        model,
        next_obs,
        expert_z,
        p_goal=float(train["train_goal_ratio"]),
        p_expert=float(train["expert_asm_ratio"]),
    )
    relabel_ratio = train.get("relabel_ratio")
    if relabel_ratio is None:
        z = batch["z"].to(device)
    else:
        mask = torch.rand((batch_size, 1), device=device) <= float(relabel_ratio)
        z = torch.where(mask, sampled_z, batch["z"].to(device))
    return batch, obs, next_obs, z


def fit_component_critics(
    *,
    model,
    agent_config: Mapping[str, Any],
    replay: MemoryMappedTrajectoryReplay,
    expert_buffer,
    batch_size: int,
    updates: int,
    learning_rate: float,
    distill_coefficient: float,
    local_action_distill_coefficient: float,
    action_distill_noise_std: float,
    seed: int,
    device: str,
    log_every: int,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    reward_names = tuple(agent_config["aux_rewards"])
    reward_scales = agent_config["aux_rewards_scaling"]
    reward_std = _normalizer_std(model, agent_config)
    fractions, fraction_report = _estimate_reward_fractions(
        replay=replay,
        reward_names=reward_names,
        reward_scales=reward_scales,
        samples=min(max(batch_size * 4, 4096), 32768),
    )
    current, head_names = build_component_critic(
        model,
        model._aux_critic,
        reward_fractions=fractions.to(model.device),
    )
    target, target_head_names = build_component_critic(
        model,
        model._target_aux_critic,
        reward_fractions=fractions.to(model.device),
    )
    if head_names != target_head_names:
        raise AssertionError("Current and target component heads do not match")
    reward_count = len(reward_names)
    # The diagnostic critic is allowed to learn a real multi-task value
    # representation.  A total-Q distillation term, evaluated at both replay
    # and current-policy actions, keeps the sum tied to the frozen scalar
    # critic without making reconstruction true by construction.
    current.requires_grad_(True)
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(current.parameters(), lr=float(learning_rate))
    train_config = agent_config["train"]
    discount_value = float(train_config["discount"])
    tau = float(train_config["critic_target_tau"])
    curves: list[dict[str, Any]] = []
    rolling: list[torch.Tensor] = []
    current.train(True)
    target.train(False)

    for update in range(1, updates + 1):
        torch.manual_seed(seed + update)
        np.random.seed(seed + update)
        batch, obs, next_obs, z = _sample_training_batch(
            model=model,
            agent_config=agent_config,
            replay=replay,
            expert_buffer=expert_buffer,
            batch_size=batch_size,
            device=device,
        )
        action = batch["action"].to(device)
        rewards = _component_rewards(
            batch,
            reward_names=reward_names,
            reward_scales=reward_scales,
            reward_std=reward_std,
            device=device,
        )
        terminated = batch["next"].get("terminated")
        if terminated is None:
            raise KeyError("Replay must expose next.terminated to preserve the current Aux Bellman semantics")
        terminated = terminated.to(device).bool().reshape(batch_size, -1).any(dim=-1, keepdim=True)
        discount = discount_value * (~terminated).float()
        with torch.no_grad():
            next_dist = model._actor(next_obs, z, model.cfg.actor_std)
            next_action = next_dist.sample(clip=float(train_config["stddev_clip"]))
            next_components = target(next_obs, z, next_action)
            next_mean = next_components.mean(dim=0)
            _next_total_mean, next_total_uncertainty = ensemble_mean_uncertainty(
                next_components.sum(dim=-1, keepdim=True)
            )
            reward_targets = rewards + discount * next_mean[..., :reward_count]
            uncertainty_target = discount * (
                next_mean[..., -1:] - float(train_config["aux_critic_pessimism_penalty"]) * next_total_uncertainty
            )
            targets = torch.cat((reward_targets, uncertainty_target), dim=-1)
            scalar_replay = model._aux_critic(obs, z, action)
            policy_dist = model._actor(obs, z, model.cfg.actor_std)
            policy_action = policy_dist.sample(clip=float(train_config["stddev_clip"])).detach()
            action_noise = torch.randn_like(policy_action) * float(action_distill_noise_std)
            policy_action_plus = (policy_action + action_noise).clamp(-1.0, 1.0)
            policy_action_minus = (policy_action - action_noise).clamp(-1.0, 1.0)
            scalar_policy = model._aux_critic(obs, z, policy_action)
            scalar_policy_plus = model._aux_critic(obs, z, policy_action_plus)
            scalar_policy_minus = model._aux_critic(obs, z, policy_action_minus)

        predictions = current(obs, z, action)
        expanded_targets = targets.unsqueeze(0).expand(predictions.shape[0], -1, -1)
        per_component_loss = 0.5 * predictions.shape[0] * (predictions - expanded_targets).square().mean(dim=(0, 1))
        policy_predictions = current(obs, z, policy_action)
        policy_predictions_plus = current(obs, z, policy_action_plus)
        policy_predictions_minus = current(obs, z, policy_action_minus)
        replay_distillation = F.mse_loss(predictions.sum(dim=-1, keepdim=True), scalar_replay)
        policy_distillation = F.mse_loss(policy_predictions.sum(dim=-1, keepdim=True), scalar_policy)
        distillation_loss = 0.5 * (replay_distillation + policy_distillation)
        local_action_distillation_loss = 0.5 * (
            F.mse_loss(policy_predictions_plus.sum(dim=-1, keepdim=True), scalar_policy_plus)
            + F.mse_loss(policy_predictions_minus.sum(dim=-1, keepdim=True), scalar_policy_minus)
        )
        loss = (
            per_component_loss.sum()
            + float(distill_coefficient) * distillation_loss
            + float(local_action_distill_coefficient) * local_action_distillation_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        soft_update_module(current, target, tau)
        rolling.append(
            torch.cat(
                (
                    per_component_loss.detach().cpu(),
                    distillation_loss.detach().cpu().reshape(1),
                    local_action_distillation_loss.detach().cpu().reshape(1),
                )
            )
        )
        if update == 1 or update % log_every == 0 or update == updates:
            window = torch.stack(rolling)
            mean_loss = window.mean(dim=0)
            component_names = (*reward_names, CRITIC_RESIDUAL_SUBCOMPONENT)
            entry = {
                "update": update,
                "bellman_loss_total": float(mean_loss[:-2].sum().item()),
                "loss_by_component": {
                    name: float(value) for name, value in zip(component_names, mean_loss[:-2].tolist())
                },
                "total_Q_distillation_loss": float(mean_loss[-2].item()),
                "local_action_distillation_loss": float(mean_loss[-1].item()),
            }
            curves.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            rolling.clear()

    current.eval()
    target.eval()
    return current, target, {
        "updates": updates,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "distill_coefficient": distill_coefficient,
        "local_action_distill_coefficient": local_action_distill_coefficient,
        "action_distill_noise_std": action_distill_noise_std,
        "target_tau": tau,
        "reward_normalizer_std": float(reward_std.item()),
        "initial_reward_fractions": fraction_report,
        "head_parameter_names": list(head_names),
        "curve": curves,
    }


def _add_gradients(*branches: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    # Component gradients can be much larger than their final, cancelling
    # sum.  Accumulate in float64 so the reconstruction check measures the
    # decomposition rather than Python-order float32 cancellation.
    output = []
    for values in zip(*branches):
        total = torch.zeros_like(values[0], dtype=torch.float64)
        for value in values:
            total.add_(value.double())
        output.append(total)
    return tuple(output)


def _subtract_gradients(first: Sequence[torch.Tensor], second: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(left - right for left, right in zip(first, second))


def _gradient_norm(gradient: Sequence[torch.Tensor], indices: Sequence[int]) -> float:
    squared = sum(gradient[index].double().square().sum() for index in indices)
    return float(torch.sqrt(squared).item())


def _gradient_report(
    *,
    actor: torch.nn.Module,
    fb_gradient: Sequence[torch.Tensor],
    component_gradients: Mapping[str, Sequence[torch.Tensor]],
    original_gradient: Sequence[torch.Tensor],
    reconstructed_gradient: Sequence[torch.Tensor],
) -> dict[str, Any]:
    parameters = tuple(actor.parameters())
    parameter_to_index = {id(parameter): index for index, parameter in enumerate(parameters)}
    all_groups = _actor_parameter_groups(actor)
    groups = {
        "all_actor": all_groups["all"],
        "depth_encoder": all_groups["depth_encoder"],
        "policy_head": all_groups["policy"],
    }
    output: dict[str, Any] = {}
    for group_name, group_parameters in groups.items():
        indices = [parameter_to_index[id(parameter)] for parameter in group_parameters]
        fb_norm = _gradient_norm(fb_gradient, indices)
        original_norm = _gradient_norm(original_gradient, indices)
        entries: dict[str, Any] = {}
        for name, gradient in component_gradients.items():
            norm = _gradient_norm(gradient, indices)
            pair = _gradient_pair_metrics(gradient, fb_gradient, indices)
            entries[name] = {
                "norm": norm,
                "norm_over_FB": norm / max(fb_norm, 1.0e-30),
                "cosine_with_FB": pair["cosine"],
                "projection_on_FB": pair["first_projection_on_second"],
            }
        reconstruction = _gradient_pair_metrics(reconstructed_gradient, original_gradient, indices)
        error = _gradient_norm(_subtract_gradients(reconstructed_gradient, original_gradient), indices)
        output[group_name] = {
            "FB_norm": fb_norm,
            "original_Aux_norm": original_norm,
            "original_Aux_over_FB": original_norm / max(fb_norm, 1.0e-30),
            "components": entries,
            "reconstruction": {
                "cosine": reconstruction["cosine"],
                "norm_ratio": reconstruction["first_norm"] / max(reconstruction["second_norm"], 1.0e-30),
                "relative_error": error / max(original_norm, 1.0e-30),
            },
        }
    return output


def evaluate_gradient_decomposition(
    *,
    model,
    component_critic: torch.nn.Module,
    agent_config: Mapping[str, Any],
    replay: MemoryMappedTrajectoryReplay,
    expert_buffer,
    batch_size: int,
    batches: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    reward_names = tuple(agent_config["aux_rewards"])
    train = agent_config["train"]
    actor = model._actor
    actor_was_training = actor.training
    actor.train(True)
    actor.requires_grad_(True)
    parameters = tuple(actor.parameters())
    reports: list[dict[str, Any]] = []
    for batch_index in range(batches):
        torch.manual_seed(seed + 100_000 + batch_index)
        np.random.seed(seed + 100_000 + batch_index)
        _batch, obs, _next_obs, z = _sample_training_batch(
            model=model,
            agent_config=agent_config,
            replay=replay,
            expert_buffer=expert_buffer,
            batch_size=batch_size,
            device=device,
        )
        distribution = actor(obs, z, model.cfg.actor_std)
        action = distribution.sample(clip=float(train["stddev_clip"]))
        scalar_predictions = model._aux_critic(obs, z, action)
        component_predictions = component_critic(obs, z, action)
        scalar_mean, scalar_uncertainty = ensemble_mean_uncertainty(scalar_predictions)
        component_mean = component_predictions.mean(dim=0)
        component_total = component_predictions.sum(dim=-1, keepdim=True)
        _component_total_mean, component_total_uncertainty = ensemble_mean_uncertainty(component_total)
        q_scalar = scalar_mean - float(train["actor_pessimism_penalty"]) * scalar_uncertainty

        forward = model._forward_map(obs, z, action)
        q_fb_predictions = (forward * z).sum(dim=-1)
        q_fb_mean, q_fb_uncertainty = ensemble_mean_uncertainty(q_fb_predictions)
        q_fb = q_fb_mean - float(train["actor_pessimism_penalty"]) * q_fb_uncertainty
        weight = q_fb.abs().mean().detach() if bool(train["scale_reg"]) else q_fb.new_ones(())
        effective_scale = float(train["reg_coeff_aux"]) * weight

        reward_losses = {
            name: -component_mean[..., index].mean() * effective_scale
            for index, name in enumerate(reward_names)
        }
        critic_residual_loss = -component_mean[..., -1].mean() * effective_scale
        actor_uncertainty_loss = (
            float(train["actor_pessimism_penalty"])
            * component_total_uncertainty.mean()
            * effective_scale
        )
        uncertainty_loss = critic_residual_loss + actor_uncertainty_loss
        original_loss = -q_scalar.mean() * effective_scale
        fb_loss = -q_fb.mean()
        reconstructed_loss = sum(reward_losses.values()) + uncertainty_loss

        fb_gradient = _gradient_vector(fb_loss, parameters, retain_graph=True)
        reward_gradients: dict[str, tuple[torch.Tensor, ...]] = {}
        for name in reward_names:
            reward_gradients[name] = _gradient_vector(reward_losses[name], parameters, retain_graph=True)
        critic_residual_gradient = _gradient_vector(critic_residual_loss, parameters, retain_graph=True)
        actor_uncertainty_gradient = _gradient_vector(actor_uncertainty_loss, parameters, retain_graph=True)
        uncertainty_gradient = _add_gradients(critic_residual_gradient, actor_uncertainty_gradient)
        original_gradient = _gradient_vector(original_loss, parameters, retain_graph=False)
        reconstructed_gradient = _add_gradients(*reward_gradients.values(), uncertainty_gradient)
        component_gradients = dict(reward_gradients)
        component_gradients[UNCERTAINTY_COMPONENT] = uncertainty_gradient
        component_gradients[ACTOR_UNCERTAINTY_SUBCOMPONENT] = actor_uncertainty_gradient
        component_gradients[CRITIC_RESIDUAL_SUBCOMPONENT] = critic_residual_gradient
        sum_error = component_sum_error(model._aux_critic, component_critic, obs, z, action)
        reports.append(
            {
                "batch_index": batch_index,
                "effective_scale": float(effective_scale.item()),
                "q": {
                    "FB": float(q_fb.mean().item()),
                    "Aux_original": float(q_scalar.mean().item()),
                    "actor_uncertainty": float(scalar_uncertainty.mean().item()),
                    "critic_bootstrap_uncertainty_return_mean": float(component_mean[..., -1].mean().item()),
                },
                "loss_reconstruction_abs": float((reconstructed_loss - original_loss).abs().item()),
                "value_sum_error": sum_error,
                "gradients": _gradient_report(
                    actor=actor,
                    fb_gradient=fb_gradient,
                    component_gradients=component_gradients,
                    original_gradient=original_gradient,
                    reconstructed_gradient=reconstructed_gradient,
                ),
            }
        )
        del distribution, action, scalar_predictions, component_predictions
        torch.cuda.empty_cache()
    actor.requires_grad_(False)
    actor.train(actor_was_training)
    summary = _summarize_reports(reports)
    validity = validate_reconstruction(reports)
    return {
        "batch_size": batch_size,
        "num_batches": batches,
        "per_batch": reports,
        "summary": summary,
        "validity": validity,
    }


def validate_reconstruction(
    reports: Sequence[Mapping[str, Any]],
    *,
    min_cosine: float = 0.999,
    max_norm_error: float = 0.01,
    max_relative_error: float = 0.01,
) -> dict[str, Any]:
    failures: list[str] = []
    for report in reports:
        batch_index = int(report["batch_index"])
        for group in ("all_actor", "depth_encoder", "policy_head"):
            reconstruction = report["gradients"][group]["reconstruction"]
            if float(reconstruction["cosine"]) < min_cosine:
                failures.append(f"batch {batch_index} {group}: cosine={reconstruction['cosine']:.6g}")
            if abs(float(reconstruction["norm_ratio"]) - 1.0) > max_norm_error:
                failures.append(f"batch {batch_index} {group}: norm_ratio={reconstruction['norm_ratio']:.6g}")
            if float(reconstruction["relative_error"]) > max_relative_error:
                failures.append(f"batch {batch_index} {group}: relative_error={reconstruction['relative_error']:.6g}")
        if float(report["value_sum_error"]["relative_l2"]) > 1.0e-4:
            failures.append(
                f"batch {batch_index}: value relative_l2={report['value_sum_error']['relative_l2']:.6g}"
            )
    return {
        "valid": not failures,
        "thresholds": {
            "min_cosine": min_cosine,
            "max_norm_ratio_error": max_norm_error,
            "max_relative_gradient_error": max_relative_error,
            "max_relative_value_error": 1.0e-4,
        },
        "failures": failures,
        "reward_change_authorized_by_diagnostic": False,
    }


def _checkpoint_step(checkpoint_dir: Path) -> int | None:
    status = checkpoint_dir / "train_status.json"
    if not status.exists():
        return None
    return int(_load_json(status).get("global_time", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--buffer-dir", type=Path, default=None)
    parser.add_argument("--buffer-rank", type=int, default=0)
    parser.add_argument("--expert-cache", type=Path, default=None)
    parser.add_argument("--expert-cache-root", type=Path, default=Path("/data/xue/UFO/cache/expert_buffers"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9371)
    parser.add_argument("--fit-batch-size", type=int, default=512)
    parser.add_argument("--fit-updates", type=int, default=2000)
    parser.add_argument("--fit-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--distill-coefficient", type=float, default=10.0)
    parser.add_argument("--local-action-distill-coefficient", type=float, default=100.0)
    parser.add_argument("--action-distill-noise-std", type=float, default=0.05)
    parser.add_argument("--fit-log-every", type=int, default=100)
    parser.add_argument(
        "--component-critics",
        type=Path,
        default=None,
        help="Load a previously fitted diagnostic_component_critics.pt and skip fitting",
    )
    parser.add_argument("--gradient-batch-size", type=int, default=256)
    parser.add_argument("--gradient-batches", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else run_dir / "checkpoint"
    buffer_dir = args.buffer_dir.expanduser().resolve() if args.buffer_dir else checkpoint_dir / "buffers"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expert_cache = args.expert_cache.expanduser().resolve() if args.expert_cache else _find_expert_cache(args.expert_cache_root)
    checkpoint_step = _checkpoint_step(checkpoint_dir)
    load_device = checkpoint_load_device(args.device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=load_device)
    checkpoint_step_after_load = _checkpoint_step(checkpoint_dir)
    if checkpoint_step_after_load != checkpoint_step:
        raise RuntimeError(
            "Checkpoint changed while the frozen diagnostic model was loading: "
            f"{checkpoint_step} -> {checkpoint_step_after_load}. Re-run on a stable checkpoint."
        )
    model.to(args.device).eval().requires_grad_(False)
    agent_config = _load_json(checkpoint_dir / "config.json")
    replay = MemoryMappedTrajectoryReplay(buffer_dir / f"train_rank_{args.buffer_rank}")
    expert_buffer = _load_expert_buffer(expert_cache)

    if args.component_critics is None:
        component_critic, target_component_critic, fit_report = fit_component_critics(
            model=model,
            agent_config=agent_config,
            replay=replay,
            expert_buffer=expert_buffer,
            batch_size=args.fit_batch_size,
            updates=args.fit_updates,
            learning_rate=args.fit_learning_rate,
            distill_coefficient=args.distill_coefficient,
            local_action_distill_coefficient=args.local_action_distill_coefficient,
            action_distill_noise_std=args.action_distill_noise_std,
            seed=args.seed,
            device=args.device,
            log_every=args.fit_log_every,
        )
    else:
        component_path = args.component_critics.expanduser().resolve()
        payload = torch.load(component_path, map_location=args.device, weights_only=True)
        if payload.get("checkpoint_global_step") != checkpoint_step:
            raise ValueError(
                f"Component critic step {payload.get('checkpoint_global_step')} does not match checkpoint {checkpoint_step}"
            )
        reward_names = tuple(agent_config["aux_rewards"])
        fraction_map = payload["fit"]["initial_reward_fractions"]
        fractions = torch.tensor([fraction_map[name] for name in reward_names], device=args.device)
        component_critic, _head_names = build_component_critic(
            model,
            model._aux_critic,
            reward_fractions=fractions,
        )
        target_component_critic, _target_head_names = build_component_critic(
            model,
            model._target_aux_critic,
            reward_fractions=fractions,
        )
        component_critic.load_state_dict(payload["component_critic"], strict=True)
        target_component_critic.load_state_dict(payload["target_component_critic"], strict=True)
        component_critic.eval().requires_grad_(False)
        target_component_critic.eval().requires_grad_(False)
        fit_report = dict(payload["fit"])
        fit_report["loaded_from"] = str(component_path)
    gradient_report = evaluate_gradient_decomposition(
        model=model,
        component_critic=component_critic,
        agent_config=agent_config,
        replay=replay,
        expert_buffer=expert_buffer,
        batch_size=args.gradient_batch_size,
        batches=args.gradient_batches,
        seed=args.seed,
        device=args.device,
    )
    report = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_global_step": checkpoint_step,
        "buffer_rank": args.buffer_rank,
        "expert_cache": str(expert_cache),
        "reward_names": list(agent_config["aux_rewards"]),
        "reward_scales": agent_config["aux_rewards_scaling"],
        "fit": fit_report,
        "gradient_decomposition": gradient_report,
        "interpretation_guard": (
            "Do not change reward weights unless reconstruction is valid and the fitted reward-head losses are stable. "
            "The sixth component combines the learned critic-bootstrap uncertainty return with actor-time ensemble "
            "uncertainty; its two subcomponents are reported separately."
        ),
    }
    with (output_dir / "aux_gradient_decomposition.json").open("w") as stream:
        json.dump(report, stream, indent=2)
    torch.save(
        {
            "component_critic": component_critic.state_dict(),
            "target_component_critic": target_component_critic.state_dict(),
            "reward_names": list(agent_config["aux_rewards"]),
            "checkpoint_global_step": report["checkpoint_global_step"],
            "fit": fit_report,
        },
        output_dir / "diagnostic_component_critics.pt",
    )
    print(json.dumps({"output": str(output_dir), "validity": gradient_report["validity"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
