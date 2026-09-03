"""Tracking inference and video export for UFO policies.

Policy rollout is rendered from the training environment, while the reference
motion is rendered from the configured robot MJCF with pure MuJoCo qpos playback.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import joblib
import mediapy as media
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.actor_override import load_actor_override
from humanoidverse.agents.behavior_context import (
    align_heading_sequence,
    heading_observation,
    rotation_between_heading_xy,
    root_heading_xy,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.export.backward_encoder import (
    UnsupportedBackwardEncoderExport,
    export_backward_encoder_from_model,
)
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    add_bool_arg,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    policy_qpos_from_env,
    replace_hydra_override,
)
from humanoidverse.perception.depth_terrain_runtime import TemporalDepthTerrainRuntime
from humanoidverse.terrains.rp1_simple import (
    RP1_CENTER_PLATFORM_WIDTH,
    RP1_STAIR_LEVELS,
    RP1_STAIR_STEP_WIDTH,
    RP1_TERRAIN_COMPONENT_NAMES,
)
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, get_backward_observation
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from humanoidverse.utils.robot_spec import assert_robot_configs_compatible, load_robot_training_spec, resolve_robot_config_path

DEFAULT_ROBOT_CONFIG = "configs/robots/g1_29dof.yaml"


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    y_idx = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
    return frame[y_idx[:, None], x_idx[None, :]]


def _control_to_qpos_order_indices(robot_training: Any) -> tuple[np.ndarray, list[str]]:
    control_joint_names = list(robot_training.robot.control_joint_names)
    qpos_joint_names = sorted(control_joint_names, key=lambda joint: robot_training.robot.joint_qpos_addr[joint])
    qpos_addrs = [int(robot_training.robot.joint_qpos_addr[joint]) for joint in qpos_joint_names]
    if len(set(qpos_addrs)) != len(qpos_addrs):
        raise ValueError(f"Duplicate MuJoCo qpos addresses for control joints: {list(zip(qpos_joint_names, qpos_addrs))}")
    index_by_control_joint = {joint: idx for idx, joint in enumerate(control_joint_names)}
    return np.asarray([index_by_control_joint[joint] for joint in qpos_joint_names], dtype=np.int64), qpos_joint_names


def _expert_qpos_from_obs(
    obs_dict: dict[str, torch.Tensor],
    *,
    num_dof: int,
    dof_qpos_order_indices: np.ndarray,
) -> np.ndarray:
    root_pos = obs_dict["ref_body_pos"][:, 0].detach().cpu().numpy()
    root_quat_wxyz = np.roll(obs_dict["ref_body_rots"][:, 0].detach().cpu().numpy(), 1, axis=-1)
    # MotionLib stores dof_pos in policy/control-joint order. MuJoCo qpos playback
    # expects hinge joints sorted by qpos address, which can differ for robots such
    # as X2 where actuator order places head joints before arm joints.
    dof_pos = obs_dict["dof_pos"].detach().cpu().numpy()[:, dof_qpos_order_indices]
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=-1)
    expected = 7 + int(num_dof)
    if qpos.shape[-1] != expected:
        raise ValueError(f"Expected expert qpos shape (*, {expected}), got {qpos.shape}")
    return qpos


def _target_states_from_obs(obs_dict: dict[str, torch.Tensor], device: str, *, num_dof: int) -> dict[str, torch.Tensor]:
    root_state_xyzw = torch.cat(
        [
            obs_dict["ref_body_pos"][0, 0],
            obs_dict["ref_body_rots"][0, 0],
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ],
        dim=-1,
    ).to(device=device, dtype=torch.float32)
    dof_state = torch.zeros((int(num_dof), 2), device=device, dtype=torch.float32)
    dof_state[:, 0] = obs_dict["dof_pos"][0].to(device=device, dtype=torch.float32)
    dof_state[:, 1] = obs_dict["ref_dof_vel"][0].to(device=device, dtype=torch.float32)
    return {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}


def _center_target_states_on_terrain(
    target_states: dict[str, torch.Tensor],
    env: Any,
    *,
    stairs_start_step: int = 0,
) -> dict[str, torch.Tensor]:
    """Place a reference reset on its assigned terrain with correct local clearance."""
    if not bool(getattr(env, "terrain_enabled", False)):
        return target_states
    centered = dict(target_states)
    root_states = target_states["root_states"].clone()
    root_states[:, :2] = env.env_origins[: root_states.shape[0], :2].to(
        device=root_states.device,
        dtype=root_states.dtype,
    )
    # Motion root Z is clearance above the source motion plane.  Convert it to
    # terrain-local coordinates before reset_idx queries the exact support
    # height.  Omitting this shift on depressed RP1 tiles double-counts the
    # negative terrain origin and spawns the robot visibly above the ground.
    root_states[:, 2] += env.env_origins[: root_states.shape[0], 2].to(
        device=root_states.device,
        dtype=root_states.dtype,
    )
    if stairs_start_step:
        if not 1 <= int(stairs_start_step) <= RP1_STAIR_LEVELS:
            raise ValueError(
                f"stairs_start_step must be in [1,{RP1_STAIR_LEVELS}], got {stairs_start_step}"
            )
        root_states[:, 0] += (
            RP1_CENTER_PLATFORM_WIDTH / 2.0
            + (float(stairs_start_step) - 0.5) * RP1_STAIR_STEP_WIDTH
        )
    centered["root_states"] = root_states
    return centered


def _configure_tracking_terrain(env_cfg: Any, mode: str) -> Any:
    """Use a true plane for canonical motion tracking unless explicitly overridden."""

    if mode == "training":
        return env_cfg
    if mode in RP1_TERRAIN_COMPONENT_NAMES:
        overrides = list(env_cfg.hydra_overrides)
        overrides = replace_hydra_override(overrides, "terrain.terrain_type", "rp1_simple")
        return env_cfg.model_copy(update={"hydra_overrides": overrides, "evaluation_fast_path": False})
    if mode != "plane":
        raise ValueError(f"Unsupported tracking terrain mode: {mode!r}")
    overrides = list(env_cfg.hydra_overrides)
    overrides = replace_hydra_override(overrides, "terrain.terrain_type", "plane")
    overrides = replace_hydra_override(overrides, "terrain.terrain_priv.mode", "flat_zero")
    return env_cfg.model_copy(update={"hydra_overrides": overrides, "evaluation_fast_path": False})


def _assign_rp1_tracking_tile(env: Any, family: str, difficulty_row: int) -> None:
    """Assign the one-env rollout to an exact RP1 family/difficulty tile."""

    if family not in RP1_TERRAIN_COMPONENT_NAMES:
        return
    if tuple(env.terrain_component_names) != tuple(RP1_TERRAIN_COMPONENT_NAMES):
        raise RuntimeError(
            f"Expected RP1 terrain families {RP1_TERRAIN_COMPONENT_NAMES}, "
            f"got {env.terrain_component_names}"
        )
    terrain = env.mjlab_env.scene["terrain"]
    rows = int(terrain.terrain_origins.shape[0])
    if not 0 <= int(difficulty_row) < rows:
        raise ValueError(f"rp1_difficulty_row must be in [0,{rows}), got {difficulty_row}")
    family_id = RP1_TERRAIN_COMPONENT_NAMES.index(family)
    terrain.terrain_levels.fill_(int(difficulty_row))
    terrain.terrain_types.fill_(int(family_id))
    env.env_origins.copy_(
        terrain.terrain_origins[int(difficulty_row), family_id].expand_as(env.env_origins)
    )


@torch.no_grad()
def _tracking_z(model: torch.nn.Module, obs: Any) -> torch.Tensor:
    z = model.backward_map(obs)
    seq_length = int(getattr(getattr(model, "cfg", None), "seq_length", 1))
    if seq_length < 1:
        raise ValueError(f"model.cfg.seq_length must be positive, got {seq_length}")
    for step in range(z.shape[0]):
        end_idx = min(step + seq_length, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    return model.project_z(z)


def _tracking_heading_relative(obs_dict: dict[str, torch.Tensor], context_length: int) -> torch.Tensor:
    """Return motion-relative heading commands aligned one-to-one with tracking z."""

    reference = root_heading_xy(obs_dict["ref_body_rots"][:, 0].float())
    if reference.shape[0] < context_length + 1:
        raise ValueError(
            "Reference heading sequence is too short for tracking context: "
            f"reference={reference.shape[0]}, context={context_length}"
        )
    relative = rotation_between_heading_xy(reference[0:1].expand_as(reference), reference)
    # Match training's exact-tracking contract: z is B(next_observation), while
    # the Actor heading observation at that action step uses the current-frame
    # reference heading carried by the same replay transition.
    return relative[:context_length]


def _save_tracking_context(
    path: Path,
    *,
    z: torch.Tensor,
    obs_dict: dict[str, torch.Tensor],
    dt: float,
    motion_id: int,
) -> None:
    heading_relative = _tracking_heading_relative(obs_dict, int(z.shape[0]))
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int64),
        motion_id=np.asarray(int(motion_id), dtype=np.int64),
        dt=np.asarray(float(dt), dtype=np.float32),
        z=z.detach().cpu().float().numpy(),
        heading_relative_xy=heading_relative.detach().cpu().float().numpy(),
        heading_valid=np.ones((int(z.shape[0]), 1), dtype=np.bool_),
    )


def _export_policy_model(model: torch.nn.Module, output_dir: Path, robot_training: Any) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.__class__.__name__
    output_name = f"{model_name}.onnx"
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    export_metadata = export_meta_policy_as_onnx(
        model,
        output_dir,
        output_name,
        z_dim=model.cfg.archi.z_dim,
    )
    if int(export_metadata["output_action_dim"]) != num_dof:
        raise ValueError(
            "Policy action dim does not match robot control joint count: "
            f"output_action_dim={export_metadata['output_action_dim']}, num_dof={num_dof}"
        )
    export_metadata.update(
        {
            "robot_name": robot_training.robot.name,
            "robot_config_path": str(Path(robot_training.config_path).expanduser().resolve()),
            "xml_path": str(Path(robot_training.robot.xml_path).expanduser().resolve()),
            "num_dof": num_dof,
            "control_joint_names": control_joint_names,
        }
    )
    metadata_path = output_dir / f"{model_name}.meta.json"
    metadata_path.write_text(json.dumps(export_metadata, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Exported model to {output_dir / output_name}")
    print(f"[INFO] Wrote policy ONNX metadata to {metadata_path}")
    return export_metadata


def _space_shape(model: torch.nn.Module, key: str) -> tuple[int, ...]:
    spaces = getattr(getattr(model, "obs_space", None), "spaces", None)
    if spaces is None or key not in spaces:
        raise KeyError(f"Missing observation space for ONNX input {key!r}")
    shape = tuple(int(value) for value in spaces[key].shape)
    if not shape:
        raise ValueError(f"Observation {key!r} must have a non-empty shape")
    return shape


def _is_direct_depth_model(model: torch.nn.Module) -> bool:
    actor = getattr(model, "_actor", None)
    return bool(
        actor is not None
        and hasattr(actor, "depth_encoder")
        and hasattr(actor, "forward_from_depth_latent")
        and hasattr(getattr(actor, "cfg", None), "depth_key")
    )


def _export_direct_depth_policy_model(
    model: torch.nn.Module,
    output_dir: Path,
    robot_training: Any,
    *,
    opset_version: int = 13,
) -> dict[str, Any]:
    """Export the deployable direct-depth policy as encoder and actor ONNX files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    actor = cpu_model._actor
    actor_keys_raw = actor.input_filter.cfg.key
    actor_keys = [actor_keys_raw] if isinstance(actor_keys_raw, str) else list(actor_keys_raw)
    expected_keys = ["state", "last_action", "history_actor"]
    if bool(getattr(cpu_model.cfg, "heading_context_enabled", False)):
        expected_keys.append("heading")
    if actor_keys != expected_keys:
        raise ValueError(
            "Direct-depth ONNX actor input contract mismatch: "
            f"checkpoint keys={actor_keys}, expected={expected_keys}"
        )
    actor_input_dims: dict[str, int] = {}
    for key in actor_keys:
        shape = _space_shape(cpu_model, key)
        if len(shape) != 1:
            raise ValueError(f"Actor ONNX input {key!r} must be 1D, got {shape}")
        actor_input_dims[key] = shape[0]

    depth_key = str(actor.cfg.depth_key)
    depth_shape = _space_shape(cpu_model, depth_key)
    expected_depth_shape = (int(actor.cfg.depth_channels), int(actor.cfg.depth_height), int(actor.cfg.depth_width))
    if depth_shape != expected_depth_shape:
        raise ValueError(f"Depth ONNX input shape mismatch: obs_space={depth_shape}, actor={expected_depth_shape}")
    depth_latent_dim = int(actor.cfg.depth_latent_dim)
    z_dim = int(cpu_model.cfg.archi.z_dim)
    action_dim = int(cpu_model.action_dim)
    control_joint_names = list(robot_training.robot.control_joint_names)
    if action_dim != len(control_joint_names):
        raise ValueError(
            f"Policy action dim {action_dim} does not match robot control joint count {len(control_joint_names)}"
        )

    class DepthEncoderWrapper(torch.nn.Module):
        def __init__(self, depth_encoder: torch.nn.Module):
            super().__init__()
            self.depth_encoder = depth_encoder

        def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
            return self.depth_encoder(depth_image).float()

    class DepthActorWrapper(torch.nn.Module):
        def __init__(self, inference_model: torch.nn.Module, input_keys: list[str]):
            super().__init__()
            self.obs_normalizer = inference_model._obs_normalizer
            self.actor = inference_model._actor
            self.actor_std = float(inference_model.cfg.actor_std)
            self.input_keys = list(input_keys)

        def forward(
            self,
            state: torch.Tensor,
            last_action: torch.Tensor,
            history_actor: torch.Tensor,
            heading: torch.Tensor,
            depth_latent: torch.Tensor,
            z: torch.Tensor,
        ) -> torch.Tensor:
            values = (state, last_action, history_actor, heading)
            obs = {key: value for key, value in zip(self.input_keys, values, strict=True)}
            normalized_obs = self.obs_normalizer(obs)
            return self.actor.forward_from_depth_latent(
                normalized_obs,
                z,
                self.actor_std,
                depth_latent,
            ).mean.float()

    if "heading" not in actor_keys:
        raise ValueError("This deployment export requires the trained 2D heading observation")

    depth_path = output_dir / "depth_encoder_ufo.onnx"
    actor_path = output_dir / "depth_actor_ufo.onnx"
    depth_example = torch.zeros((1, *depth_shape), dtype=torch.float32)
    actor_examples = tuple(
        torch.zeros((1, actor_input_dims[key]), dtype=torch.float32) for key in actor_keys
    ) + (
        torch.zeros((1, depth_latent_dim), dtype=torch.float32),
        torch.zeros((1, z_dim), dtype=torch.float32),
    )

    torch.onnx.export(
        DepthEncoderWrapper(actor.depth_encoder),
        depth_example,
        depth_path,
        verbose=False,
        input_names=["depth_image"],
        output_names=["depth_latent"],
        dynamic_axes={"depth_image": {0: "batch"}, "depth_latent": {0: "batch"}},
        opset_version=int(opset_version),
    )
    actor_input_names = actor_keys + ["depth_latent", "z"]
    torch.onnx.export(
        DepthActorWrapper(cpu_model, actor_keys),
        actor_examples,
        actor_path,
        verbose=False,
        input_names=actor_input_names,
        output_names=["action"],
        dynamic_axes={name: {0: "batch"} for name in actor_input_names + ["action"]},
        opset_version=int(opset_version),
    )

    metadata = {
        "format": "ufo_direct_depth_split_v1",
        "actor_model": actor_path.name,
        "depth_encoder_model": depth_path.name,
        "actor_input_keys": actor_keys,
        "actor_input_dims": actor_input_dims,
        "actor_input_names": actor_input_names,
        "depth_input_name": "depth_image",
        "depth_input_shape": list(depth_shape),
        "depth_latent_name": "depth_latent",
        "depth_latent_dim": depth_latent_dim,
        "z_dim": z_dim,
        "output_name": "action",
        "output_action_dim": action_dim,
        "heading_semantics": "[1-cos(target-current), sin(target-current)]",
        "robot_name": robot_training.robot.name,
        "robot_config_path": str(Path(robot_training.config_path).expanduser().resolve()),
        "xml_path": str(Path(robot_training.robot.xml_path).expanduser().resolve()),
        "num_dof": len(control_joint_names),
        "control_joint_names": control_joint_names,
    }
    metadata_path = output_dir / "depth_actor_ufo.meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Exported direct-depth encoder to {depth_path}")
    print(f"[INFO] Exported direct-depth actor to {actor_path}")
    print(f"[INFO] Wrote direct-depth ONNX metadata to {metadata_path}")
    return metadata


def _export_tracking_onnx(model: torch.nn.Module, output_dir: Path, robot_training: Any) -> None:
    if _is_direct_depth_model(model):
        _export_direct_depth_policy_model(model, output_dir, robot_training)
    else:
        _export_policy_model(model, output_dir, robot_training)
    try:
        export_backward_encoder_from_model(model, output_dir / "backward_encoder.onnx")
    except UnsupportedBackwardEncoderExport as exc:
        print(f"[INFO] Skip backward encoder ONNX export: {exc}")


def _resolve_tracking_robot_config(
    cli_robot_config: str | Path | None,
    manifest_robot_config: str | Path | None,
) -> Path:
    if cli_robot_config is not None and manifest_robot_config is not None:
        return assert_robot_configs_compatible(cli_robot_config, manifest_robot_config)
    if cli_robot_config is not None:
        return resolve_robot_config_path(cli_robot_config)
    if manifest_robot_config is not None:
        return resolve_robot_config_path(manifest_robot_config)
    return resolve_robot_config_path(DEFAULT_ROBOT_CONFIG)


def run_tracking_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    robot_config: Path | None,
    headless: bool,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    motion_list: list[int],
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    fps: int,
    max_steps: int | None,
    log_every_steps: int,
    max_episode_length_s: float,
    export_onnx: bool,
    tracking_terrain: str = "plane",
    rp1_difficulty_row: int = 5,
    stairs_start_step: int = 0,
    actor_override: Path | None = None,
    perception_checkpoint: Path | None = None,
) -> None:
    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    robot_config = _resolve_tracking_robot_config(robot_config, None)
    robot_training = load_robot_training_spec(robot_config)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    if not robot_xml.exists():
        raise FileNotFoundError(f"Missing robot XML: {robot_xml}")
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    dof_qpos_order_indices, qpos_joint_names = _control_to_qpos_order_indices(robot_training)

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()
    actor_override_info = load_actor_override(model, actor_override) if actor_override is not None else None
    if actor_override_info is not None:
        print(f"[INFO] Loaded read-only Actor override: {actor_override_info}")

    if export_onnx:
        _export_tracking_onnx(model, model_folder / "exported", robot_training)

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=device,
        headless=headless,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=max_episode_length_s,
    )
    env_cfg = _configure_tracking_terrain(env_cfg, tracking_terrain)
    perception_runtime = (
        TemporalDepthTerrainRuntime(
            env_cfg,
            perception_checkpoint=perception_checkpoint,
            device=device,
        )
        if perception_checkpoint is not None
        else None
    )
    wrapped_env = perception_runtime.wrapped_env if perception_runtime is not None else env_cfg.build(num_envs=1)[0]
    env = wrapped_env._env
    _assign_rp1_tracking_tile(env, tracking_terrain, rp1_difficulty_row)

    output_dir = model_folder / "tracking_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] UFO tracking inference model_folder={model_folder}")
    print(f"[INFO] Rollout XML={env_cfg.mjcf_path}")
    print(f"[INFO] Motion data={env_cfg.lafan_tail_path}")
    print(f"[INFO] Expert renderer XML={robot_xml}")
    print(f"[INFO] Tracking terrain={tracking_terrain}")
    if tracking_terrain in {"low_stairs_up", "low_stairs_down"}:
        print(
            f"[INFO] RP1 stair placement: difficulty_row={rp1_difficulty_row}, "
            f"start_step={stairs_start_step}"
        )
    if qpos_joint_names != control_joint_names:
        print(f"[INFO] Expert qpos joint order differs from control order: {qpos_joint_names}")
    print(f"[INFO] device={device} disable_dr={disable_dr} disable_obs_noise={disable_obs_noise} save_mp4={save_mp4}")

    env._motion_lib.load_all_motions()
    env.is_evaluating = True
    expert_renderer = (
        MujocoQposRenderer(
            robot_xml,
            render_size=render_size,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            expected_qpos_size=7 + num_dof,
        )
        if save_mp4
        else None
    )
    policy_renderer = (
        MujocoQposRenderer(
            None,
            render_size=render_size,
            scene_spec=env.mjlab_env.scene.spec,
            source_xml_path=robot_xml,
            add_floor=False,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            expected_qpos_size=7 + num_dof,
        )
        if save_mp4
        else None
    )
    try:
        for motion_id in motion_list:
            backward_obs, obs_dict = get_backward_observation(env, motion_id, use_root_height_obs=use_root_height_obs)
            z = _tracking_z(model, tree_map(lambda x: x[1:].to(device) if hasattr(x, "to") else x, backward_obs))
            joblib.dump(z.detach().cpu().numpy(), output_dir / f"zs_{motion_id}.pkl")
            print(f"[INFO] Saved z embedding: {output_dir / f'zs_{motion_id}.pkl'}")
            context_path = output_dir / f"tracking_context_{motion_id}.npz"
            _save_tracking_context(
                context_path,
                z=z,
                obs_dict=obs_dict,
                dt=float(env.dt),
                motion_id=motion_id,
            )
            print(f"[INFO] Saved deployable tracking context: {context_path}")

            target_states = _target_states_from_obs(obs_dict, device=device, num_dof=num_dof)
            target_states = _center_target_states_on_terrain(
                target_states,
                env,
                stairs_start_step=(
                    stairs_start_step
                    if tracking_terrain in {"low_stairs_up", "low_stairs_down"}
                    else 0
                ),
            )
            observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)
            heading_targets = None
            if bool(getattr(model.cfg, "heading_context_enabled", False)):
                reference_heading = root_heading_xy(
                    obs_dict["ref_body_rots"][:, 0].to(device).float()
                )
                heading_targets = align_heading_sequence(
                    reference_heading.unsqueeze(0),
                    root_heading_xy(env.base_quat.float()),
                    torch.zeros(1, device=device, dtype=torch.long),
                )[0][: z.shape[0]]
            if perception_runtime is not None:
                perception_runtime.reset()
            episode_len = int(z.shape[0])
            if max_steps is not None:
                episode_len = min(episode_len, int(max_steps))
            expert_qpos = _expert_qpos_from_obs(
                obs_dict,
                num_dof=num_dof,
                dof_qpos_order_indices=dof_qpos_order_indices,
            )
            frames: list[np.ndarray] = []

            print(f"[INFO] Running policy rollout for motion_id={motion_id}, steps={episode_len}", flush=True)
            for step in range(episode_len):
                if perception_runtime is not None:
                    observation["terrain_actor"] = perception_runtime.terrain_actor(
                        observation,
                        reset_mask=torch.ones(1, device=device, dtype=torch.bool) if step == 0 else None,
                    )
                if heading_targets is not None:
                    target = heading_targets[min(step, heading_targets.shape[0] - 1)].unsqueeze(0)
                    valid = torch.ones((1, 1), device=device, dtype=torch.bool)
                    observation["heading"] = heading_observation(
                        root_heading_xy(env.base_quat.float()),
                        target,
                        valid,
                    )
                action = model.act(observation, z[step].unsqueeze(0), mean=True)
                observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                if perception_runtime is not None:
                    reset = torch.as_tensor(terminated, device=device).bool() | torch.as_tensor(truncated, device=device).bool()
                    perception_runtime.after_step(reset)

                if save_mp4:
                    policy_qpos = policy_qpos_from_env(
                        wrapped_env,
                        expected_qpos_size=policy_renderer.input_nq,
                    )
                    policy_frame = policy_renderer.render_qpos(policy_qpos)
                    expert_frame = expert_renderer.render_qpos(expert_qpos[min(step + 1, len(expert_qpos) - 1)])
                    expert_frame = _resize_nearest(expert_frame, policy_frame.shape[0], policy_frame.shape[1])
                    frames.append(np.concatenate([expert_frame, policy_frame], axis=1))

                if step == 0 or (step + 1) == episode_len or (log_every_steps > 0 and (step + 1) % log_every_steps == 0):
                    print(f"[INFO] motion_id={motion_id} rollout/render progress {step + 1}/{episode_len}", flush=True)

                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    print(f"[INFO] Episode ended at step={step}; stopping rollout for motion_id={motion_id}")
                    break

            if save_mp4:
                video_path = output_dir / f"tracking_{motion_id}.mp4"
                if not frames:
                    raise RuntimeError(f"No frames were rendered for motion_id={motion_id}")
                media.write_video(str(video_path), frames, fps=fps)
                print(f"[INFO] Saved side-by-side video: {video_path}")
    finally:
        if expert_renderer is not None:
            expert_renderer.close()
        if policy_renderer is not None:
            policy_renderer.close()
        if perception_runtime is not None:
            perception_runtime.close()
        else:
            wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFO tracking inference with MuJoCo expert rendering.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=None, help="Robot YAML for rollout and expert rendering.")
    parser.add_argument("--data-manifest", type=Path, default=None, help="Motion data manifest. Use with --dataset.")
    parser.add_argument("--dataset", default=None, help="Dataset name inside --data-manifest for tracking inference.")
    parser.add_argument("--rebuild-motion-cache", action="store_true", help="Rebuild manifest-generated motion pkl cache.")
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save side-by-side expert/policy MP4.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--motion-list", type=int, nargs="+", default=[20])
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rollout/video frames for quick previews.")
    parser.add_argument(
        "--log-every-steps", type=int, default=100, help="Print rollout/render progress every N steps; 0 disables periodic logs."
    )
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    parser.add_argument(
        "--tracking-terrain",
        choices=("plane", "training", "low_stairs_up", "low_stairs_down"),
        default="plane",
        help=(
            "Canonical tracking uses plane. Use low_stairs_down for the exact "
            "RP1 ascent-from-center staircase or low_stairs_up for descent."
        ),
    )
    parser.add_argument(
        "--rp1-difficulty-row",
        type=int,
        default=5,
        help="RP1 curriculum row [0,9] used by an explicitly selected terrain family.",
    )
    parser.add_argument(
        "--stairs-start-step",
        type=int,
        default=0,
        help="Place the policy reset on this RP1 stair band; 0 keeps the center platform.",
    )
    parser.add_argument(
        "--actor-override",
        type=Path,
        default=None,
        help="Actor-only milestone to load in memory; the source full checkpoint is never modified.",
    )
    parser.add_argument(
        "--perception-checkpoint",
        type=Path,
        default=None,
        help="Optional temporal terrain checkpoint. When set, clean depth replaces GT terrain_actor.",
    )
    add_bool_arg(parser, "--export-onnx", True, "Export ONNX next to the checkpoint before inference.")
    args = parser.parse_args()
    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        if args.dataset is None:
            parser.error("--dataset is required when --data-manifest is provided")
        manifest_robot_config = prepare_manifest_robot_config_path(args.data_manifest)
        args.data_path = Path(
            prepare_manifest_dataset_path(
                args.data_manifest,
                args.dataset,
                split="inference",
                rebuild_cache=bool(args.rebuild_motion_cache),
            )
        )
    args.robot_config = _resolve_tracking_robot_config(args.robot_config, manifest_robot_config)
    return args


def main() -> None:
    args = parse_args()
    run_tracking_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,
        headless=args.headless,
        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        motion_list=args.motion_list,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        fps=args.fps,
        max_steps=args.max_steps,
        log_every_steps=args.log_every_steps,
        max_episode_length_s=args.max_episode_length_s,
        export_onnx=args.export_onnx,
        tracking_terrain=args.tracking_terrain,
        rp1_difficulty_row=args.rp1_difficulty_row,
        stairs_start_step=args.stairs_start_step,
        actor_override=args.actor_override,
        perception_checkpoint=args.perception_checkpoint,
    )


if __name__ == "__main__":
    main()
