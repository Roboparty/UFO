"""Expert trajectory loading and content-addressed full-FK cache helpers."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from humanoidverse.envs.env_utils.history_handler import HistoryHandler as HVHistoryHandler
from humanoidverse.envs.motion_observations import compute_humanoid_observations_max
from humanoidverse.utils.reference_observations import reference_base_ang_vel
from humanoidverse.utils.storage_paths import expert_buffer_cache_parent
from humanoidverse.utils.torch_utils import quat_rotate_inverse

from ..buffers.trajectory import TrajectoryDictBuffer

EXPERT_BUFFER_CACHE_VERSION = 2
_MOTION_LIB_CACHE_TENSOR_FIELDS = (
    "_motion_lengths",
    "_motion_fps",
    "_motion_bodies",
    "_motion_aa",
    "_motion_dt",
    "_motion_num_frames",
    "_motion_smpl_poses",
    "_motion_actions",
    "gts",
    "grs",
    "lrs",
    "grvs",
    "gravs",
    "gavs",
    "gvs",
    "dvs",
    "gts_t",
    "grs_t",
    "gvs_t",
    "gavs_t",
    "dof_pos",
    "length_starts",
    "motion_ids",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _file_identity(path: str | Path, *, hash_contents: bool = False) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    identity: dict[str, Any] = {"path": str(resolved)}
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        identity["missing"] = True
    else:
        identity.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        if hash_contents:
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            identity["sha256"] = digest.hexdigest()
    return identity


def _motion_cache_metadata(motion_lib) -> dict[str, Any]:
    motion_file = motion_lib.m_cfg.motion_file
    motion_files = [motion_file] if isinstance(motion_file, (str, bytes, Path)) else list(motion_file)
    asset = motion_lib.m_cfg.asset
    skeleton_path = Path(str(asset.assetRoot)) / str(asset.assetFileName)
    motion_keys = [str(key) for key in motion_lib._motion_data_keys.tolist()]
    motion_config = _jsonable(motion_lib.m_cfg)
    return {
        "motion_files": [_file_identity(path) for path in motion_files],
        "robot_xml": _file_identity(skeleton_path, hash_contents=True),
        "motion_count": int(motion_lib._num_unique_motions),
        "motion_keys_sha256": hashlib.sha256("\0".join(motion_keys).encode("utf-8")).hexdigest(),
        "motion_config": motion_config,
        "body_mapping": {
            "body_names": motion_config.get("body_names"),
            "dof_names": motion_config.get("dof_names"),
        },
        "extend_bodies": motion_config.get("extend_config"),
    }


def _expert_observation_schema(env) -> dict[str, Any]:
    privileged = getattr(env, "_max_local_self", None)
    schema: dict[str, Any] = {
        "observation": {
            "state": 2 * int(env.num_dof) + 6,
            "last_action": int(env.num_dof),
            "privileged_state": int(privileged.shape[-1]) if isinstance(privileged, torch.Tensor) else None,
        },
        "terminated": "bool",
        "truncated": "bool",
        "motion_id": "int64",
    }
    creation_config = getattr(env, "_creation_config", None)
    if creation_config is not None and bool(getattr(creation_config, "include_history_noaction", False)):
        schema["observation"]["history_noaction"] = _jsonable(env.config.obs.obs_auxiliary.get("history_actor"))
    return schema


def _cache_dependencies(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_version": metadata.get("cache_version"),
        "motion_files": metadata.get("motion_files"),
        "robot_xml": metadata.get("robot_xml"),
        "motion_count": metadata.get("motion_count"),
        "motion_keys_sha256": metadata.get("motion_keys_sha256"),
        "motion_config": metadata.get("motion_config"),
        "body_mapping": metadata.get("body_mapping"),
        "extend_bodies": metadata.get("extend_bodies"),
        "step_dt": metadata.get("step_dt"),
        "control_frequency_hz": metadata.get("control_frequency_hz"),
        "seq_length": metadata.get("seq_length"),
        "default_dof_pos": metadata.get("default_dof_pos"),
        "observation_schema": metadata.get("observation_schema"),
        "observation_config": metadata.get("observation_config"),
    }


def expert_buffer_cache_spec(env, agent_cfg, cache_root: str | Path | None = None) -> tuple[Path, str, dict[str, Any]]:
    """Return a cache destination keyed by all inputs that affect expert tensors or FK."""
    metadata = {
        "cache_version": EXPERT_BUFFER_CACHE_VERSION,
        **_motion_cache_metadata(env._motion_lib),
        "step_dt": float(env.dt),
        "control_frequency_hz": 1.0 / float(env.dt),
        "seq_length": int(agent_cfg.model.seq_length),
        "default_dof_pos": env.default_dof_pos[0].detach().cpu().tolist(),
        "observation_schema": _expert_observation_schema(env),
        "observation_config": _jsonable(env.config.obs),
    }
    encoded = json.dumps(_cache_dependencies(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    root = expert_buffer_cache_parent(cache_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / "expert_buffers" / f"v{EXPERT_BUFFER_CACHE_VERSION}-{timestamp}-{fingerprint[:20]}", fingerprint, metadata


def _read_cache_metadata(cache_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def expert_buffer_cache_is_ready(cache_dir: Path, fingerprint: str) -> bool:
    metadata = _read_cache_metadata(cache_dir)
    return bool(
        metadata is not None
        and metadata.get("fingerprint") == fingerprint
        and metadata.get("cache_version") == EXPERT_BUFFER_CACHE_VERSION
        and (cache_dir / "expert_buffer.pt").is_file()
        and (cache_dir / "motion_lib_fk.pt").is_file()
    )


def find_compatible_expert_buffer_cache(
    expected_dir: Path,
    expected_fingerprint: str,
    expected_metadata: Mapping[str, Any],
) -> tuple[Path, str] | None:
    if expert_buffer_cache_is_ready(expected_dir, expected_fingerprint):
        return expected_dir, expected_fingerprint
    expected_dependencies = _cache_dependencies(expected_metadata)
    candidates = sorted(
        expected_dir.parent.glob(f"v{EXPERT_BUFFER_CACHE_VERSION}-*"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in candidates:
        metadata = _read_cache_metadata(candidate)
        if metadata is None or _cache_dependencies(metadata) != expected_dependencies:
            continue
        fingerprint = metadata.get("fingerprint")
        if isinstance(fingerprint, str) and expert_buffer_cache_is_ready(candidate, fingerprint):
            return candidate, fingerprint
    return None


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _motion_lib_cache_state(motion_lib) -> dict[str, Any]:
    tensors = {
        name: getattr(motion_lib, name)
        for name in _MOTION_LIB_CACHE_TENSOR_FIELDS
        if isinstance(getattr(motion_lib, name, None), torch.Tensor)
    }
    required = {"_motion_lengths", "_motion_fps", "_motion_num_frames", "gts", "grs", "lrs", "length_starts"}
    missing = sorted(required.difference(tensors))
    if missing:
        raise RuntimeError(f"Cannot cache expert MotionLib state; missing tensors: {missing}")
    return {
        "tensors": tensors,
        "num_motions": int(motion_lib._num_motions),
        "num_bodies": int(motion_lib.num_bodies),
        "num_joints": int(motion_lib.num_joints),
    }


def save_expert_buffer_cache(
    env,
    expert_buffer: TrajectoryDictBuffer,
    cache_dir: Path,
    fingerprint: str,
    metadata: dict[str, Any],
) -> None:
    """Publish expert trajectories and full FK atomically; metadata is the ready marker."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ready_path = cache_dir / "metadata.json"
    try:
        ready_path.unlink()
    except FileNotFoundError:
        pass
    _atomic_torch_save(
        {"cache_version": EXPERT_BUFFER_CACHE_VERSION, "fingerprint": fingerprint, "buffer": expert_buffer.cache_state_dict()},
        cache_dir / "expert_buffer.pt",
    )
    _atomic_torch_save(
        {"cache_version": EXPERT_BUFFER_CACHE_VERSION, "fingerprint": fingerprint, "motion_lib": _motion_lib_cache_state(env._motion_lib)},
        cache_dir / "motion_lib_fk.pt",
    )
    temporary = ready_path.with_name(f".{ready_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump({"fingerprint": fingerprint, **metadata}, stream, indent=2, sort_keys=True)
        os.replace(temporary, ready_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _motion_lib_has_full_fk(motion_lib) -> bool:
    required = ("_motion_lengths", "_motion_num_frames", "gts", "grs", "lrs", "length_starts")
    return (
        bool(getattr(motion_lib, "all_motions_loaded", False))
        and int(getattr(motion_lib, "_num_motions", -1)) == int(motion_lib._num_unique_motions)
        and all(isinstance(getattr(motion_lib, name, None), torch.Tensor) for name in required)
    )


def _restore_motion_lib_cache(motion_lib, payload: Mapping[str, Any], fingerprint: str) -> None:
    if payload.get("cache_version") != EXPERT_BUFFER_CACHE_VERSION or payload.get("fingerprint") != fingerprint:
        raise ValueError("Cached MotionLib FK metadata does not match")
    state = payload["motion_lib"]
    for name, value in state["tensors"].items():
        setattr(motion_lib, name, value.to(device=motion_lib._device))
    motion_lib._num_motions = int(state["num_motions"])
    motion_lib.num_bodies = int(state["num_bodies"])
    motion_lib.num_joints = int(state["num_joints"])
    motion_lib._curr_motion_ids = torch.arange(motion_lib._num_unique_motions, device=motion_lib._device)
    motion_lib.curr_motion_keys = motion_lib._motion_data_keys.tolist()
    motion_lib.all_motions_loaded = True
    motion_lib._refresh_sampling_batch_prob()


def load_expert_buffer_cache(
    env,
    cache_dir: Path,
    fingerprint: str,
    device: str = "cpu",
) -> TrajectoryDictBuffer:
    if not expert_buffer_cache_is_ready(cache_dir, fingerprint):
        raise FileNotFoundError(f"Expert buffer cache is incomplete or stale: {cache_dir}")
    expert_payload = torch.load(cache_dir / "expert_buffer.pt", map_location="cpu", weights_only=True, mmap=True)
    motion_payload = None
    if not _motion_lib_has_full_fk(env._motion_lib):
        motion_payload = torch.load(cache_dir / "motion_lib_fk.pt", map_location="cpu", weights_only=True, mmap=True)
    for payload, label in ((expert_payload, "expert buffer"), (motion_payload, "MotionLib FK")):
        if payload is not None and (
            payload.get("cache_version") != EXPERT_BUFFER_CACHE_VERSION or payload.get("fingerprint") != fingerprint
        ):
            raise ValueError(f"Cached {label} metadata does not match {cache_dir}")
    if motion_payload is not None:
        _restore_motion_lib_cache(env._motion_lib, motion_payload, fingerprint)
    expert_buffer = TrajectoryDictBuffer.from_cache_state_dict(expert_payload["buffer"], device=device)
    if len(expert_buffer.motion_ids) != env._motion_lib._num_unique_motions:
        raise ValueError(
            "Cached expert buffer motion count does not match MotionLib: "
            f"expert={len(expert_buffer.motion_ids)} motion_lib={env._motion_lib._num_unique_motions}"
        )
    del expert_payload, motion_payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return expert_buffer


def load_expert_trajectories_from_motion_lib(env, agent_cfg, device="cpu", add_history_noaction: bool = False):
    """Load expert trajectories directly from an environment motion library."""
    env._motion_lib.load_motions_for_training()
    episodes = []
    file_names = []
    history_handler = HVHistoryHandler(1, env.config.obs.obs_auxiliary, env.config.obs.obs_dims, device)
    history_config = env.config.obs.obs_auxiliary["history_actor"]
    for i in range(env._motion_lib._num_unique_motions):
        motion_times = torch.arange(int(np.ceil((env._motion_lib._motion_lengths[i] / env.dt).cpu()))).to(env.device) * env.dt
        motion_id = torch.tensor([i]).to(env.device).repeat(motion_times.shape[0])
        motion_res = env._motion_lib.get_motion_state(motion_id, motion_times)
        file_names.append(env._motion_lib._motion_data_keys[i])

        ref_body_pos = motion_res["rg_pos_t"]
        ref_body_rots = motion_res["rg_rot_t"]
        ref_body_vels = motion_res["body_vel_t"]
        ref_body_angular_vels = motion_res["body_ang_vel_t"]

        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
        )
        max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

        base_quat = ref_body_rots[:, 0]
        ref_dof_pos = motion_res["dof_pos"] - env.default_dof_pos[0]
        ref_dof_vel = motion_res["dof_vel"]
        ref_ang_vel = reference_base_ang_vel(env, base_quat, ref_body_angular_vels[:, 0])
        projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True)
        bogus_actions = ref_dof_pos * 0

        state = torch.cat(
            [
                ref_dof_pos,
                ref_dof_vel,
                projected_gravity,
                ref_ang_vel,
            ],
            dim=-1,
        )

        data = {
            "base_ang_vel": ref_ang_vel,
            "projected_gravity": projected_gravity,
            "dof_pos": ref_dof_pos,
            "dof_vel": ref_dof_vel,
        }

        if add_history_noaction:
            history_handler.reset([0])
            history_actor = []
            for ii in range(state.shape[0]):
                history_tensors = []
                for key in sorted(history_config.keys()):
                    if key not in ["action", "actions"]:
                        history_length = history_config[key]
                        history_tensor = history_handler.query(key)[:, :history_length]
                        history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
                        history_tensors.append(history_tensor)
                history_tensors = torch.cat(history_tensors, dim=1)
                history_actor.append(history_tensors)

                for key in history_handler.history.keys():
                    if key not in ["action", "actions"]:
                        history_handler.add(key, data[key][ii][None, ...])
            history_actor = torch.stack(history_actor, dim=0).squeeze(1)

        curr_motion_len = state.shape[0]
        truncated = torch.zeros(curr_motion_len, dtype=bool).to(env.device)
        truncated[-1] = True

        assert state.shape[0] == curr_motion_len, f"{env._motion_lib._motion_data_keys[i]}: {state.shape[0]} vs {curr_motion_len}"
        assert max_local_self_obs.shape[0] == curr_motion_len, (
            f"{env._motion_lib._motion_data_keys[i]}: {max_local_self_obs.shape[0]} vs {curr_motion_len}"
        )
        assert bogus_actions.shape[0] == curr_motion_len, (
            f"{env._motion_lib._motion_data_keys[i]}: {bogus_actions.shape[0]} vs {curr_motion_len}"
        )
        assert truncated.shape[0] == curr_motion_len, f"{env._motion_lib._motion_data_keys[i]}: {truncated.shape[0]} vs {curr_motion_len}"
        if add_history_noaction:
            assert history_actor.shape[0] == curr_motion_len, (
                f"{env._motion_lib._motion_data_keys[i]}: {history_actor.shape[0]} vs {curr_motion_len}"
            )

        ep = {
            "observation": {
                "state": state,
                "last_action": bogus_actions,
                "privileged_state": max_local_self_obs,
            },
            "terminated": torch.zeros(curr_motion_len, dtype=bool).to(env.device),
            "truncated": truncated,
            "motion_id": torch.ones(curr_motion_len, dtype=torch.long) * i,
        }
        if add_history_noaction:
            ep["observation"]["history_noaction"] = history_actor
        episodes.append(ep)

    expert_buffer = TrajectoryDictBuffer(
        episodes=episodes,
        seq_length=agent_cfg.model.seq_length,
        device=device,
    )

    assert expert_buffer.storage["observation"]["state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["observation"]["last_action"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["observation"]["privileged_state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["terminated"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["motion_id"].shape[0] == expert_buffer.storage["truncated"].shape[0]

    expert_buffer.file_names = file_names
    return expert_buffer
