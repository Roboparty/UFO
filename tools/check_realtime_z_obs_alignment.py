#!/usr/bin/env python3
"""
Compare BFM-Zero offline backward observations with the deploy realtime-z path.

This tool does not touch robot control. It replays one offline motion through:

1. BFM-Zero training/offline get_backward_observation()
2. UFO-Deploy scripts/realtime/realtime_z_server.py FK + obs helpers

Both observation streams are fed to the same exported backward_encoder.onnx, then
state, privileged_state, and z are compared frame by frame.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(av, bv) / denom)


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_max(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")


def _safe_p95(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 95)) if arr.size else float("nan")


def _state_slices(state_dim: int) -> Dict[str, slice]:
    if state_dim != 64:
        raise ValueError(f"Expected 29-DoF state dim 64, got {state_dim}")
    return {
        "state.dof_pos": slice(0, 29),
        "state.dof_vel": slice(29, 58),
        "state.projected_gravity": slice(58, 61),
        "state.root_ang_vel": slice(61, 64),
    }


def _privileged_slices(priv_dim: int, num_bodies: int, root_height_obs: bool) -> Dict[str, slice]:
    expected = (1 if root_height_obs else 0) + (num_bodies - 1) * 3 + num_bodies * 6 + num_bodies * 3 + num_bodies * 3
    if priv_dim != expected:
        raise ValueError(
            f"privileged_state dim mismatch: got {priv_dim}, expected {expected} "
            f"(num_bodies={num_bodies}, root_height_obs={root_height_obs})"
        )

    out: Dict[str, slice] = {}
    pos = 0
    if root_height_obs:
        out["priv.root_height"] = slice(pos, pos + 1)
        pos += 1
    out["priv.local_body_pos"] = slice(pos, pos + (num_bodies - 1) * 3)
    pos = out["priv.local_body_pos"].stop
    out["priv.local_body_rot"] = slice(pos, pos + num_bodies * 6)
    pos = out["priv.local_body_rot"].stop
    out["priv.local_body_vel"] = slice(pos, pos + num_bodies * 3)
    pos = out["priv.local_body_vel"].stop
    out["priv.local_body_ang_vel"] = slice(pos, pos + num_bodies * 3)
    return out


def _infer_root_height_obs(priv_dim: int, num_bodies: int) -> bool:
    no_height = (num_bodies - 1) * 3 + num_bodies * 6 + num_bodies * 3 + num_bodies * 3
    with_height = no_height + 1
    if priv_dim == with_height:
        return True
    if priv_dim == no_height:
        return False
    raise ValueError(
        f"Cannot infer root_height_obs from privileged_state dim {priv_dim}; "
        f"expected {no_height} or {with_height} for {num_bodies} bodies"
    )


def _slice_metrics(prefix: str, offline: np.ndarray, realtime: np.ndarray, slices: Mapping[str, slice]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for name, sl in slices.items():
        a = offline[sl]
        b = realtime[sl]
        key = name if name.startswith(prefix) else f"{prefix}.{name}"
        row[f"{key}.l2"] = _l2(a, b)
        row[f"{key}.mae"] = _mae(a, b)
        row[f"{key}.max_abs"] = _max_abs(a, b)
        row[f"{key}.cosine"] = _cosine(a, b)
    return row


def _summarize_rows(rows: List[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and isinstance(r[key], (int, float, np.floating))]
        summary[key] = {
            "mean": _safe_mean(vals),
            "p95": _safe_p95(vals),
            "max": _safe_max(vals),
        }
    return summary


def _top_error_sources(summary: Mapping[str, Mapping[str, float]], suffix: str, limit: int) -> List[Dict[str, float]]:
    items: List[Tuple[str, float]] = []
    for key, stats in summary.items():
        if key.endswith(suffix):
            value = float(stats.get("mean", float("nan")))
            if np.isfinite(value):
                items.append((key, value))
    items.sort(key=lambda x: x[1], reverse=True)
    return [{"metric": k, "mean": v} for k, v in items[:limit]]


def _top_frames_by_metric(rows: Sequence[Mapping[str, Any]], metric: str, top_k: int, reverse: bool = True) -> List[int]:
    valid = [r for r in rows if metric in r and np.isfinite(float(r[metric]))]
    valid.sort(key=lambda r: float(r[metric]), reverse=reverse)
    return [int(r["frame"]) for r in valid[:top_k]]


def _quat_continuity_summary(
    realtime: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    body_names: Sequence[str],
    top_k: int,
) -> Dict[str, Any]:
    dots = np.asarray(realtime.get("quat_pair_dot", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    flip_mask = np.asarray(realtime.get("quat_flip_mask", np.zeros_like(dots, dtype=bool)), dtype=bool)
    if dots.size == 0:
        return {
            "total_pairs": 0,
            "flip_count": 0,
            "flip_frame_count": 0,
            "min_dot": None,
            "events": [],
            "body_counts": [],
            "overlap_top_k": {},
        }

    frame_ids, body_ids = np.where(flip_mask)
    frame_ids = frame_ids + 1
    flip_frames = {int(x) for x in frame_ids.tolist()}
    compared_flip_frames = {int(r["frame"]) for r in rows if int(r.get("quat.flip_count", 0)) > 0}

    body_counts: List[Dict[str, Any]] = []
    for body_i in range(flip_mask.shape[1]):
        count = int(np.count_nonzero(flip_mask[:, body_i]))
        if count > 0:
            name = body_names[body_i] if body_i < len(body_names) else str(body_i)
            body_counts.append({"body_id": int(body_i), "body": name, "count": count})
    body_counts.sort(key=lambda x: int(x["count"]), reverse=True)

    events: List[Dict[str, Any]] = []
    for frame_i, body_i in zip(frame_ids.tolist(), body_ids.tolist()):
        name = body_names[body_i] if body_i < len(body_names) else str(body_i)
        events.append({
            "frame": int(frame_i),
            "body_id": int(body_i),
            "body": name,
            "dot": float(dots[int(frame_i) - 1, int(body_i)]),
        })
    events.sort(key=lambda x: x["dot"])

    overlap_metrics = [
        ("raw.body_ang_vel.l2", True),
        ("priv.local_body_ang_vel.l2", True),
        ("state.root_ang_vel.l2", True),
        ("z.l2", True),
        ("z.cosine", False),
    ]
    overlap: Dict[str, Any] = {}
    for metric, reverse in overlap_metrics:
        top_frames = _top_frames_by_metric(rows, metric, top_k, reverse=reverse)
        overlap_frames = [f for f in top_frames if f in compared_flip_frames]
        overlap[metric] = {
            "top_frames": top_frames,
            "overlap_frames": overlap_frames,
            "overlap_count": len(overlap_frames),
        }

    return {
        "total_pairs": int(dots.size),
        "flip_count": int(np.count_nonzero(flip_mask)),
        "flip_frame_count": int(len(flip_frames)),
        "compared_flip_frame_count": int(len(compared_flip_frames)),
        "min_dot": float(np.min(dots)),
        "p01_dot": float(np.percentile(dots, 1)),
        "body_counts": body_counts,
        "events": events[: max(0, int(top_k))],
        "overlap_top_k": overlap,
    }


class RealtimeObsBuilder:
    def __init__(
        self,
        realtime_module: Any,
        mujoco_xml: Path,
        root_height_obs: bool,
        fixed_dt: float,
        fix_quat_continuity: bool = False,
        angvel_delta_frame: str = "local",
    ) -> None:
        import mujoco

        self.rzs = realtime_module
        self.root_height_obs = bool(root_height_obs)
        self.fixed_dt = float(fixed_dt)
        self.fix_quat_continuity = bool(fix_quat_continuity)
        if str(angvel_delta_frame) not in ("local", "world"):
            raise ValueError(f"angvel_delta_frame must be local or world, got {angvel_delta_frame!r}")
        self.angvel_delta_frame = str(angvel_delta_frame)
        self.mj_model = mujoco.MjModel.from_xml_path(str(mujoco_xml))
        self.mj_data = mujoco.MjData(self.mj_model)

        self.base_body_names = list(self.rzs.G1_29DOF_BODY_NAMES)
        self.extend_cfg = [dict(c) for c in self.rzs.G1_29DOF_EXTEND_CONFIG]
        self.default_pose_29 = np.asarray(self.rzs.G1_29DOF_DEFAULT_POSE_RAD, dtype=np.float32)

        body_ids: List[int] = []
        for name in self.base_body_names:
            try:
                body_ids.append(self.mj_model.body(name).id)
            except KeyError as exc:
                raise KeyError(f"Body {name!r} not found in MuJoCo XML {mujoco_xml}") from exc
        self.body_ids = np.asarray(body_ids, dtype=np.int32)
        self.body_names = self.base_body_names + [str(ext["joint_name"]) for ext in self.extend_cfg]
        self.num_bodies = len(self.body_names)

    def _fk_all_bodies(
        self,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        dof_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        qpos = np.zeros(self.mj_model.nq, dtype=np.float64)
        qpos[0:3] = root_pos.astype(np.float64)
        qpos[3:7] = root_quat_wxyz.astype(np.float64)
        qpos[7:] = dof_pos.astype(np.float64)
        base_pos, base_quat_xyzw = self.rzs._fk_bodies_from_qpos(self.mj_model, self.mj_data, self.body_ids, qpos)

        all_names = self.base_body_names.copy()
        all_body_pos = base_pos
        all_body_rot_xyzw = base_quat_xyzw

        for ext in self.extend_cfg:
            parent_idx = all_names.index(ext["parent_name"])
            pos_in_parent = np.asarray(ext["pos"], dtype=np.float64).reshape(3)
            rot_xyzw = self.rzs._wxyz_to_xyzw(np.asarray(ext["rot"], dtype=np.float64).reshape(4))
            parent_pos = all_body_pos[parent_idx]
            parent_rot = all_body_rot_xyzw[parent_idx]

            ext_pos = parent_pos + self.rzs._quat_rotate_xyzw(parent_rot[None, :], pos_in_parent[None, :])[0]
            ext_rot = self.rzs._quat_mul_xyzw(parent_rot[None, :], rot_xyzw[None, :])[0]

            all_body_pos = np.concatenate([all_body_pos, ext_pos[None, :]], axis=0)
            all_body_rot_xyzw = np.concatenate([all_body_rot_xyzw, ext_rot[None, :]], axis=0)
            all_names.append(ext["joint_name"])

        return all_body_pos.astype(np.float64), all_body_rot_xyzw.astype(np.float64)

    def build_sequence(
        self,
        root_pos: np.ndarray,
        root_quat_xyzw: np.ndarray,
        dof_pos: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        n = int(root_pos.shape[0])
        if root_quat_xyzw.shape[0] != n or dof_pos.shape[0] != n:
            raise ValueError("root_pos, root_quat_xyzw, and dof_pos must have the same length")

        all_pos: List[np.ndarray] = []
        all_rot: List[np.ndarray] = []
        for i in range(n):
            root_quat_wxyz = root_quat_xyzw[i, [3, 0, 1, 2]]
            pos_i, rot_i = self._fk_all_bodies(root_pos[i], root_quat_wxyz, dof_pos[i])
            all_pos.append(pos_i)
            all_rot.append(rot_i)

        body_pos = np.stack(all_pos, axis=0).astype(np.float32)
        body_rot = np.stack(all_rot, axis=0).astype(np.float32)

        body_vel = np.zeros_like(body_pos, dtype=np.float32)
        dof_vel = np.zeros_like(dof_pos, dtype=np.float32)
        body_ang_vel = np.zeros((n, body_rot.shape[1], 3), dtype=np.float32)
        quat_pair_dot = np.zeros((max(0, n - 1), body_rot.shape[1]), dtype=np.float32)
        quat_flip_mask = np.zeros_like(quat_pair_dot, dtype=bool)

        if n > 1:
            body_vel[1:] = (body_pos[1:] - body_pos[:-1]) / self.fixed_dt
            dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / self.fixed_dt
            quat_pair_dot = self.rzs._quat_pair_dot_xyzw(body_rot[:-1], body_rot[1:]).astype(np.float32)
            quat_flip_mask = quat_pair_dot < 0.0
            body_ang_vel[1:] = self.rzs._quat_to_ang_vel_xyzw(
                body_rot[:-1],
                body_rot[1:],
                self.fixed_dt,
                fix_quat_continuity=self.fix_quat_continuity,
                delta_frame=self.angvel_delta_frame,
            ).astype(np.float32)

        obs_dict_priv = self.rzs._compute_humanoid_observations_max_np(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            local_root_obs=True,
            root_height_obs=self.root_height_obs,
        )
        privileged_state = np.concatenate([v.astype(np.float32) for v in obs_dict_priv.values()], axis=-1).astype(np.float32)

        ref_dof_pos = (dof_pos.astype(np.float32) - self.default_pose_29[None, :]).astype(np.float32)
        projected_gravity = self.rzs._quat_rotate_inverse_xyzw(
            body_rot[:, 0, :],
            np.repeat(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), n, axis=0),
        ).astype(np.float32)
        ref_ang_vel = body_ang_vel[:, 0, :].astype(np.float32)
        state = np.concatenate([ref_dof_pos, dof_vel.astype(np.float32), projected_gravity, ref_ang_vel], axis=-1).astype(np.float32)

        return {
            "state": state,
            "privileged_state": privileged_state,
            "body_pos": body_pos,
            "body_rot": body_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "quat_pair_dot": quat_pair_dot,
            "quat_flip_mask": quat_flip_mask,
            "dof_pos": dof_pos.astype(np.float32),
            "dof_vel": dof_vel.astype(np.float32),
        }


def _load_offline_motion_obs(
    bfm_zero_root: Path,
    model_folder: Path,
    data_path: Optional[Path],
    motion_id: int,
    simulator: str,
    device: str,
    root_height_obs: bool,
    max_episode_length_s: float,
    disable_dr: bool,
    disable_obs_noise: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    sys.path.insert(0, str(bfm_zero_root))

    import json as _json
    import torch
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
    from humanoidverse.utils.helpers import get_backward_observation

    model_folder = model_folder.resolve()
    config_path = model_folder / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"model config.json not found: {config_path}")

    with config_path.open("r") as f:
        config = _json.load(f)

    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(data_path.resolve())
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = bfm_zero_root / "humanoidverse" / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)

    config["env"].setdefault("hydra_overrides", [])
    config["env"]["hydra_overrides"].append(f"env.config.max_episode_length_s={float(max_episode_length_s)}")
    config["env"]["hydra_overrides"].append("env.config.headless=True")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = bool(disable_dr)
    config["env"]["disable_obs_noise"] = bool(disable_obs_noise)

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(1)
    env = wrapped_env._env
    env.set_is_evaluating(int(motion_id))

    obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=bool(root_height_obs))
    offline = {
        "state": _to_numpy(obs["state"]).astype(np.float32),
        "privileged_state": _to_numpy(obs["privileged_state"]).astype(np.float32),
        "root_pos": _to_numpy(obs_dict["ref_body_pos"])[:, 0, :].astype(np.float32),
        "root_quat_xyzw": _to_numpy(obs_dict["ref_body_rots"])[:, 0, :].astype(np.float32),
        "dof_pos": _to_numpy(obs_dict["dof_pos"]).astype(np.float32),
        "ref_dof_pos": _to_numpy(obs_dict["ref_dof_pos"]).astype(np.float32),
        "ref_dof_vel": _to_numpy(obs_dict["ref_dof_vel"]).astype(np.float32),
        "projected_gravity": _to_numpy(obs_dict["projected_gravity"]).astype(np.float32),
        "ref_ang_vel": _to_numpy(obs_dict["ref_ang_vel"]).astype(np.float32),
        "body_pos": _to_numpy(obs_dict["ref_body_pos"]).astype(np.float32),
        "body_rot": _to_numpy(obs_dict["ref_body_rots"]).astype(np.float32),
        "body_vel": _to_numpy(obs_dict["ref_body_vels"]).astype(np.float32),
        "body_ang_vel": _to_numpy(obs_dict["ref_body_angular_vels"]).astype(np.float32),
    }

    meta = {
        "env_dt": float(getattr(env, "dt")),
        "motion_id": int(motion_id),
        "lafan_tail_path": config["env"].get("lafan_tail_path"),
        "root_height_obs": bool(root_height_obs),
        "offline_num_frames": int(offline["state"].shape[0]),
    }

    try:
        # Avoid holding simulator resources longer than needed in repeated runs.
        del wrapped_env
        del env
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return offline, meta


def _load_onnx_session(backward_onnx: Path, provider_device: str):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider_device.startswith("cuda") else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(backward_onnx.resolve()), providers=providers)
    input_names = {x.name for x in session.get_inputs()}
    output_names = [x.name for x in session.get_outputs()]
    output_name = "z" if "z" in output_names else output_names[0]
    return session, input_names, output_name


def _run_encoder(session: Any, input_names: set[str], output_name: str, state: np.ndarray, privileged_state: np.ndarray) -> np.ndarray:
    feed: Dict[str, np.ndarray] = {}
    if "state" in input_names:
        feed["state"] = state.astype(np.float32)
    if "privileged_state" in input_names:
        feed["privileged_state"] = privileged_state.astype(np.float32)
    if "last_action" in input_names:
        # Current z10 ONNX drops this input, but keep a deterministic fallback for other exports.
        feed["last_action"] = state[:, 0:29].astype(np.float32)
    return np.asarray(session.run([output_name], feed)[0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root_from_this_file()
    parser = argparse.ArgumentParser(description="Check realtime z observation alignment against offline BFM-Zero observations.")
    parser.add_argument("--bfm-zero-root", type=Path, default=repo_root.parent / "BFM-Zero", help="Path to BFM-Zero training repo.")
    parser.add_argument("--deploy-root", type=Path, default=repo_root, help="Path to UFO-Deploy repo.")
    parser.add_argument("--model-folder", type=Path, required=True, help="Training run folder containing config.json.")
    parser.add_argument("--backward-onnx", type=Path, default=None, help="Exported backward_encoder.onnx. Defaults to model-folder/exported/backward_encoder.onnx.")
    parser.add_argument("--mujoco-xml", type=Path, default=None, help="Deploy MuJoCo XML used by realtime_z_server.")
    parser.add_argument("--motion-id", type=int, required=True, help="Motion id passed to env.set_is_evaluating().")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional lafan_tail_path override.")
    parser.add_argument("--simulator", type=str, default="mujoco", help="Hydra simulator override for the offline env.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for building offline env/model-side code.")
    parser.add_argument("--onnx-device", type=str, default="cpu", help="cpu|cuda providers for backward_encoder.onnx.")
    parser.add_argument("--root-height-obs", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--start-frame", type=int, default=1, help="First frame to compare. Default 1 skips realtime velocity warmup.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum compared frames; 0 means all available.")
    parser.add_argument("--dt", type=float, default=0.0, help="Realtime finite-difference dt override. Default uses offline env.dt.")
    parser.add_argument("--fix-quat-continuity", action="store_true", help="Flip q_curr when dot(q_prev, q_curr) < 0 before realtime angular-velocity differencing.")
    parser.add_argument("--angvel-delta-frame", choices=["local", "world"], default="local", help="Quaternion delta order for angular velocity: local=inv(q_prev)*q_curr, world=q_curr*inv(q_prev).")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/realtime_z_obs_alignment"))
    parser.add_argument("--summary-name", type=str, default="summary.json")
    parser.add_argument("--csv-name", type=str, default="frame_metrics.csv")
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    parser.add_argument("--allow-domain-randomization", action="store_true", help="Do not force disable_domain_randomization=True.")
    parser.add_argument("--allow-obs-noise", action="store_true", help="Do not force disable_obs_noise=True.")
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deploy_root = args.deploy_root.resolve()
    backward_onnx = args.backward_onnx or (args.model_folder / "exported" / "backward_encoder.onnx")
    mujoco_xml = args.mujoco_xml or (deploy_root / "data" / "robots" / "g1" / "scene_29dof_freebase.xml")
    realtime_path = deploy_root / "scripts" / "realtime" / "realtime_z_server.py"

    if not realtime_path.exists():
        raise FileNotFoundError(f"realtime_z_server.py not found: {realtime_path}")
    if not backward_onnx.exists():
        raise FileNotFoundError(f"backward_encoder.onnx not found: {backward_onnx}")
    if not mujoco_xml.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {mujoco_xml}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    rzs = _load_module_from_path("deploy_realtime_z_server_for_alignment", realtime_path)
    session, input_names, output_name = _load_onnx_session(backward_onnx, args.onnx_device)

    num_bodies = len(rzs.G1_29DOF_BODY_NAMES) + len(rzs.G1_29DOF_EXTEND_CONFIG)
    onnx_inputs = [(x.name, list(x.shape), x.type) for x in session.get_inputs()]
    priv_dim_from_onnx = None
    for name, shape, _typ in onnx_inputs:
        if name == "privileged_state" and len(shape) >= 2 and isinstance(shape[1], int):
            priv_dim_from_onnx = int(shape[1])
    if priv_dim_from_onnx is None:
        raise ValueError(f"Cannot infer privileged_state dim from ONNX inputs: {onnx_inputs}")

    if args.root_height_obs == "auto":
        root_height_obs = _infer_root_height_obs(priv_dim_from_onnx, num_bodies)
    else:
        root_height_obs = args.root_height_obs == "true"

    offline, meta = _load_offline_motion_obs(
        bfm_zero_root=args.bfm_zero_root.resolve(),
        model_folder=args.model_folder.resolve(),
        data_path=args.data_path,
        motion_id=args.motion_id,
        simulator=args.simulator,
        device=args.device,
        root_height_obs=root_height_obs,
        max_episode_length_s=args.max_episode_length_s,
        disable_dr=not args.allow_domain_randomization,
        disable_obs_noise=not args.allow_obs_noise,
    )

    dt = float(args.dt) if args.dt > 0 else float(meta["env_dt"])
    realtime_builder = RealtimeObsBuilder(
        rzs,
        mujoco_xml.resolve(),
        root_height_obs=root_height_obs,
        fixed_dt=dt,
        fix_quat_continuity=bool(args.fix_quat_continuity),
        angvel_delta_frame=str(args.angvel_delta_frame),
    )
    realtime = realtime_builder.build_sequence(offline["root_pos"], offline["root_quat_xyzw"], offline["dof_pos"])

    n_total = min(offline["state"].shape[0], realtime["state"].shape[0])
    start = max(0, int(args.start_frame))
    end = n_total if args.max_frames <= 0 else min(n_total, start + int(args.max_frames))
    if start >= end:
        raise ValueError(f"No frames to compare: start={start}, end={end}, n_total={n_total}")

    offline_state = offline["state"][start:end]
    offline_priv = offline["privileged_state"][start:end]
    realtime_state = realtime["state"][start:end]
    realtime_priv = realtime["privileged_state"][start:end]

    offline_z = _run_encoder(session, input_names, output_name, offline_state, offline_priv)
    realtime_z = _run_encoder(session, input_names, output_name, realtime_state, realtime_priv)

    state_fields = _state_slices(offline_state.shape[1])
    priv_fields = _privileged_slices(offline_priv.shape[1], num_bodies, root_height_obs)
    raw_fields = {
        "raw.dof_vel": ("ref_dof_vel", "dof_vel"),
        "raw.body_vel": ("body_vel", "body_vel"),
        "raw.body_ang_vel": ("body_ang_vel", "body_ang_vel"),
    }

    rows: List[Dict[str, Any]] = []
    for local_i, frame_idx in enumerate(range(start, end)):
        row: Dict[str, Any] = {
            "frame": int(frame_idx),
            "state.l2": _l2(offline_state[local_i], realtime_state[local_i]),
            "state.mae": _mae(offline_state[local_i], realtime_state[local_i]),
            "state.max_abs": _max_abs(offline_state[local_i], realtime_state[local_i]),
            "state.cosine": _cosine(offline_state[local_i], realtime_state[local_i]),
            "privileged_state.l2": _l2(offline_priv[local_i], realtime_priv[local_i]),
            "privileged_state.mae": _mae(offline_priv[local_i], realtime_priv[local_i]),
            "privileged_state.max_abs": _max_abs(offline_priv[local_i], realtime_priv[local_i]),
            "privileged_state.cosine": _cosine(offline_priv[local_i], realtime_priv[local_i]),
            "z.l2": _l2(offline_z[local_i], realtime_z[local_i]),
            "z.mae": _mae(offline_z[local_i], realtime_z[local_i]),
            "z.max_abs": _max_abs(offline_z[local_i], realtime_z[local_i]),
            "z.cosine": _cosine(offline_z[local_i], realtime_z[local_i]),
            "z.offline_norm": float(np.linalg.norm(offline_z[local_i])),
            "z.realtime_norm": float(np.linalg.norm(realtime_z[local_i])),
            "z.norm_absdiff": float(abs(np.linalg.norm(offline_z[local_i]) - np.linalg.norm(realtime_z[local_i]))),
        }
        if frame_idx > 0 and (frame_idx - 1) < realtime["quat_pair_dot"].shape[0]:
            quat_dots = realtime["quat_pair_dot"][frame_idx - 1]
            quat_flips = realtime["quat_flip_mask"][frame_idx - 1]
            row["quat.flip_count"] = int(np.count_nonzero(quat_flips))
            row["quat.min_dot"] = float(np.min(quat_dots))
            row["quat.root_dot"] = float(quat_dots[0])
        else:
            row["quat.flip_count"] = 0
            row["quat.min_dot"] = 1.0
            row["quat.root_dot"] = 1.0
        row.update(_slice_metrics("state", offline_state[local_i], realtime_state[local_i], state_fields))
        row.update(_slice_metrics("priv", offline_priv[local_i], realtime_priv[local_i], priv_fields))

        for metric_name, (offline_key, realtime_key) in raw_fields.items():
            a = offline[offline_key][frame_idx]
            b = realtime[realtime_key][frame_idx]
            row[f"{metric_name}.l2"] = _l2(a, b)
            row[f"{metric_name}.mae"] = _mae(a, b)
            row[f"{metric_name}.max_abs"] = _max_abs(a, b)
            row[f"{metric_name}.cosine"] = _cosine(a, b)
        rows.append(row)

    metric_keys = sorted({k for row in rows for k, v in row.items() if k != "frame" and isinstance(v, (int, float, np.floating))})
    summary_metrics = _summarize_rows(rows, metric_keys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / args.csv_name
    json_path = args.output_dir / args.summary_name

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame"] + metric_keys)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "inputs": {
            "bfm_zero_root": str(args.bfm_zero_root.resolve()),
            "deploy_root": str(deploy_root),
            "model_folder": str(args.model_folder.resolve()),
            "backward_onnx": str(backward_onnx.resolve()),
            "mujoco_xml": str(mujoco_xml.resolve()),
            "motion_id": int(args.motion_id),
            "data_path": str(args.data_path.resolve()) if args.data_path else None,
            "simulator": args.simulator,
            "device": args.device,
            "onnx_device": args.onnx_device,
            "fix_quat_continuity": bool(args.fix_quat_continuity),
            "angvel_delta_frame": str(args.angvel_delta_frame),
        },
        "onnx": {
            "inputs": onnx_inputs,
            "output_name": output_name,
            "actual_input_names": sorted(input_names),
        },
        "alignment": {
            "root_height_obs": bool(root_height_obs),
            "num_bodies": int(num_bodies),
            "env_dt": float(meta["env_dt"]),
            "realtime_dt": float(dt),
            "start_frame": int(start),
            "end_frame_exclusive": int(end),
            "num_compared_frames": int(end - start),
            "offline_num_frames": int(meta["offline_num_frames"]),
            "fix_quat_continuity": bool(args.fix_quat_continuity),
            "angvel_delta_frame": str(args.angvel_delta_frame),
        },
        "metrics": summary_metrics,
        "top_l2_error_sources": _top_error_sources(summary_metrics, ".l2", args.top_k),
        "top_mae_error_sources": _top_error_sources(summary_metrics, ".mae", args.top_k),
        "quat_continuity": _quat_continuity_summary(
            realtime,
            rows,
            realtime_builder.body_names,
            args.top_k,
        ),
        "watchlist": {
            "dof_vel": summary_metrics.get("state.dof_vel.l2"),
            "body_vel": summary_metrics.get("priv.local_body_vel.l2"),
            "body_ang_vel": summary_metrics.get("priv.local_body_ang_vel.l2"),
            "raw_dof_vel": summary_metrics.get("raw.dof_vel.l2"),
            "raw_body_vel": summary_metrics.get("raw.body_vel.l2"),
            "raw_body_ang_vel": summary_metrics.get("raw.body_ang_vel.l2"),
            "z_cosine": summary_metrics.get("z.cosine"),
            "z_l2": summary_metrics.get("z.l2"),
        },
        "files": {
            "frame_metrics_csv": str(csv_path),
            "summary_json": str(json_path),
        },
    }

    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[alignment] wrote CSV: {csv_path}")
    print(f"[alignment] wrote JSON: {json_path}")
    print("[alignment] key metrics:")
    for key in [
        "state.l2",
        "state.dof_vel.l2",
        "privileged_state.l2",
        "priv.local_body_vel.l2",
        "priv.local_body_ang_vel.l2",
        "z.l2",
        "z.cosine",
        "z.offline_norm",
        "z.realtime_norm",
    ]:
        stats = summary_metrics.get(key)
        if stats:
            print(f"  {key}: mean={stats['mean']:.6g}, p95={stats['p95']:.6g}, max={stats['max']:.6g}")


if __name__ == "__main__":
    main()
