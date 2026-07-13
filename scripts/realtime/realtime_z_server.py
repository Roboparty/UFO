#!/usr/bin/env python3
"""
Realtime z server (deploy-only stack).

Pipeline:
    PICO/GMR teleop pose (JSON over ZMQ)
        -> MuJoCo forward kinematics (numpy)
        -> obs construction (state, privileged_state)
        -> backward_encoder.onnx
        -> z float32[256]
        -> PUB to tracking policy (ZMQ)

Runtime dependencies: numpy, mujoco, onnxruntime, pyzmq.

No imports from humanoidverse / pico_npz_to_z_stream / pytorch.
The required quaternion + FK + observation helpers are vendored locally below
to mirror the math used during training.
"""

from __future__ import annotations

import argparse
import atexit
import json
import select
import sys
import termios
import threading
import time
import tty
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zmq

# -----------------------------------------------------------------------------
# Robot constants (G1, 29-DOF)
# -----------------------------------------------------------------------------
# Must match the body order used while training the latent policy. These names come from
# humanoidverse robot config (g1_29dof_hard_waist.yaml: robot.motion.body_names
# and robot.motion.extend_config).
G1_29DOF_BODY_NAMES: List[str] = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
]

# Extra "extend" bodies computed analytically from a parent body pose (not in
# MuJoCo model). Order matters: appended after the base body list.
G1_29DOF_EXTEND_CONFIG: List[Dict[str, Any]] = [
    {
        "joint_name": "head_link",
        "parent_name": "torso_link",
        "pos": [0.0, 0.0, 0.35],
        "rot": [1.0, 0.0, 0.0, 0.0],  # wxyz convention as in the training yaml
    },
]

# Default 29-DoF "stand" pose used to center dof_pos in state/last_action.
# Order: 6 left leg, 6 right leg, 3 waist, 4 left shoulder/elbow, 3 left wrist,
# 4 right shoulder/elbow, 3 right wrist. Values come from the training XML's
# `stand` keyframe and must match the training-time dof_names order.
G1_29DOF_DEFAULT_POSE_RAD: np.ndarray = np.array(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg
        0.0, 0.0, 0.0,                    # waist
        0.0, 0.0, 0.0, 0.0,               # left shoulder/elbow
        0.0, 0.0, 0.0,                    # left wrist
        0.0, 0.0, 0.0, 0.0,               # right shoulder/elbow
        0.0, 0.0, 0.0,                    # right wrist
    ],
    dtype=np.float32,
)
assert G1_29DOF_DEFAULT_POSE_RAD.shape == (29,)


def _standing_z() -> np.ndarray:
    z = np.zeros(256, dtype=np.float32)
    z[0] = 16.0
    return z


# -----------------------------------------------------------------------------
# Quaternion + FK helpers (numpy)
# -----------------------------------------------------------------------------

def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return q[..., [1, 2, 3, 0]]


def _quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q).copy()
    out[..., 0:3] *= -1.0
    return out


def _quat_normalize_xyzw(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, eps, None)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton quaternion multiplication, batched, xyzw (w-last)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack([x, y, z, w], axis=-1)


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (xyzw). Assumes q is unit.

    Mirrors humanoidverse.utils.torch_utils.my_quat_rotate.
    """
    q_w = q[..., 3:4]
    q_vec = q[..., 0:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a + b + c


def _quat_rotate_inverse_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by inverse of quaternion q (xyzw). Assumes q is unit.

    Mirrors humanoidverse.utils.torch_utils.quat_rotate_inverse(w_last=True).
    """
    q_w = q[..., 3:4]
    q_vec = q[..., 0:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def _quat_from_angle_axis_xyzw(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Build unit quaternion (xyzw) from angle (radians) and unit axis."""
    half = 0.5 * np.asarray(angle)
    s = np.sin(half)[..., None]
    c = np.cos(half)[..., None]
    return np.concatenate([axis * s, c], axis=-1)


def _calc_heading_xyzw(q: np.ndarray) -> np.ndarray:
    """Heading angle (rotation around +Z) of a body's orientation."""
    ref_dir = np.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1.0
    rot_dir = _quat_rotate_xyzw(q, ref_dir)
    return np.arctan2(rot_dir[..., 1], rot_dir[..., 0])


def _calc_heading_quat_inv_xyzw(q: np.ndarray) -> np.ndarray:
    """Inverse heading rotation (around +Z) from quaternion."""
    heading = _calc_heading_xyzw(q)
    axis = np.zeros_like(q[..., 0:3])
    axis[..., 2] = 1.0
    return _quat_from_angle_axis_xyzw(-heading, axis)


def _quat_to_tan_norm_xyzw(q: np.ndarray) -> np.ndarray:
    """Encode rotation as concatenated (tangent, normal) 6D vector."""
    ref_tan = np.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1.0
    ref_norm = np.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1.0
    tan = _quat_rotate_xyzw(q, ref_tan)
    norm = _quat_rotate_xyzw(q, ref_norm)
    return np.concatenate([tan, norm], axis=-1)


def _quat_pair_dot_xyzw(q_prev: np.ndarray, q_curr: np.ndarray) -> np.ndarray:
    """Dot product between normalized xyzw quaternion pairs."""
    q_prev = _quat_normalize_xyzw(q_prev)
    q_curr = _quat_normalize_xyzw(q_curr)
    return np.sum(q_prev * q_curr, axis=-1)


def _quat_fix_continuity_xyzw(q_prev: np.ndarray, q_curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip q_curr to the same quaternion hemisphere as q_prev when needed."""
    q_prev = _quat_normalize_xyzw(q_prev)
    q_curr = _quat_normalize_xyzw(q_curr)
    dots = np.sum(q_prev * q_curr, axis=-1)
    flip_mask = dots < 0.0
    q_curr_fixed = np.where(flip_mask[..., None], -q_curr, q_curr)
    return q_curr_fixed, flip_mask, dots


def _quat_to_ang_vel_xyzw(
    q_prev: np.ndarray,
    q_curr: np.ndarray,
    dt: float,
    fix_quat_continuity: bool = False,
    delta_frame: str = "local",
) -> np.ndarray:
    """Angular velocity from a pair of unit quaternions (xyzw), small-angle stable."""
    q_prev = _quat_normalize_xyzw(q_prev)
    q_curr = _quat_normalize_xyzw(q_curr)
    if fix_quat_continuity:
        q_curr, _, _ = _quat_fix_continuity_xyzw(q_prev, q_curr)
    if str(delta_frame) == "world":
        dq = _quat_mul_xyzw(q_curr, _quat_conj_xyzw(q_prev))
    elif str(delta_frame) == "local":
        dq = _quat_mul_xyzw(_quat_conj_xyzw(q_prev), q_curr)
    else:
        raise ValueError(f"delta_frame must be 'local' or 'world', got {delta_frame!r}")
    dq = _quat_normalize_xyzw(dq)
    v = dq[..., 0:3]
    w = np.clip(dq[..., 3], -1.0, 1.0)
    angle = 2.0 * np.arctan2(np.linalg.norm(v, axis=-1), w)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = v / np.clip(v_norm, 1e-8, None)
    omega = (angle[..., None] / dt) * axis
    tiny = angle < 1e-6
    if np.any(tiny):
        omega[tiny] = (2.0 / dt) * v[tiny]
    return omega


def _fk_bodies_from_qpos(model, data, body_ids: np.ndarray, qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import mujoco  # lazy import so module import is cheap

    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    pos = data.xpos[body_ids].copy()
    quat_wxyz = data.xquat[body_ids].copy()
    return pos, _wxyz_to_xyzw(quat_wxyz)


# -----------------------------------------------------------------------------
# Privileged obs construction (numpy port of compute_humanoid_observations_max)
# -----------------------------------------------------------------------------

def _compute_humanoid_observations_max_np(
    body_pos: np.ndarray,        # (T, Nb, 3)
    body_rot: np.ndarray,        # (T, Nb, 4) xyzw
    body_vel: np.ndarray,        # (T, Nb, 3)
    body_ang_vel: np.ndarray,    # (T, Nb, 3)
    local_root_obs: bool,
    root_height_obs: bool,
) -> "OrderedDict[str, np.ndarray]":
    obs_dict: "OrderedDict[str, np.ndarray]" = OrderedDict()
    T, Nb, _ = body_pos.shape
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    root_h = root_pos[:, 2:3]

    heading_rot_inv = _calc_heading_quat_inv_xyzw(root_rot)  # (T, 4)

    if root_height_obs:
        obs_dict["root_height"] = root_h

    heading_rot_inv_expand = np.repeat(heading_rot_inv[:, None, :], Nb, axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(T * Nb, 4)

    local_body_pos = body_pos - root_pos[:, None, :]
    flat_local_body_pos = local_body_pos.reshape(T * Nb, 3)
    flat_local_body_pos = _quat_rotate_xyzw(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(T, Nb * 3)[..., 3:]  # drop root (=0)

    flat_body_rot = body_rot.reshape(T * Nb, 4)
    flat_local_body_rot = _quat_mul_xyzw(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = _quat_to_tan_norm_xyzw(flat_local_body_rot)  # (T*Nb, 6)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(T, Nb * 6)
    if not local_root_obs:
        root_rot_obs = _quat_to_tan_norm_xyzw(root_rot)
        local_body_rot_obs = local_body_rot_obs.copy()
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(T * Nb, 3)
    flat_local_body_vel = _quat_rotate_xyzw(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(T, Nb * 3)

    flat_body_ang_vel = body_ang_vel.reshape(T * Nb, 3)
    flat_local_body_ang_vel = _quat_rotate_xyzw(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(T, Nb * 3)

    obs_dict["local_body_pos"] = local_body_pos
    obs_dict["local_body_rot"] = local_body_rot_obs
    obs_dict["local_body_vel"] = local_body_vel
    obs_dict["local_body_ang_vel"] = local_body_ang_vel
    return obs_dict


# -----------------------------------------------------------------------------
# Teleop pose payload
# -----------------------------------------------------------------------------

@dataclass
class PoseFrame:
    root_pos: np.ndarray         # (3,)
    root_quat_wxyz: np.ndarray   # (4,)
    dof_pos: np.ndarray          # (29,)


def _extract_latest_frame(payload: Dict[str, Any]) -> Optional[PoseFrame]:
    frames = payload.get("frames", None)
    if not isinstance(frames, list) or len(frames) == 0:
        return None
    f = frames[-1]
    if not isinstance(f, dict):
        return None
    root_pos = f.get("root_pos", None)
    root_quat = f.get("root_quat", None)
    dof_pos = f.get("dof_pos", None)
    if not (isinstance(root_pos, list) and isinstance(root_quat, list) and isinstance(dof_pos, list)):
        return None
    if not (len(root_pos) == 3 and len(root_quat) == 4):
        return None
    if len(dof_pos) != 29:
        return None
    try:
        rp = np.asarray(root_pos, dtype=np.float32).reshape(3)
        rq = np.asarray(root_quat, dtype=np.float32).reshape(4)
        dq = np.asarray(dof_pos, dtype=np.float32).reshape(29)
    except (TypeError, ValueError):
        return None
    if not (
        np.all(np.isfinite(rp))
        and np.all(np.isfinite(rq))
        and np.all(np.isfinite(dq))
    ):
        return None
    n = float(np.linalg.norm(rq))
    if not np.isfinite(n) or n <= 1e-6:
        return None
    rq = (rq / n).astype(np.float32)
    return PoseFrame(root_pos=rp, root_quat_wxyz=rq, dof_pos=dq)


def _is_pose_stale(
    last_valid_pose_monotonic: Optional[float],
    max_pose_stale_s: Optional[float],
    now: Optional[float] = None,
) -> bool:
    if last_valid_pose_monotonic is None or max_pose_stale_s is None:
        return False
    now_s = time.monotonic() if now is None else float(now)
    return now_s - float(last_valid_pose_monotonic) > float(max_pose_stale_s)


def _quat_normalize_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _slerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _quat_normalize_wxyz(q0).astype(np.float64)
    q1 = _quat_normalize_wxyz(q1).astype(np.float64)
    t = float(np.clip(alpha, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize_wxyz(q0 + t * (q1 - q0))
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    if abs(sin_theta_0) < 1e-8:
        return _quat_normalize_wxyz(q0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return _quat_normalize_wxyz(s0 * q0 + s1 * q1)


def _interpolate_pose_frame(a: PoseFrame, b: PoseFrame, alpha: float) -> PoseFrame:
    t = float(np.clip(alpha, 0.0, 1.0))
    root_pos = (a.root_pos * (1.0 - t) + b.root_pos * t).astype(np.float32)
    root_quat = _slerp_quat_wxyz(a.root_quat_wxyz, b.root_quat_wxyz, t)
    dof_pos = (a.dof_pos * (1.0 - t) + b.dof_pos * t).astype(np.float32)
    return PoseFrame(root_pos=root_pos, root_quat_wxyz=root_quat, dof_pos=dof_pos)


@dataclass
class TimedPoseFrame:
    recv_ns: int
    pose: PoseFrame


class PoseFrameBuffer:
    def __init__(self, lookback_ms: float, window_ms: float) -> None:
        self.lookback_ns = int(max(0.0, float(lookback_ms)) * 1e6)
        self.window_ns = int(max(1.0, float(window_ms)) * 1e6)
        self.frames: List[TimedPoseFrame] = []
        self.last_info: Dict[str, Any] = {"mode": "no_data", "buffer_len": 0}

    def append(self, pose: PoseFrame, recv_ns: int) -> None:
        stamp = int(recv_ns)
        self.frames.append(TimedPoseFrame(recv_ns=stamp, pose=pose))
        cutoff = stamp - self.window_ns
        while self.frames and self.frames[0].recv_ns < cutoff:
            self.frames.pop(0)

    def sample(self, now_ns: int) -> Tuple[Optional[PoseFrame], Dict[str, Any]]:
        target_ns = int(now_ns) - self.lookback_ns
        frames = self.frames
        if not frames:
            info = {"mode": "no_data", "buffer_len": 0, "target_age_ms": self.lookback_ns / 1e6}
            self.last_info = info
            return None, info

        def _info(mode: str, older_ns: Optional[int], newer_ns: Optional[int], alpha: Optional[float]) -> Dict[str, Any]:
            return {
                "mode": mode,
                "buffer_len": len(frames),
                "target_age_ms": round((int(now_ns) - target_ns) / 1e6, 3),
                "older_age_ms": None if older_ns is None else round((int(now_ns) - int(older_ns)) / 1e6, 3),
                "newer_age_ms": None if newer_ns is None else round((int(now_ns) - int(newer_ns)) / 1e6, 3),
                "span_ms": None if older_ns is None or newer_ns is None else round((int(newer_ns) - int(older_ns)) / 1e6, 3),
                "alpha": alpha,
            }

        if len(frames) == 1:
            only = frames[0]
            info = _info("single_frame", only.recv_ns, only.recv_ns, None)
            self.last_info = info
            return only.pose, info

        if target_ns <= frames[0].recv_ns:
            first = frames[0]
            info = _info("fallback_oldest", first.recv_ns, first.recv_ns, None)
            self.last_info = info
            return first.pose, info

        if target_ns >= frames[-1].recv_ns:
            last = frames[-1]
            info = _info("fallback_latest", last.recv_ns, last.recv_ns, None)
            self.last_info = info
            return last.pose, info

        for i in range(1, len(frames)):
            older = frames[i - 1]
            newer = frames[i]
            if target_ns <= newer.recv_ns:
                span = int(newer.recv_ns - older.recv_ns)
                if span <= 0:
                    info = _info("degenerate_dt", newer.recv_ns, newer.recv_ns, None)
                    self.last_info = info
                    return newer.pose, info
                alpha = float(target_ns - older.recv_ns) / float(span)
                info = _info("interpolate", older.recv_ns, newer.recv_ns, alpha)
                self.last_info = info
                return _interpolate_pose_frame(older.pose, newer.pose, alpha), info

        last = frames[-1]
        info = _info("fallback_latest", last.recv_ns, last.recv_ns, None)
        self.last_info = info
        return last.pose, info


# -----------------------------------------------------------------------------
# Teleop mode switching
# -----------------------------------------------------------------------------

class ModeState:
    def __init__(self, initial_mode: str = "follow") -> None:
        self._mode = initial_mode
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            return self._mode

    def set(self, mode: str) -> bool:
        if mode not in ("follow", "freeze"):
            raise ValueError(f"Unsupported mode: {mode}")
        with self._lock:
            changed = mode != self._mode
            self._mode = mode
        return changed


def _set_mode_from_input(mode_state: ModeState, mode: str, source: str) -> None:
    if mode_state.set(mode):
        print(f"[realtime_z_server] mode={mode} source={source}", flush=True)


def _start_keyboard_control(mode_state: ModeState) -> None:
    if not sys.stdin.isatty():
        print(
            "[realtime_z_server] keyboard control requested but stdin is not a TTY; disabled",
            flush=True,
        )
        return

    def _keyboard_loop() -> None:
        fd = sys.stdin.fileno()
        old_attrs = None
        restored = threading.Event()

        def restore_terminal() -> None:
            if old_attrs is None or restored.is_set():
                return
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass
            restored.set()

        try:
            old_attrs = termios.tcgetattr(fd)
            atexit.register(restore_terminal)
            tty.setcbreak(fd)
            print(
                "[realtime_z_server] keyboard control enabled: f=follow, s=freeze",
                flush=True,
            )
            while True:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                ch = sys.stdin.read(1).lower()
                if ch == "f":
                    _set_mode_from_input(mode_state, "follow", "keyboard")
                elif ch == "s":
                    _set_mode_from_input(mode_state, "freeze", "keyboard")
        except Exception as e:
            print(f"[realtime_z_server] keyboard control stopped: {e}", flush=True)
        finally:
            restore_terminal()

    t = threading.Thread(target=_keyboard_loop, name="realtime-z-keyboard", daemon=True)
    t.start()


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime z publisher from teleop pose stream")
    p.add_argument("--teleop_req", type=str, default="tcp://127.0.0.1:28701",
                   help="Teleop pose request socket (PUSH -> server PULL)")
    p.add_argument("--teleop_rep", type=str, default="tcp://127.0.0.1:28702",
                   help="Teleop pose reply socket (server PUSH -> PULL)")
    p.add_argument("--teleop_ctrl", type=str, default="tcp://127.0.0.1:28703",
                   help="Teleop controller button socket (server PUSH -> PULL)")
    p.add_argument("--z_bind", type=str, default="tcp://*:28711",
                   help="ZMQ PUB bind address for the 256-D z stream")
    p.add_argument("--hz", type=float, default=50.0, help="Loop frequency in Hz")
    p.add_argument("--lookback_ms", type=float, default=15.0,
                   help="Ask teleop to sample t_req_ms - lookback_ms")
    p.add_argument("--enable-pose-buffer", action="store_true",
                   help="Buffer received teleop poses and sample a delayed, interpolated reference pose before FK.")
    p.add_argument("--pose-buffer-lookback-ms", type=float, default=40.0,
                   help="When --enable-pose-buffer is set, sample pose at now - this delay in ms.")
    p.add_argument("--pose-buffer-window-ms", type=float, default=500.0,
                   help="When --enable-pose-buffer is set, keep this much received pose history in ms.")
    p.add_argument("--dry_run", action="store_true",
                   help="Do not publish z; only print pose stats")

    p.add_argument("--backward_onnx", type=str, default="",
                   help="ONNX file containing backward_map + project_z. Required unless --dry_run.")
    p.add_argument("--mujoco_xml", type=str, default="",
                   help="G1 29-DoF MuJoCo XML used for FK (must define 'stand' keyframe).")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu|cuda (controls onnxruntime providers)")
    p.add_argument("--root_height_obs", action="store_true",
                   help="Include root height in privileged obs (must match training).")
    p.add_argument("--drop_first_n", type=int, default=1,
                   help="Skip the first N frames before publishing z (matches offline behavior).")
    p.add_argument("--wall-clock-dt", action="store_true",
                   help="Use wall time between consecutive step() calls for finite differences.")
    p.add_argument("--fix-quat-continuity", action="store_true",
                   help="Before angular-velocity finite differencing, flip q_curr when dot(q_prev, q_curr) < 0. Default off for A/B checks.")
    p.add_argument("--angvel-delta-frame", choices=["local", "world"], default="local",
                   help="Quaternion delta order for angular velocity: local=inv(q_prev)*q_curr, world=q_curr*inv(q_prev). Default keeps old behavior.")
    p.add_argument("--no-freeze-static-pose", action="store_true",
                   help="By default we skip backward_map when teleop repeats the same pose; "
                        "pass this flag to always recompute z.")
    p.add_argument("--debug-z", action="store_true",
                   help="Print max|Δz| about once per second when z changes.")
    p.add_argument("--max-retarget-age-ms", type=float, default=200.0,
                   help="Drop teleop replies older than this age (ms). Set <0 to disable.")
    p.add_argument("--max-z-delta", type=float, default=0.75,
                   help="Per-step clamp on abs(z_t - z_{t-1}). Set <=0 to disable.")
    p.add_argument("--enable-keyboard-control", action="store_true",
                   help="Enable PC keyboard mode switching: f=follow, s=freeze.")
    p.add_argument("--enable-pico-control", action="store_true",
                   help="Enable PICO controller mode switching via teleop_ctrl: A/right_key_one=follow, X/left_key_one=freeze.")
    p.add_argument("--pico-follow-button", type=str, default="right_key_one",
                   help="Controller button name that switches to follow mode.")
    p.add_argument("--pico-freeze-button", type=str, default="left_key_one",
                   help="Controller button name that switches to freeze mode.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Online z inferer (ONNX-only)
# -----------------------------------------------------------------------------

class OnlineZInferer:
    def __init__(
        self,
        backward_onnx: str,
        mujoco_xml: str,
        device: str,
        hz: float,
        root_height_obs: bool = False,
        drop_first_n: int = 1,
        use_wall_clock_dt: bool = False,
        freeze_z_on_static_pose: bool = True,
        static_pose_tol: float = 1e-3,
        debug_z: bool = False,
        max_z_delta: float = 0.0,
        fix_quat_continuity: bool = False,
        angvel_delta_frame: str = "local",
    ) -> None:
        import mujoco
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if str(device).lower().startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(
            str(Path(backward_onnx).expanduser().resolve()),
            providers=providers,
        )
        self.session_input_names = {i.name for i in self.session.get_inputs()}
        output_names = [o.name for o in self.session.get_outputs()]
        self.session_output_name = "z" if "z" in output_names else output_names[0]

        self.dt = 1.0 / float(hz)
        self.drop_first_n = int(drop_first_n)
        self._step_count = 0
        self.use_wall_clock_dt = bool(use_wall_clock_dt)
        self.freeze_z_on_static_pose = bool(freeze_z_on_static_pose)
        self.static_pose_tol = float(static_pose_tol)
        self.debug_z = bool(debug_z)
        self._wall_step_t0: Optional[float] = None
        self._prev_z_dbg: Optional[np.ndarray] = None
        self._dbg_last_print: float = 0.0
        self._last_invalid_z_warning: float = 0.0
        self.max_z_delta = float(max_z_delta)
        self.fix_quat_continuity = bool(fix_quat_continuity)
        if str(angvel_delta_frame) not in ("local", "world"):
            raise ValueError(f"angvel_delta_frame must be local or world, got {angvel_delta_frame!r}")
        self.angvel_delta_frame = str(angvel_delta_frame)
        self.quat_flip_total = 0
        self.quat_flip_step_total = 0
        self._last_quat_min_dot = 1.0
        self._last_quat_flip_count = 0
        self.root_height_obs = bool(root_height_obs)

        xml_path = Path(mujoco_xml).expanduser().resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"--mujoco_xml not found: {xml_path}")
        self.xml_path = xml_path
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.default_pose_29 = G1_29DOF_DEFAULT_POSE_RAD.copy()

        self.base_body_names = list(G1_29DOF_BODY_NAMES)
        self.extend_cfg = [dict(c) for c in G1_29DOF_EXTEND_CONFIG]

        body_ids: List[int] = []
        for name in self.base_body_names:
            try:
                body_ids.append(self.mj_model.body(name).id)
            except KeyError as e:
                raise KeyError(
                    f"Body '{name}' not found in MuJoCo model at {xml_path}. "
                    f"Either fix the XML or update G1_29DOF_BODY_NAMES."
                ) from e
        self.body_ids = np.asarray(body_ids, dtype=np.int32)

        self.prev_root_pos: Optional[np.ndarray] = None
        self.prev_root_quat_xyzw: Optional[np.ndarray] = None
        self.prev_dof_pos: Optional[np.ndarray] = None
        self.prev_all_body_pos: Optional[np.ndarray] = None
        self.prev_all_body_rot_xyzw: Optional[np.ndarray] = None

        # Fallback z (used until enough history is collected)
        self.last_z = _standing_z()

    def get_last_z(self) -> np.ndarray:
        if self.last_z.shape == (256,):
            return self.last_z.copy()
        return _standing_z()

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
        base_pos, base_quat_xyzw = _fk_bodies_from_qpos(self.mj_model, self.mj_data, self.body_ids, qpos)

        all_names = self.base_body_names.copy()
        all_body_pos = base_pos
        all_body_rot_xyzw = base_quat_xyzw

        for ext in self.extend_cfg:
            parent_idx = all_names.index(ext["parent_name"])
            pos_in_parent = np.asarray(ext["pos"], dtype=np.float64).reshape(3)
            rot_wxyz = np.asarray(ext["rot"], dtype=np.float64).reshape(4)
            rot_xyzw = _wxyz_to_xyzw(rot_wxyz)

            parent_pos = all_body_pos[parent_idx]
            parent_rot = all_body_rot_xyzw[parent_idx]

            ext_pos = parent_pos + _quat_rotate_xyzw(parent_rot[None, :], pos_in_parent[None, :])[0]
            ext_rot = _quat_mul_xyzw(parent_rot[None, :], rot_xyzw[None, :])[0]

            all_body_pos = np.concatenate([all_body_pos, ext_pos[None, :]], axis=0)
            all_body_rot_xyzw = np.concatenate([all_body_rot_xyzw, ext_rot[None, :]], axis=0)
            all_names.append(ext["joint_name"])

        return all_body_pos.astype(np.float64), all_body_rot_xyzw.astype(np.float64)

    def _warn_invalid_z(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_invalid_z_warning >= 1.0:
            print(f"[realtime_z_server] {message}", flush=True)
            self._last_invalid_z_warning = now

    def _validate_z_output(self, z: Any) -> Optional[np.ndarray]:
        try:
            z_np = np.asarray(z, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            self._warn_invalid_z("invalid ONNX z output; not publishing z")
            return None
        if z_np.shape != (256,):
            self._warn_invalid_z(f"invalid ONNX z shape {z_np.shape}; not publishing z")
            return None
        if not np.all(np.isfinite(z_np)):
            self._warn_invalid_z("non-finite ONNX z output; not publishing z")
            return None
        return z_np

    def step(self, pose: PoseFrame, mode: str = "follow") -> Optional[np.ndarray]:
        now_wall = time.perf_counter()

        root_pos = pose.root_pos.astype(np.float32)
        root_quat_wxyz = pose.root_quat_wxyz.astype(np.float32)
        root_quat_xyzw = _wxyz_to_xyzw(root_quat_wxyz)
        dof_pos = pose.dof_pos.astype(np.float32)

        all_body_pos, all_body_rot_xyzw = self._fk_all_bodies(root_pos, root_quat_wxyz, dof_pos)

        if self.prev_all_body_pos is None:
            self.prev_root_pos = root_pos
            self.prev_root_quat_xyzw = root_quat_xyzw
            self.prev_dof_pos = dof_pos
            self.prev_all_body_pos = all_body_pos
            self.prev_all_body_rot_xyzw = all_body_rot_xyzw
            self._step_count += 1
            self._wall_step_t0 = now_wall
            return self.last_z

        # Same teleop pose repeated at 50Hz -> velocities would be all zero, which
        # is not what we want to feed into backward_map. Hold the last z instead.
        if (
            self.freeze_z_on_static_pose
            and self._step_count >= self.drop_first_n
            and np.allclose(dof_pos, self.prev_dof_pos, rtol=0.0, atol=self.static_pose_tol)
            and np.allclose(root_pos, self.prev_root_pos, rtol=0.0, atol=self.static_pose_tol)
            and np.allclose(root_quat_xyzw, self.prev_root_quat_xyzw, rtol=0.0, atol=self.static_pose_tol)
        ):
            self._step_count += 1
            return self.last_z

        if self.use_wall_clock_dt and self._wall_step_t0 is not None:
            dt_vel = float(now_wall - self._wall_step_t0)
            dt_vel = max(1.0 / 240.0, min(dt_vel, 0.5))
        else:
            dt_vel = float(self.dt)

        body_vel = (all_body_pos - self.prev_all_body_pos) / dt_vel
        dof_vel = (dof_pos - self.prev_dof_pos) / dt_vel
        quat_dots = _quat_pair_dot_xyzw(self.prev_all_body_rot_xyzw, all_body_rot_xyzw)
        quat_flip_count = int(np.count_nonzero(quat_dots < 0.0))
        self._last_quat_min_dot = float(np.min(quat_dots)) if quat_dots.size else 1.0
        self._last_quat_flip_count = quat_flip_count
        if quat_flip_count > 0:
            self.quat_flip_total += quat_flip_count
            self.quat_flip_step_total += 1

        body_ang_vel = _quat_to_ang_vel_xyzw(
            self.prev_all_body_rot_xyzw,
            all_body_rot_xyzw,
            dt_vel,
            fix_quat_continuity=self.fix_quat_continuity,
            delta_frame=self.angvel_delta_frame,
        )

        self.prev_root_pos = root_pos
        self.prev_root_quat_xyzw = root_quat_xyzw
        self.prev_dof_pos = dof_pos
        self.prev_all_body_pos = all_body_pos
        self.prev_all_body_rot_xyzw = all_body_rot_xyzw
        self._step_count += 1
        self._wall_step_t0 = now_wall

        if self._step_count <= self.drop_first_n:
            return self.last_z

        # Build single-step obs (T = 1).
        ref_body_pos = all_body_pos[None, :, :].astype(np.float32)
        ref_body_rots = all_body_rot_xyzw[None, :, :].astype(np.float32)
        ref_body_vels = body_vel[None, :, :].astype(np.float32)
        ref_body_angular_vels = body_ang_vel[None, :, :].astype(np.float32)

        ref_dof_pos = (dof_pos - self.default_pose_29)[None, :].astype(np.float32)
        ref_dof_vel = dof_vel[None, :].astype(np.float32)

        obs_dict_priv = _compute_humanoid_observations_max_np(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=bool(self.root_height_obs),
        )
        privileged_state = np.concatenate(
            [v.astype(np.float32) for v in obs_dict_priv.values()], axis=-1
        ).astype(np.float32)

        base_quat = ref_body_rots[:, 0, :]  # (1, 4)
        gravity_vec = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        projected_gravity = _quat_rotate_inverse_xyzw(base_quat, gravity_vec).astype(np.float32)
        ref_ang_vel = ref_body_angular_vels[:, 0, :].astype(np.float32)

        state = np.concatenate(
            [ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel], axis=-1
        ).astype(np.float32)
        last_action = ref_dof_pos  # mirrors humanoidverse.get_backward_observation()

        feed: Dict[str, np.ndarray] = {}
        if "state" in self.session_input_names:
            feed["state"] = state
        if "last_action" in self.session_input_names:
            feed["last_action"] = last_action
        if "privileged_state" in self.session_input_names:
            feed["privileged_state"] = privileged_state

        z_np = self._validate_z_output(self.session.run([self.session_output_name], feed)[0])
        if z_np is None:
            return None

        if self.max_z_delta > 0.0 and z_np.shape[0] == self.last_z.shape[0]:
            dz_raw = z_np - self.last_z
            dz_clip = np.clip(dz_raw, -self.max_z_delta, self.max_z_delta)
            z_np = (self.last_z + dz_clip).astype(np.float32)

        self.last_z = z_np

        if self.debug_z and self._prev_z_dbg is not None:
            if now_wall - self._dbg_last_print >= 1.0:
                dz = float(np.max(np.abs(z_np - self._prev_z_dbg)))
                body_ang_norm = float(np.max(np.linalg.norm(body_ang_vel, axis=-1)))
                print(
                    f"[realtime_z_server] mode={mode} max|dz|={dz:.4f} "
                    f"body_ang_vel_max={body_ang_norm:.3f} "
                    f"quat_min_dot={self._last_quat_min_dot:.6f} "
                    f"quat_flips={self._last_quat_flip_count} "
                    f"fix_quat_continuity={self.fix_quat_continuity} "
                    f"angvel_delta_frame={self.angvel_delta_frame}"
                )
                self._dbg_last_print = now_wall
        self._prev_z_dbg = z_np.copy()

        return self.last_z


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dt = 1.0 / float(args.hz)

    mode_state = ModeState("follow")
    if args.enable_keyboard_control:
        _start_keyboard_control(mode_state)

    if not args.dry_run and not args.backward_onnx:
        raise SystemExit("--backward_onnx is required unless --dry_run")

    ctx = zmq.Context.instance()

    req_sock = ctx.socket(zmq.PUSH)
    req_sock.setsockopt(zmq.LINGER, 0)
    req_sock.setsockopt(zmq.SNDHWM, 10)
    req_sock.connect(args.teleop_req)

    rep_sock = ctx.socket(zmq.PULL)
    rep_sock.setsockopt(zmq.LINGER, 0)
    rep_sock.setsockopt(zmq.RCVHWM, 10)
    rep_sock.connect(args.teleop_rep)

    ctrl_sock = None
    if args.enable_pico_control:
        ctrl_sock = ctx.socket(zmq.PULL)
        ctrl_sock.setsockopt(zmq.LINGER, 0)
        ctrl_sock.setsockopt(zmq.RCVHWM, 100)
        ctrl_sock.connect(args.teleop_ctrl)

    z_sock = None
    if not args.dry_run:
        z_sock = ctx.socket(zmq.PUB)
        z_sock.setsockopt(zmq.SNDHWM, 1)
        z_sock.setsockopt(zmq.LINGER, 0)
        z_sock.bind(args.z_bind)

    inferer: Optional[OnlineZInferer] = None
    if args.backward_onnx:
        if not args.mujoco_xml:
            raise SystemExit("--mujoco_xml is required when --backward_onnx is set")
        inferer = OnlineZInferer(
            backward_onnx=args.backward_onnx,
            mujoco_xml=args.mujoco_xml,
            device=args.device,
            hz=float(args.hz),
            root_height_obs=bool(args.root_height_obs),
            drop_first_n=int(args.drop_first_n),
            use_wall_clock_dt=bool(args.wall_clock_dt),
            freeze_z_on_static_pose=not bool(args.no_freeze_static_pose),
            debug_z=bool(args.debug_z),
            max_z_delta=float(args.max_z_delta),
            fix_quat_continuity=bool(args.fix_quat_continuity),
            angvel_delta_frame=str(args.angvel_delta_frame),
        )
        print("[realtime_z_server] z_mode: UFO realtime inference (ONNX)")
        print("[realtime_z_server] backward_onnx:", args.backward_onnx)
        print("[realtime_z_server] backward_onnx_inputs:", sorted(inferer.session_input_names))
        print("[realtime_z_server] backward_onnx_output:", inferer.session_output_name)
        try:
            print("[realtime_z_server] active_providers:", inferer.session.get_providers())
        except Exception:
            pass
        print("[realtime_z_server] mujoco_xml:", args.mujoco_xml)
        print("[realtime_z_server] device:", args.device)
        print("[realtime_z_server] root_height_obs:", bool(inferer.root_height_obs))
        print("[realtime_z_server] drop_first_n:", int(args.drop_first_n))
        print("[realtime_z_server] wall_clock_dt:", bool(args.wall_clock_dt))
        print("[realtime_z_server] fix_quat_continuity:", bool(args.fix_quat_continuity))
        print("[realtime_z_server] angvel_delta_frame:", str(args.angvel_delta_frame))
        print("[realtime_z_server] freeze_static_pose:", not bool(args.no_freeze_static_pose))
        print("[realtime_z_server] max_retarget_age_ms:", float(args.max_retarget_age_ms))
        print("[realtime_z_server] max_z_delta:", float(args.max_z_delta))
        print("[realtime_z_server] pose_buffer:", bool(args.enable_pose_buffer))
        if args.enable_pose_buffer:
            print("[realtime_z_server] pose_buffer_lookback_ms:", float(args.pose_buffer_lookback_ms))
            print("[realtime_z_server] pose_buffer_window_ms:", float(args.pose_buffer_window_ms))
    else:
        print("[realtime_z_server] z_mode: standing (placeholder)")

    print("[realtime_z_server] teleop_req:", args.teleop_req)
    print("[realtime_z_server] teleop_rep:", args.teleop_rep)
    print("[realtime_z_server] teleop_ctrl:", args.teleop_ctrl)
    print("[realtime_z_server] z_bind:", args.z_bind if not args.dry_run else "(dry_run: not bound)")
    print("[realtime_z_server] hz:", float(args.hz))
    print("[realtime_z_server] mode=follow")
    print("[realtime_z_server] keyboard_control:", bool(args.enable_keyboard_control))
    print("[realtime_z_server] pico_control:", bool(args.enable_pico_control))
    if args.enable_pico_control:
        print(
            "[realtime_z_server] pico_buttons: "
            f"{args.pico_follow_button}=follow, {args.pico_freeze_button}=freeze"
        )
    time.sleep(0.2)  # PUB slow-joiner + teleop connect

    latest_pose: Optional[PoseFrame] = None
    last_valid_pose_monotonic: Optional[float] = None
    last_pose_stale_warning = 0.0
    max_pose_stale_s = (
        float(args.max_retarget_age_ms) / 1000.0
        if float(args.max_retarget_age_ms) >= 0.0
        else None
    )
    current_pose: Optional[PoseFrame] = None
    pose_buffer: Optional[PoseFrameBuffer] = None
    pose_buffer_info: Dict[str, Any] = {"mode": "disabled", "buffer_len": 0}
    if args.enable_pose_buffer:
        pose_buffer = PoseFrameBuffer(
            lookback_ms=float(args.pose_buffer_lookback_ms),
            window_ms=float(args.pose_buffer_window_ms),
        )
    last_z = _standing_z()
    have_last_z = False
    prev_pico_follow_pressed = False
    prev_pico_freeze_pressed = False
    n = 0
    next_t = time.perf_counter()
    while True:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
        next_t += dt

        req = {"start": False, "t_req_ms": int(time.time() * 1000 - args.lookback_ms)}
        try:
            req_sock.send_string(json.dumps(req), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

        latest_raw: Optional[str] = None
        while True:
            try:
                latest_raw = rep_sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        if latest_raw is not None:
            try:
                pose_recv_ns = time.monotonic_ns()
                payload = json.loads(latest_raw)
                if isinstance(payload, dict):
                    retarget_age_ms = payload.get("retarget_age_ms")
                    if (
                        retarget_age_ms is not None
                        and float(args.max_retarget_age_ms) >= 0.0
                        and float(retarget_age_ms) > float(args.max_retarget_age_ms)
                    ):
                        frame = None
                    else:
                        frame = _extract_latest_frame(payload)
                    if frame is not None:
                        latest_pose = frame
                        last_valid_pose_monotonic = time.monotonic()
                        if pose_buffer is not None:
                            pose_buffer.append(frame, pose_recv_ns)
            except Exception:
                pass

        if ctrl_sock is not None:
            latest_ctrl_raw: Optional[str] = None
            while True:
                try:
                    latest_ctrl_raw = ctrl_sock.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
            if latest_ctrl_raw is not None:
                try:
                    ctrl_payload = json.loads(latest_ctrl_raw)
                    buttons = ctrl_payload.get("controller_buttons") if isinstance(ctrl_payload, dict) else None
                    if isinstance(buttons, dict):
                        follow_pressed = bool(buttons.get(args.pico_follow_button, False))
                        freeze_pressed = bool(buttons.get(args.pico_freeze_button, False))
                        if follow_pressed and not prev_pico_follow_pressed:
                            _set_mode_from_input(mode_state, "follow", args.pico_follow_button)
                        if freeze_pressed and not prev_pico_freeze_pressed:
                            _set_mode_from_input(mode_state, "freeze", args.pico_freeze_button)
                        prev_pico_follow_pressed = follow_pressed
                        prev_pico_freeze_pressed = freeze_pressed
                except Exception:
                    pass

        mode = mode_state.get()
        current_pose = latest_pose
        if mode == "follow" and pose_buffer is not None:
            sampled_pose, pose_buffer_info = pose_buffer.sample(time.monotonic_ns())
            if sampled_pose is not None:
                current_pose = sampled_pose
        if mode == "follow" and _is_pose_stale(last_valid_pose_monotonic, max_pose_stale_s):
            now_warn = time.monotonic()
            if now_warn - last_pose_stale_warning >= 1.0:
                print(
                    "[realtime_z_server] teleop pose stale; not publishing z",
                    flush=True,
                )
                last_pose_stale_warning = now_warn
            current_pose = None
            continue
        if current_pose is None and mode == "follow":
            continue

        if mode == "freeze":
            z = last_z.copy() if have_last_z else _standing_z()
        elif inferer is not None:
            assert current_pose is not None
            z = inferer.step(current_pose, mode=mode)
        else:
            z = _standing_z()
        if z is None:
            continue

        if not args.dry_run:
            try:
                assert z_sock is not None
                z_sock.send(np.asarray(z, dtype=np.float32).tobytes(), flags=zmq.DONTWAIT)
            except zmq.Again:
                pass

        z_arr = np.asarray(z, dtype=np.float32).reshape(-1)
        if z_arr.shape[0] == 256:
            last_z = z_arr.copy()
            have_last_z = True

        n += 1
        if n % int(max(1, args.hz)) == 0:
            zn = float(np.linalg.norm(z_arr))
            if current_pose is not None:
                rp = current_pose.root_pos
                rq = current_pose.root_quat_wxyz
                pb = pose_buffer_info if pose_buffer is not None else {"mode": "disabled", "buffer_len": 0}
                pb_alpha = pb.get("alpha")
                pb_alpha_str = "None" if pb_alpha is None else f"{float(pb_alpha):.3f}"
                print(
                    f"[realtime_z_server] mode={mode} | pose ok | "
                    f"root_pos=({rp[0]:+.3f},{rp[1]:+.3f},{rp[2]:+.3f}) "
                    f"quat_wxyz=({rq[0]:+.3f},{rq[1]:+.3f},{rq[2]:+.3f},{rq[3]:+.3f}) "
                    f"||z||={zn:.3f} "
                    f"pose_buffer_mode={pb.get('mode')} "
                    f"pose_buffer_len={pb.get('buffer_len')} "
                    f"pose_buffer_span_ms={pb.get('span_ms')} "
                    f"pose_buffer_alpha={pb_alpha_str}"
                )
            else:
                print(f"[realtime_z_server] mode={mode} | no pose yet | ||z||={zn:.3f}")


if __name__ == "__main__":
    main()
