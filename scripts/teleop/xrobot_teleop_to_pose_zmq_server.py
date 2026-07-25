#!/usr/bin/env python3
"""
Low-latency PICO/XRobot teleop bridge for sim2real.

Shipped in UFO-Deploy as scripts/teleop/ (see SOURCE). Run from this directory or
set PYTHONPATH; requires the vendored motion_tracking_retarget dependencies and
xrobotoolkit_sdk in the active environment.

Architecture:
1. XR callback thread stores the latest VR snapshot with a monotonic timestamp.
2. A retarget thread waits for new VR data and only retargets the latest snapshot.
3. A request thread serves the newest ZMQ request using time-based interpolation over a
   short retarget history buffer.
4. A control thread publishes controller buttons at a fixed rate.
"""

import argparse
import tempfile
import json
import multiprocessing as mp
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

from default_mimic_obs import DEFAULT_MIMIC_OBS
from motion_tracking_retarget.helper import default_controller_buttons, parse_xrobot_motion_snapshot
from motion_tracking_retarget.params import XR_BODY_JOINT_NAMES

xrt = None


def _load_runtime_dependencies() -> None:
    global xrt

    try:
        import mink  # noqa: F401
        import mujoco  # noqa: F401
        import scipy  # noqa: F401
        import yaml  # noqa: F401
        import zmq  # noqa: F401
        from motion_tracking_retarget.xrobot_retarget import XRobotRetargetWorkerRuntime as _Runtime  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Failed to import vendored motion_tracking_retarget runtime dependencies. "
            "Install mink, mujoco, scipy, pyyaml, numpy, and pyzmq in the active teleop environment."
        ) from exc

    try:
        import xrobotoolkit_sdk as _xrt
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'xrobotoolkit_sdk'. Install the patched SDK in the active Python environment."
        ) from exc

    if not hasattr(_xrt, "init"):
        raise ImportError("Installed xrobotoolkit_sdk does not expose init().")

    callback_names = (
        "register_frame_callback",
        "clear_frame_callback",
        "has_frame_callback",
    )
    polling_names = (
        "is_body_data_available",
        "get_body_joints_pose",
        "get_body_timestamp_ns",
        "get_A_button",
        "get_B_button",
        "get_X_button",
        "get_Y_button",
    )
    if not all(hasattr(_xrt, name) for name in callback_names) and not all(
        hasattr(_xrt, name) for name in polling_names
    ):
        raise ImportError(
            "Installed xrobotoolkit_sdk exposes neither callback APIs nor the required polling APIs."
        )

    xrt = _xrt


class _ShutdownTolerantExecutor:
    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def submit(self, *args: Any, **kwargs: Any) -> Future:
        try:
            return self._executor.submit(*args, **kwargs)
        except RuntimeError as exc:
            if "shutdown" not in str(exc):
                raise
            future: Future = Future()
            future.set_result(None)
            return future

    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)


def _format_floor_rgb(rgb: tuple[float, float, float]) -> str:
    return " ".join(f"{float(channel):.3f}" for channel in rgb)


def _scale_floor_rgb(rgb: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(min(max(float(channel) * scale, 0.0), 1.0) for channel in rgb)


def _has_named_xml_element(xml_text: str, tag: str, name: str) -> bool:
    start = 0
    open_token = f"<{tag}"
    while True:
        tag_start = xml_text.find(open_token, start)
        if tag_start < 0:
            return False
        tag_end = xml_text.find(">", tag_start)
        if tag_end < 0:
            return False
        tag_text = xml_text[tag_start:tag_end]
        if f'name="{name}"' in tag_text or f"name='{name}'" in tag_text:
            return True
        start = tag_end + 1


def inject_floor_scene_xml(
    xml_text: str,
    ground_rgb: tuple[float, float, float] = (0.35, 0.35, 0.35),
) -> str:
    if "<asset>" not in xml_text or "</asset>" not in xml_text:
        raise ValueError("MJCF XML structure error: expected <asset>...</asset> block for viewer floor assets.")
    if "</worldbody>" not in xml_text:
        raise ValueError("MJCF XML structure error: expected </worldbody> block for viewer floor geom.")

    if "<visual>" not in xml_text:
        visual_xml = """\
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
"""
        asset_open = xml_text.find("<asset>")
        xml_text = xml_text[:asset_open] + visual_xml + xml_text[asset_open:]

    ground_rgb_dark = _scale_floor_rgb(ground_rgb, 0.75)
    asset_parts: list[str] = []
    if not _has_named_xml_element(xml_text, "texture", "groundplane"):
        asset_parts.append(
            f"""\
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="{_format_floor_rgb(ground_rgb)}" rgb2="{_format_floor_rgb(ground_rgb_dark)}" markrgb="0.8 0.8 0.8" width="300" height="300"/>
"""
        )
    if not _has_named_xml_element(xml_text, "material", "groundplane"):
        asset_parts.append(
            """\
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
"""
        )
    if asset_parts:
        asset_close = xml_text.find("</asset>")
        xml_text = xml_text[:asset_close] + "".join(asset_parts) + xml_text[asset_close:]

    if _has_named_xml_element(xml_text, "geom", "floor"):
        return xml_text

    worldbody_xml = """\
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
"""
    worldbody_close = xml_text.find("</worldbody>")
    return xml_text[:worldbody_close] + worldbody_xml + xml_text[worldbody_close:]


@contextmanager
def temp_mjcf_with_floor(mjcf_path: str | Path) -> Iterator[Path]:
    source_path = Path(mjcf_path).expanduser()
    if not source_path.is_absolute():
        source_path = source_path.resolve(strict=False)

    viewer_xml_text = inject_floor_scene_xml(source_path.read_text(encoding="utf-8"))
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=".ufo_viewer_floor_",
            dir=source_path.parent,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(viewer_xml_text)
            temp_path = Path(tmp.name)
        yield temp_path
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


class WebRetargetViewer:
    def __init__(self, mujoco_xml: str, port: int) -> None:
        try:
            import mujoco
            import viser
            from mjviser import ViserMujocoScene
        except ImportError as exc:
            raise ImportError(
                "--web-visualize requires mujoco, viser, and mjviser in the active Python environment."
            ) from exc

        xml_path = Path(mujoco_xml).expanduser()
        if not xml_path.is_file():
            raise FileNotFoundError(f"web viewer MuJoCo XML not found: {xml_path}")

        self.mujoco = mujoco
        with temp_mjcf_with_floor(xml_path) as viewer_xml:
            self.model = mujoco.MjModel.from_xml_path(str(viewer_xml))
        self.data = mujoco.MjData(self.model)
        self.server = viser.ViserServer(port=int(port), label="ufo-retarget")
        executor = _ShutdownTolerantExecutor(self.server._thread_executor)
        self.server._thread_executor = executor
        self.server.scene._thread_executor = executor
        self.server.gui._thread_executor = executor
        self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)
        self.scene.create_visualization_gui(camera_distance=3.0, camera_azimuth=120.0, camera_elevation=20.0)
        self.step(np.zeros(int(self.model.nq), dtype=np.float32))
        print(f"[web-retarget-viewer] http://localhost:{self.server.get_port()}")

    def step(self, qpos: np.ndarray) -> None:
        qpos_arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        n = min(int(self.model.nq), qpos_arr.shape[0])
        self.data.qpos[:n] = qpos_arr[:n]
        self.mujoco.mj_forward(self.model, self.data)
        self.scene.update_from_arrays(
            body_xpos=self.data.xpos[None, ...],
            body_xmat=self.data.xmat.reshape(1, -1, 3, 3),
            mocap_pos=np.zeros((1, 0, 3), dtype=np.float64),
            mocap_quat=np.zeros((1, 0, 4), dtype=np.float64),
            qpos=self.data.qpos[None, ...],
            qvel=self.data.qvel[None, ...],
            ctrl=self.data.ctrl[None, ...] if self.model.nu > 0 else None,
        )

    def close(self) -> None:
        self.server.stop()


@dataclass
class RetargetedFrame:
    recv_ns: int
    qpos: np.ndarray


def _retarget_worker_main(
    raw_recv_conn: Any,
    result_send_conn: Any,
    worker_config: Dict[str, Any],
) -> None:
    try:
        from motion_tracking_retarget.xrobot_retarget import XRobotRetargetWorkerRuntime

        runtime = XRobotRetargetWorkerRuntime(worker_config)
    except Exception as exc:
        try:
            result_send_conn.send({"type": "worker_init_error", "error": str(exc)})
        except Exception:
            pass
        return

    try:
        result_send_conn.send({"type": "worker_ready"})
    except Exception:
        return

    last_processed_seq = 0
    while True:
        try:
            if not raw_recv_conn.poll(0.1):
                continue
            packet = raw_recv_conn.recv()
        except EOFError:
            break
        except Exception as exc:
            try:
                result_send_conn.send({"type": "worker_runtime_error", "error": str(exc)})
            except Exception:
                pass
            continue

        if isinstance(packet, dict) and packet.get("type") == "shutdown":
            break

        dropped_before_process = 0
        while raw_recv_conn.poll():
            try:
                newer_packet = raw_recv_conn.recv()
            except EOFError:
                newer_packet = None
            if newer_packet is None:
                break
            if isinstance(newer_packet, dict) and newer_packet.get("type") == "shutdown":
                return
            dropped_before_process += 1
            packet = newer_packet

        prev_processed_seq = last_processed_seq
        try:
            result = runtime.process_packet(packet)
        except Exception as exc:
            try:
                result_send_conn.send({"type": "worker_runtime_error", "error": str(exc)})
            except Exception:
                pass
            continue

        if result is None:
            continue

        result["dropped_before_process"] = int(dropped_before_process)
        result["prev_processed_seq"] = int(prev_processed_seq)
        last_processed_seq = int(result["seq"])

        try:
            result_send_conn.send(result)
        except (BrokenPipeError, EOFError, OSError):
            break


class LowLatencyTeleopPoseZMQServer:
    BODY_JOINT_NAMES = XR_BODY_JOINT_NAMES

    def __init__(self, args: argparse.Namespace):
        from motion_tracking_retarget.joint_mapping import (
            build_joint_permutation,
            canonical_joint_names,
            policy_joint_names,
            qpos_size,
        )
        from motion_tracking_retarget.robot_config import load_teleop_robot_config

        self.args = args
        self.robot = args.robot
        self.target_robot = "g1"
        self.teleop_config = load_teleop_robot_config(self.target_robot)
        self.vis_fps = int(args.vis_fps if args.vis_fps is not None else self.teleop_config.vis_fps)
        self.ctrl_fps = int(args.ctrl_fps if args.ctrl_fps is not None else self.teleop_config.ctrl_fps)
        lookback_ms = (
            float(args.lookback_ms)
            if args.lookback_ms is not None
            else float(self.teleop_config.lookback_ms)
        )
        self.lookback_ns = int(lookback_ms * 1e6)
        retarget_buffer_window_s = (
            float(args.retarget_buffer_window_s)
            if args.retarget_buffer_window_s is not None
            else float(self.teleop_config.retarget_buffer_window_s)
        )
        self.retarget_buffer_window_ns = int(retarget_buffer_window_s * 1e9)
        self.log_interval_s = float(
            args.log_interval_s if args.log_interval_s is not None else self.teleop_config.log_interval_s
        )
        self.actual_human_height = float(
            args.actual_human_height
            if args.actual_human_height is not None
            else self.teleop_config.actual_human_height
        )
        self.req_bind_addr = str(args.req_bind_addr or self.teleop_config.req_bind_addr)
        self.rep_bind_addr = str(args.rep_bind_addr or self.teleop_config.rep_bind_addr)
        self.ctrl_bind_addr = str(args.ctrl_bind_addr or self.teleop_config.ctrl_bind_addr)
        self.ctrl_pub_bind_addr = str(args.ctrl_pub_bind_addr or "")
        self.retarget_max_iter = int(self.teleop_config.max_iter)
        self.calibration_button = (
            str(args.calibration_button).strip()
            if args.calibration_button is not None and str(args.calibration_button).strip()
            else self.teleop_config.calibration_button
        )
        self.canonical_joint_names = canonical_joint_names(self.target_robot)
        self.ufo_output_joint_names = policy_joint_names()
        self.joint_permutation = build_joint_permutation(
            self.canonical_joint_names,
            self.ufo_output_joint_names,
        )
        self.canonical_qpos_size = qpos_size(self.target_robot)
        self.canonical_dof_count = len(self.canonical_joint_names)
        if self.canonical_dof_count != 29:
            raise ValueError(f"canonical G1 dof_count must be 29, got {self.canonical_dof_count}")
        if self.canonical_qpos_size != 36:
            raise ValueError(f"canonical G1 qpos_size must be 36, got {self.canonical_qpos_size}")

        if self.vis_fps <= 0:
            raise ValueError("vis_fps must be > 0")
        if self.ctrl_fps <= 0:
            raise ValueError("ctrl_fps must be > 0")
        if self.lookback_ns < 0:
            raise ValueError("lookback_ms must be >= 0")
        if self.retarget_buffer_window_ns <= 0:
            raise ValueError("retarget_buffer_window_s must be > 0")
        if self.log_interval_s < 0:
            raise ValueError("log_interval_s must be >= 0")

        self.retarget = None
        self.zmq_context = None
        self.req_sock = None
        self.rep_sock = None
        self.ctrl_sock = None
        self.ctrl_pub_sock = None
        self.web_viewer = None
        self.xrt_input_mode = "uninitialized"

        self.default_qpos = self._build_default_qpos()
        self.last_controller_buttons: Dict[str, Any] = default_controller_buttons()

        self.latest_vr_lock = threading.Lock()
        self.latest_vr_poses: Optional[Any] = None
        self.latest_vr_recv_ns: int = 0
        self.latest_vr_seq: int = 0
        self.latest_vr_motion_timestamp_ns: Optional[int] = None
        self.latest_vr_calibration_request_id = 0
        self.prev_calibration_button_pressed = False

        self.retarget_buffer_lock = threading.Lock()
        self.retarget_buffer: deque[RetargetedFrame] = deque()
        self.vis_lock = threading.Lock()
        self.latest_vis_qpos: Optional[np.ndarray] = None

        self.vr_frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.stats_lock = threading.Lock()

        self.frame_seq = 0
        self.last_vis_monotonic = 0.0
        self.last_req_monotonic: Optional[float] = None
        self.req_count = 0
        self.reply_count = 0
        self.reply_drop_count = 0
        self.req_merged_total = 0
        self.fallback_count = 0
        self.raw_motion_drop_count = 0
        self.latest_req_dt_ms: Optional[float] = None
        self.latest_merged_reqs = 0

        self.callback_count = 0
        self.retarget_count = 0
        self.latest_debug_info: Dict[str, Any] = {
            "mode": "no_data",
            "target_age_ms": None,
            "older_age_ms": None,
            "newer_age_ms": None,
            "span_ms": None,
            "buffer_len": 0,
            "retarget_age_ms": None,
            "raw_motion_age_ms": None,
        }

        self.retarget_thread = None
        self.raw_sender_thread = None
        self.worker_result_thread = None
        self.request_thread = None
        self.control_thread = None
        self.stats_thread = None
        self.visualization_thread = None
        self.xrt_poll_thread = None

        self.mp_ctx = mp.get_context("spawn")
        self.raw_send_conn = None
        self.raw_recv_conn = None
        self.result_send_conn = None
        self.result_recv_conn = None
        self.retarget_process = None

    def _build_default_qpos(self) -> np.ndarray:
        mimic = np.asarray(DEFAULT_MIMIC_OBS[self.robot], dtype=np.float32).reshape(-1)
        if mimic.shape[0] < 35:
            raise ValueError(f"DEFAULT_MIMIC_OBS[{self.robot}] must be at least 35 dims")
        dof_pos = mimic[6:35]
        root_z = float(mimic[2])
        root_pos = np.array([0.0, 0.0, root_z], dtype=np.float32)
        root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.concatenate([root_pos, root_quat, dof_pos], axis=0).astype(np.float32)

    @staticmethod
    def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return (q / norm).astype(np.float32)

    def _slerp_quat_wxyz(self, quat0: np.ndarray, quat1: np.ndarray, alpha: float) -> np.ndarray:
        q0 = self._normalize_quat_wxyz(quat0).astype(np.float64)
        q1 = self._normalize_quat_wxyz(quat1).astype(np.float64)
        t = float(np.clip(alpha, 0.0, 1.0))

        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        if dot > 0.9995:
            out = q0 + t * (q1 - q0)
            return self._normalize_quat_wxyz(out)

        theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        sin_theta_0 = float(np.sin(theta_0))
        if abs(sin_theta_0) < 1e-8:
            return self._normalize_quat_wxyz(q0)

        theta = theta_0 * t
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        out = s0 * q0 + s1 * q1
        return self._normalize_quat_wxyz(out)

    def _interpolate_qpos(self, prev_qpos: np.ndarray, next_qpos: np.ndarray, alpha: float) -> np.ndarray:
        t = float(np.clip(alpha, 0.0, 1.0))
        frame = prev_qpos * (1.0 - t) + next_qpos * t
        frame[3:7] = self._slerp_quat_wxyz(prev_qpos[3:7], next_qpos[3:7], t)
        return frame.astype(np.float32)

    @staticmethod
    def _xrt_has_callback_api() -> bool:
        return all(
            hasattr(xrt, name)
            for name in ("register_frame_callback", "clear_frame_callback", "has_frame_callback")
        )

    @staticmethod
    def _xrt_has_polling_api() -> bool:
        return all(
            hasattr(xrt, name)
            for name in (
                "is_body_data_available",
                "get_body_joints_pose",
                "get_body_timestamp_ns",
                "get_A_button",
                "get_B_button",
                "get_X_button",
                "get_Y_button",
            )
        )

    @staticmethod
    def _safe_xrt_call(name: str, default: Any = None) -> Any:
        fn = getattr(xrt, name, None)
        if fn is None:
            return default
        try:
            return fn()
        except Exception:
            return default

    @staticmethod
    def _axis(values: Any) -> list[float]:
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            try:
                return [float(values[0]), float(values[1])]
            except Exception:
                return [0.0, 0.0]
        return [0.0, 0.0]

    def _build_polling_snapshot(self) -> Dict[str, Any]:
        body_available = bool(self._safe_xrt_call("is_body_data_available", False))
        body_timestamp_ns = int(self._safe_xrt_call("get_body_timestamp_ns", 0) or 0)
        top_timestamp_ns = int(self._safe_xrt_call("get_time_stamp_ns", body_timestamp_ns) or body_timestamp_ns)
        poses = None
        if body_available:
            poses = self._safe_xrt_call("get_body_joints_pose", None)

        return {
            "timestamp_ns": top_timestamp_ns,
            "controllers": {
                "left": {
                    "primary_button": bool(self._safe_xrt_call("get_X_button", False)),
                    "secondary_button": bool(self._safe_xrt_call("get_Y_button", False)),
                    "axis_click": bool(self._safe_xrt_call("get_left_axis_click", False)),
                    "trigger": float(self._safe_xrt_call("get_left_trigger", 0.0) or 0.0),
                    "grip": float(self._safe_xrt_call("get_left_grip", 0.0) or 0.0),
                    "axis": self._axis(self._safe_xrt_call("get_left_axis", [0.0, 0.0])),
                },
                "right": {
                    "primary_button": bool(self._safe_xrt_call("get_A_button", False)),
                    "secondary_button": bool(self._safe_xrt_call("get_B_button", False)),
                    "axis_click": bool(self._safe_xrt_call("get_right_axis_click", False)),
                    "trigger": float(self._safe_xrt_call("get_right_trigger", 0.0) or 0.0),
                    "grip": float(self._safe_xrt_call("get_right_grip", 0.0) or 0.0),
                    "axis": self._axis(self._safe_xrt_call("get_right_axis", [0.0, 0.0])),
                },
            },
            "body": {
                "available": body_available,
                "timestamp_ns": body_timestamp_ns,
                "poses": poses,
            },
        }

    def _xrt_poll_loop(self) -> None:
        period_s = 1.0 / max(1e-6, float(self.args.xr_poll_hz))
        while not self.stop_event.is_set():
            snapshot = self._build_polling_snapshot()
            self._on_vr_frame(snapshot)
            self.stop_event.wait(timeout=period_s)

    def _serialize_qpos_frame(self, qpos: np.ndarray) -> Dict[str, Any]:
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.canonical_qpos_size:
            raise ValueError(f"canonical qpos shape must be ({self.canonical_qpos_size},), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("canonical qpos contains non-finite values")
        root_quat = q[3:7]
        quat_norm = float(np.linalg.norm(root_quat))
        if not np.isfinite(quat_norm) or quat_norm < 1e-6:
            raise ValueError(f"canonical root quaternion norm invalid: {quat_norm}")
        dof_pos = q[7 : 7 + self.canonical_dof_count][self.joint_permutation]
        return {
            "root_pos": q[0:3].tolist(),
            "root_quat": root_quat.tolist(),
            "dof_pos": dof_pos.tolist(),
        }

    def _on_vr_frame(self, snapshot: dict) -> None:
        recv_ns = time.monotonic_ns()
        parsed = parse_xrobot_motion_snapshot(
            snapshot,
            joint_count=len(XR_BODY_JOINT_NAMES),
        )

        should_wake_retarget = False
        with self.latest_vr_lock:
            self.callback_count += 1
            if parsed is not None:
                self.last_controller_buttons = parsed.controller_buttons
                calibration_requested = False
                if self.calibration_button is not None:
                    pressed = bool(parsed.controller_buttons.get(self.calibration_button, False))
                    calibration_requested = pressed and not self.prev_calibration_button_pressed
                    self.prev_calibration_button_pressed = pressed
                timestamp_changed = self.latest_vr_motion_timestamp_ns != parsed.motion_timestamp_ns
                if timestamp_changed or calibration_requested:
                    self.latest_vr_poses = parsed.poses
                    self.latest_vr_recv_ns = recv_ns
                    self.latest_vr_seq += 1
                    self.latest_vr_motion_timestamp_ns = parsed.motion_timestamp_ns
                    if calibration_requested:
                        self.latest_vr_calibration_request_id += 1
                    should_wake_retarget = True
        if should_wake_retarget:
            self.vr_frame_event.set()

    def _append_retarget_frame(self, recv_ns: int, qpos: np.ndarray) -> None:
        cutoff_ns = recv_ns - self.retarget_buffer_window_ns
        with self.retarget_buffer_lock:
            self.retarget_buffer.append(RetargetedFrame(recv_ns=recv_ns, qpos=qpos.astype(np.float32, copy=True)))
            while self.retarget_buffer and self.retarget_buffer[0].recv_ns < cutoff_ns:
                self.retarget_buffer.popleft()

    def _get_retarget_frames_snapshot(self) -> list[RetargetedFrame]:
        with self.retarget_buffer_lock:
            return list(self.retarget_buffer)

    def _sample_target_qpos(self, frames: list[RetargetedFrame], target_ns: int) -> tuple[np.ndarray, bool, Dict[str, Any]]:
        if not frames:
            return self.default_qpos.copy(), True, {
                "mode": "default",
                "target_ns": target_ns,
                "older_ns": None,
                "newer_ns": None,
                "alpha": None,
                "buffer_len": 0,
        }
        if len(frames) == 1:
            only_ns = frames[0].recv_ns
            return frames[0].qpos.astype(np.float32, copy=True), True, {
                "mode": "single_frame",
                "target_ns": target_ns,
                "older_ns": only_ns,
                "newer_ns": only_ns,
                "alpha": None,
                "buffer_len": 1,
            }
        if target_ns <= frames[0].recv_ns:
            oldest_ns = frames[0].recv_ns
            return frames[0].qpos.astype(np.float32, copy=True), True, {
                "mode": "fallback_oldest",
                "target_ns": target_ns,
                "older_ns": oldest_ns,
                "newer_ns": oldest_ns,
                "alpha": None,
                "buffer_len": len(frames),
            }
        if target_ns >= frames[-1].recv_ns:
            latest_ns = frames[-1].recv_ns
            return frames[-1].qpos.astype(np.float32, copy=True), True, {
                "mode": "fallback_latest",
                "target_ns": target_ns,
                "older_ns": latest_ns,
                "newer_ns": latest_ns,
                "alpha": None,
                "buffer_len": len(frames),
            }

        for idx in range(1, len(frames)):
            prev_frame = frames[idx - 1]
            next_frame = frames[idx]
            if target_ns <= next_frame.recv_ns:
                dt = next_frame.recv_ns - prev_frame.recv_ns
                if dt <= 0:
                    same_ns = next_frame.recv_ns
                    return next_frame.qpos.astype(np.float32, copy=True), True, {
                        "mode": "degenerate_dt",
                        "target_ns": target_ns,
                        "older_ns": same_ns,
                        "newer_ns": same_ns,
                        "alpha": None,
                        "buffer_len": len(frames),
                    }
                alpha = float(target_ns - prev_frame.recv_ns) / float(dt)
                return self._interpolate_qpos(prev_frame.qpos, next_frame.qpos, alpha), False, {
                    "mode": "interpolate",
                    "target_ns": target_ns,
                    "older_ns": prev_frame.recv_ns,
                    "newer_ns": next_frame.recv_ns,
                    "alpha": alpha,
                    "buffer_len": len(frames),
                }

        latest_ns = frames[-1].recv_ns
        return frames[-1].qpos.astype(np.float32, copy=True), True, {
            "mode": "fallback_latest",
            "target_ns": target_ns,
            "older_ns": latest_ns,
            "newer_ns": latest_ns,
            "alpha": None,
            "buffer_len": len(frames),
        }

    def _build_reply_frames(self, req_recv_ns: int) -> tuple[list[np.ndarray], bool, Dict[str, Any]]:
        frames = self._get_retarget_frames_snapshot()
        target_base_ns = req_recv_ns - self.lookback_ns
        qpos, used_fallback, sample_info = self._sample_target_qpos(frames, target_base_ns)
        return [qpos], used_fallback, sample_info

    def _get_latest_frame_ages_ms(self, now_ns: Optional[int] = None) -> tuple[Optional[float], Optional[float]]:
        if now_ns is None:
            now_ns = time.monotonic_ns()

        with self.latest_vr_lock:
            latest_raw_recv_ns = int(self.latest_vr_recv_ns) if self.latest_vr_recv_ns > 0 else None

        with self.retarget_buffer_lock:
            latest_retarget_recv_ns = self.retarget_buffer[-1].recv_ns if self.retarget_buffer else None

        raw_motion_age_ms = None
        if latest_raw_recv_ns is not None:
            raw_motion_age_ms = round((now_ns - latest_raw_recv_ns) / 1e6, 3)

        retarget_age_ms = None
        if latest_retarget_recv_ns is not None:
            retarget_age_ms = round((now_ns - latest_retarget_recv_ns) / 1e6, 3)

        return retarget_age_ms, raw_motion_age_ms

    def _update_debug_info(self, sample_info: Dict[str, Any], req_recv_ns: int) -> None:
        older_ns = sample_info.get("older_ns")
        newer_ns = sample_info.get("newer_ns")
        retarget_age_ms, raw_motion_age_ms = self._get_latest_frame_ages_ms()

        info = {
            "mode": sample_info.get("mode"),
            "target_age_ms": round((req_recv_ns - int(sample_info["target_ns"])) / 1e6, 3),
            "older_age_ms": None if older_ns is None else round((req_recv_ns - int(older_ns)) / 1e6, 3),
            "newer_age_ms": None if newer_ns is None else round((req_recv_ns - int(newer_ns)) / 1e6, 3),
            "span_ms": None
            if older_ns is None or newer_ns is None
            else round((int(newer_ns) - int(older_ns)) / 1e6, 3),
            "alpha": sample_info.get("alpha"),
            "buffer_len": int(sample_info.get("buffer_len", 0)),
            "retarget_age_ms": retarget_age_ms,
            "raw_motion_age_ms": raw_motion_age_ms,
        }
        with self.stats_lock:
            self.latest_debug_info = info

    def _warn_on_fallback(self, sample_info: Dict[str, Any]) -> None:
        now_ns = time.monotonic_ns()
        retarget_age_ms, raw_motion_age_ms = self._get_latest_frame_ages_ms(now_ns=now_ns)
        target_age_ms = round((now_ns - int(sample_info["target_ns"])) / 1e6, 3)
        older_ns = sample_info.get("older_ns")
        newer_ns = sample_info.get("newer_ns")
        older_age_ms = None if older_ns is None else round((now_ns - int(older_ns)) / 1e6, 3)
        newer_age_ms = None if newer_ns is None else round((now_ns - int(newer_ns)) / 1e6, 3)
        print(
            "[Warning] interpolation fallback "
            f"mode={sample_info.get('mode')}, "
            f"target_age_ms={target_age_ms}, "
            f"older_age_ms={older_age_ms}, "
            f"newer_age_ms={newer_age_ms}, "
            f"buffer={int(sample_info.get('buffer_len', 0))}, "
            f"latest_retarget_age_ms={retarget_age_ms}, "
            f"latest_raw_motion_age_ms={raw_motion_age_ms}"
        )

    def _warn_on_raw_motion_drop(self, dropped_count: int, latest_seq: int, last_processed_seq: int) -> None:
        now_ns = time.monotonic_ns()
        retarget_age_ms, raw_motion_age_ms = self._get_latest_frame_ages_ms(now_ns=now_ns)
        print(
            "[Warning] retarget lag dropped raw motion frames "
            f"dropped={int(dropped_count)}, "
            f"last_processed_seq={int(last_processed_seq)}, "
            f"latest_seq={int(latest_seq)}, "
            f"latest_retarget_age_ms={retarget_age_ms}, "
            f"latest_raw_motion_age_ms={raw_motion_age_ms}"
        )

    def _raw_sender_loop(self) -> None:
        last_sent_seq = 0
        last_sent_calibration_request_id = 0

        while not self.stop_event.is_set():
            if not self.vr_frame_event.wait(timeout=0.1):
                continue

            while not self.stop_event.is_set():
                with self.latest_vr_lock:
                    poses = self.latest_vr_poses
                    recv_ns = self.latest_vr_recv_ns
                    seq = self.latest_vr_seq
                    calibration_request_id = self.latest_vr_calibration_request_id

                if poses is None or seq == last_sent_seq:
                    with self.latest_vr_lock:
                        if self.latest_vr_seq == last_sent_seq:
                            self.vr_frame_event.clear()
                            break
                    continue

                if last_sent_seq != 0 and seq > last_sent_seq + 1:
                    dropped_count = seq - last_sent_seq - 1
                    self.raw_motion_drop_count += int(dropped_count)
                    self._warn_on_raw_motion_drop(
                        dropped_count=dropped_count,
                        latest_seq=seq,
                        last_processed_seq=last_sent_seq,
                    )

                try:
                    self.raw_send_conn.send(
                        {
                            "seq": int(seq),
                            "recv_ns": int(recv_ns),
                            "poses": poses,
                            "calibration_requested": bool(
                                calibration_request_id != last_sent_calibration_request_id
                            ),
                        }
                    )
                except (BrokenPipeError, EOFError, OSError) as exc:
                    print(f"[Warning] raw->worker pipe failed: {exc}")
                    self.stop_event.set()
                    self.vr_frame_event.set()
                    break

                last_sent_seq = seq
                last_sent_calibration_request_id = calibration_request_id

    def _worker_result_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if not self.result_recv_conn.poll(0.1):
                    continue
                payload = self.result_recv_conn.recv()
            except EOFError:
                print("[Warning] worker->main pipe closed")
                self.stop_event.set()
                break
            except Exception as exc:
                print(f"[Warning] worker result recv failed: {exc}")
                self.stop_event.set()
                break

            if not isinstance(payload, dict):
                continue

            payload_type = payload.get("type")
            if payload_type == "worker_ready":
                continue
            if payload_type in ("worker_init_error", "worker_runtime_error"):
                print(f"[Warning] retarget worker error: {payload.get('error')}")
                if payload_type == "worker_init_error":
                    self.stop_event.set()
                continue
            if payload_type != "retarget_result":
                continue

            dropped_before_process = int(payload.get("dropped_before_process", 0))
            if dropped_before_process > 0:
                self.raw_motion_drop_count += dropped_before_process
                self._warn_on_raw_motion_drop(
                    dropped_count=dropped_before_process,
                    latest_seq=int(payload.get("seq", 0)),
                    last_processed_seq=int(payload.get("prev_processed_seq", 0)),
                )

            qpos_curr = np.asarray(payload.get("qpos"), dtype=np.float32).reshape(-1)
            recv_ns = int(payload["recv_ns"])
            self._append_retarget_frame(recv_ns=recv_ns, qpos=qpos_curr)
            self.retarget_count += 1

            if self.web_viewer is not None:
                with self.vis_lock:
                    self.latest_vis_qpos = qpos_curr.astype(np.float32, copy=True)

    def _drain_requests_blocking(self) -> tuple[Optional[Dict[str, Any]], Optional[int], int]:
        import zmq

        poller = zmq.Poller()
        poller.register(self.req_sock, zmq.POLLIN)

        while not self.stop_event.is_set():
            events = dict(poller.poll(timeout=100))
            if self.req_sock not in events:
                continue

            latest_req: Optional[Dict[str, Any]] = None
            req_recv_ns: Optional[int] = None
            merged_reqs = 0
            any_start = False

            while True:
                try:
                    raw = self.req_sock.recv_string(flags=zmq.NOBLOCK)
                    req_recv_ns = time.monotonic_ns()
                except zmq.Again:
                    break
                except Exception as exc:
                    print(f"[Warning] request recv failed: {exc}")
                    break

                try:
                    req = json.loads(raw)
                except Exception:
                    print("[Warning] bad request JSON")
                    continue
                if not isinstance(req, dict):
                    continue

                merged_reqs += 1
                any_start = any_start or bool(req.get("start", False))
                latest_req = req

            if latest_req is None:
                continue

            latest_req["start"] = any_start
            return latest_req, req_recv_ns, merged_reqs

        return None, None, 0

    def _request_loop(self) -> None:
        import zmq

        while not self.stop_event.is_set():
            req, req_recv_ns, merged_reqs = self._drain_requests_blocking()
            if req is None or req_recv_ns is None:
                continue

            now = time.monotonic()
            self.req_count += 1
            self.req_merged_total += int(merged_reqs)
            self.latest_merged_reqs = int(merged_reqs)
            if self.last_req_monotonic is None:
                self.latest_req_dt_ms = None
            else:
                self.latest_req_dt_ms = (now - self.last_req_monotonic) * 1000.0
            self.last_req_monotonic = now

            out_frames, used_fallback, sample_info = self._build_reply_frames(req_recv_ns=req_recv_ns)
            self._update_debug_info(sample_info=sample_info, req_recv_ns=req_recv_ns)
            if used_fallback:
                self.fallback_count += 1
                self._warn_on_fallback(sample_info=sample_info)
            seq_start = int(self.frame_seq)
            self.frame_seq += len(out_frames)

            retarget_frames = self._get_retarget_frames_snapshot()
            retarget_age_ms = None
            if retarget_frames:
                retarget_age_ms = int((time.monotonic_ns() - retarget_frames[-1].recv_ns) / 1e6)

            payload = {
                "start": bool(req.get("start", False)),
                "no_interp_applied": bool(used_fallback),
                "chunk_size": len(out_frames),
                "frame_seq_start": seq_start,
                "retarget_age_ms": retarget_age_ms,
                "t_rep_ms": int(time.time() * 1000),
                "frames": [self._serialize_qpos_frame(x) for x in out_frames],
            }

            try:
                self.rep_sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
                self.reply_count += 1
            except zmq.Again:
                self.reply_drop_count += 1
                print("[Warning] reply queue full, drop one reply")
            except Exception as exc:
                print(f"[Warning] reply send failed: {exc}")

    def _stats_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.stop_event.wait(timeout=self.log_interval_s):
                break

            with self.stats_lock:
                info = dict(self.latest_debug_info)
            with self.latest_vr_lock:
                callback_count = int(self.callback_count)
            retarget_count = int(self.retarget_count)
            req_count = int(self.req_count)
            reply_count = int(self.reply_count)
            reply_drop_count = int(self.reply_drop_count)
            req_merged_total = int(self.req_merged_total)
            fallback_count = int(self.fallback_count)
            raw_motion_drop_count = int(self.raw_motion_drop_count)
            latest_merged_reqs = int(self.latest_merged_reqs)
            latest_req_dt_ms = self.latest_req_dt_ms

            alpha = info.get("alpha")
            alpha_str = "None" if alpha is None else f"{float(alpha):.3f}"
            req_dt_str = "None" if latest_req_dt_ms is None else f"{float(latest_req_dt_ms):.2f}"
            print(
                "[Stats] "
                f"req={req_count}, rep={reply_count}, rep_drop={reply_drop_count}, "
                f"req_merged_total={req_merged_total}, latest_merged={latest_merged_reqs}, "
                f"fallback={fallback_count}, raw_drop={raw_motion_drop_count}, "
                f"cb={callback_count}, retarget={retarget_count}, "
                f"mode={info.get('mode')}, buffer={info.get('buffer_len')}, "
                f"latest_req_dt_ms={req_dt_str}, "
                f"target_age_ms={info.get('target_age_ms')}, "
                f"older_age_ms={info.get('older_age_ms')}, "
                f"newer_age_ms={info.get('newer_age_ms')}, "
                f"span_ms={info.get('span_ms')}, alpha={alpha_str}, "
                f"retarget_age_ms={info.get('retarget_age_ms')}, "
                f"raw_motion_age_ms={info.get('raw_motion_age_ms')}"
            )

    def _control_loop(self) -> None:
        import zmq

        period_s = 1.0 / float(self.ctrl_fps)
        while not self.stop_event.is_set():
            with self.latest_vr_lock:
                buttons = dict(self.last_controller_buttons)

            payload = {
                "t_ms": int(time.time() * 1000),
                "controller_buttons": buttons,
            }
            payload_raw = json.dumps(payload)
            try:
                self.ctrl_sock.send_string(payload_raw, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as exc:
                print(f"[Warning] control send failed: {exc}")

            if self.ctrl_pub_sock is not None:
                try:
                    self.ctrl_pub_sock.send_string(payload_raw, flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
                except Exception as exc:
                    print(f"[Warning] control pub send failed: {exc}")

            self.stop_event.wait(timeout=period_s)

    def _visualization_loop(self) -> None:
        if self.web_viewer is None:
            return

        period_s = 1.0 / float(self.vis_fps)
        while not self.stop_event.is_set():
            with self.vis_lock:
                qpos = None if self.latest_vis_qpos is None else self.latest_vis_qpos.copy()

            if qpos is not None:
                if self.web_viewer is not None:
                    try:
                        self.web_viewer.step(qpos)
                    except Exception as exc:
                        print(f"[Warning] web visualization failed, disabling viewer: {exc}")
                        self.web_viewer.close()
                        self.web_viewer = None

            self.stop_event.wait(timeout=period_s)

    def setup(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq is required for the teleop ZMQ server.") from exc

        if self.args.visualize:
            print("[Warning] --visualize legacy native viewer is disabled; use --web-visualize for UFO teleop debug.")

        if self.args.web_visualize:
            web_xml = str(self.args.web_mujoco_xml or _default_web_mujoco_xml())
            try:
                self.web_viewer = WebRetargetViewer(web_xml, int(self.args.web_port))
            except Exception as exc:
                print(f"[Warning] web viewer disabled: {exc}")
                self.web_viewer = None

        self.raw_recv_conn, self.raw_send_conn = self.mp_ctx.Pipe(duplex=False)
        self.result_recv_conn, self.result_send_conn = self.mp_ctx.Pipe(duplex=False)
        worker_config = {
            "qpos_size": int(self.canonical_qpos_size),
            "target_robot": self.target_robot,
            "actual_human_height": self.actual_human_height,
            "max_iter": int(self.retarget_max_iter),
            "send_human_motion": bool(self.args.visualize),
            "enable_height_alignment": bool(self.teleop_config.height_alignment_enabled),
            "height_alignment_xrobot_body_min_each_frame": bool(
                self.teleop_config.height_alignment_xrobot_body_min_each_frame
            ),
            "height_alignment_target_z": float(self.teleop_config.height_alignment_target_z),
            "height_bootstrap_frames": int(self.teleop_config.height_alignment_bootstrap_frames),
            "height_alignment_foot_body_names": tuple(self.teleop_config.height_alignment_foot_body_names),
        }
        self.retarget_process = self.mp_ctx.Process(
            target=_retarget_worker_main,
            args=(self.raw_recv_conn, self.result_send_conn, worker_config),
            name="teleop-retarget-worker",
            daemon=True,
        )
        self.retarget_process.start()
        self.raw_recv_conn.close()
        self.raw_recv_conn = None
        self.result_send_conn.close()
        self.result_send_conn = None

        if not self.result_recv_conn.poll(10.0):
            raise RuntimeError("Retarget worker did not become ready within 10 seconds.")
        worker_msg = self.result_recv_conn.recv()
        if not isinstance(worker_msg, dict) or worker_msg.get("type") != "worker_ready":
            raise RuntimeError(f"Retarget worker failed to start: {worker_msg}")

        xrt.init()
        if self._xrt_has_callback_api():
            xrt.register_frame_callback(self._on_vr_frame)
            self.xrt_input_mode = "callback"
        elif self._xrt_has_polling_api():
            self.xrt_input_mode = "polling"
        else:
            raise RuntimeError("xrobotoolkit_sdk has no supported input API")

        self.zmq_context = zmq.Context.instance()

        self.req_sock = self.zmq_context.socket(zmq.PULL)
        self.req_sock.setsockopt(zmq.LINGER, 0)
        self.req_sock.setsockopt(zmq.RCVHWM, 500)
        self.req_sock.bind(self.req_bind_addr)

        self.rep_sock = self.zmq_context.socket(zmq.PUSH)
        self.rep_sock.setsockopt(zmq.LINGER, 0)
        self.rep_sock.setsockopt(zmq.SNDHWM, 500)
        self.rep_sock.bind(self.rep_bind_addr)

        self.ctrl_sock = self.zmq_context.socket(zmq.PUSH)
        self.ctrl_sock.setsockopt(zmq.LINGER, 0)
        self.ctrl_sock.setsockopt(zmq.SNDHWM, 500)
        self.ctrl_sock.bind(self.ctrl_bind_addr)

        if self.ctrl_pub_bind_addr:
            self.ctrl_pub_sock = self.zmq_context.socket(zmq.PUB)
            self.ctrl_pub_sock.setsockopt(zmq.LINGER, 0)
            self.ctrl_pub_sock.setsockopt(zmq.SNDHWM, 10)
            self.ctrl_pub_sock.bind(self.ctrl_pub_bind_addr)

        print("Low-latency teleop ZMQ pose server initialized")
        print(f"  req_bind_addr: {self.req_bind_addr}")
        print(f"  rep_bind_addr: {self.rep_bind_addr}")
        print(f"  ctrl_bind_addr: {self.ctrl_bind_addr}")
        if self.ctrl_pub_bind_addr:
            print(f"  ctrl_pub_bind_addr: {self.ctrl_pub_bind_addr}")
        print(f"  ctrl_fps: {self.ctrl_fps}")
        print(f"  canonical_target_robot: {self.target_robot}")
        print(f"  actual_human_height: {self.actual_human_height:.3f}")
        print(f"  retarget_max_iter: {self.retarget_max_iter}")
        print(f"  canonical_qpos_size: {self.canonical_qpos_size}")
        print(f"  canonical_joint_order: {list(self.canonical_joint_names)}")
        print(f"  ufo_output_joint_order: {list(self.ufo_output_joint_names)}")
        if np.array_equal(self.joint_permutation, np.arange(len(self.joint_permutation))):
            print("  joint_permutation: identity")
        else:
            print(f"  joint_permutation: {self.joint_permutation.tolist()}")
        print(f"  calibration_button: {self.calibration_button}")
        print(
            "  height_alignment: "
            f"enabled={self.teleop_config.height_alignment_enabled}, "
            f"feet={list(self.teleop_config.height_alignment_foot_body_names)}, "
            f"target_z={self.teleop_config.height_alignment_target_z}, "
            f"bootstrap_frames={self.teleop_config.height_alignment_bootstrap_frames}"
        )
        print(f"  pico_policy_control: {'enabled legacy/debug compatibility mode' if self.ctrl_pub_bind_addr else 'disabled'}")
        print("  chunk_size: fixed to 1 frame per reply")
        print(f"  lookback_ms: {self.lookback_ns / 1e6:.3f}")
        print(f"  retarget_buffer_window_s: {self.retarget_buffer_window_ns / 1e9:.3f}")
        print(f"  log_interval_s: {self.log_interval_s:.3f}")
        print(f"  visualize: {self.args.visualize}")
        print(f"  web_visualize: {self.args.web_visualize}")
        print(f"  xrt_input_mode: {self.xrt_input_mode}")
        print(f"  retarget_worker_pid: {self.retarget_process.pid if self.retarget_process else None}")

    def run(self) -> None:
        self.setup()

        self.raw_sender_thread = threading.Thread(
            target=self._raw_sender_loop,
            name="teleop-raw-sender",
            daemon=True,
        )
        self.worker_result_thread = threading.Thread(
            target=self._worker_result_loop,
            name="teleop-worker-result",
            daemon=True,
        )
        self.request_thread = threading.Thread(
            target=self._request_loop,
            name="teleop-request",
            daemon=True,
        )
        self.control_thread = threading.Thread(
            target=self._control_loop,
            name="teleop-control",
            daemon=True,
        )
        if self.web_viewer is not None:
            self.visualization_thread = threading.Thread(
                target=self._visualization_loop,
                name="teleop-visualization",
                daemon=True,
            )
        if self.log_interval_s > 0.0:
            self.stats_thread = threading.Thread(
                target=self._stats_loop,
                name="teleop-stats",
                daemon=True,
            )
        if self.xrt_input_mode == "polling":
            self.xrt_poll_thread = threading.Thread(
                target=self._xrt_poll_loop,
                name="teleop-xrt-poll",
                daemon=True,
            )

        self.raw_sender_thread.start()
        self.worker_result_thread.start()
        self.request_thread.start()
        self.control_thread.start()
        if self.xrt_poll_thread is not None:
            self.xrt_poll_thread.start()
        if self.visualization_thread is not None:
            self.visualization_thread.start()
        if self.stats_thread is not None:
            self.stats_thread.start()

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("KeyboardInterrupt, exiting low-latency teleop ZMQ pose server.")
        finally:
            self.stop_event.set()
            self.vr_frame_event.set()
            if self.xrt_input_mode == "callback":
                try:
                    xrt.clear_frame_callback()
                except Exception:
                    pass

            for thread in (
                self.raw_sender_thread,
                self.worker_result_thread,
                self.request_thread,
                self.control_thread,
                self.xrt_poll_thread,
                self.visualization_thread,
                self.stats_thread,
            ):
                if thread is not None:
                    thread.join(timeout=1.0)

            if self.raw_send_conn is not None:
                try:
                    self.raw_send_conn.send({"type": "shutdown"})
                except Exception:
                    pass
            if self.raw_send_conn is not None:
                self.raw_send_conn.close()
            if self.raw_recv_conn is not None:
                self.raw_recv_conn.close()
            if self.result_send_conn is not None:
                self.result_send_conn.close()
            if self.result_recv_conn is not None:
                self.result_recv_conn.close()
            if self.retarget_process is not None:
                self.retarget_process.join(timeout=2.0)
                if self.retarget_process.is_alive():
                    self.retarget_process.terminate()
                    self.retarget_process.join(timeout=1.0)

            if self.web_viewer is not None:
                self.web_viewer.close()
            if self.req_sock is not None:
                self.req_sock.close(0)
            if self.rep_sock is not None:
                self.rep_sock.close(0)
            if self.ctrl_sock is not None:
                self.ctrl_sock.close(0)
            if self.ctrl_pub_sock is not None:
                self.ctrl_pub_sock.close(0)


def _default_web_mujoco_xml() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "data/robots/g1/scene_29dof_freebase.xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-latency ZMQ teleop pose server")
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "g1"],
        default="unitree_g1",
        help="Robot key for defaults",
    )
    parser.add_argument("--actual_human_height", type=float, default=None)
    parser.add_argument("--vis_fps", type=int, default=None, help="Viewer update frequency")
    parser.add_argument("--ctrl_fps", type=int, default=None, help="Controller button publish frequency")
    parser.add_argument("--xr-poll-hz", type=float, default=50.0, help="XRobot SDK polling frequency when callback APIs are unavailable")
    parser.add_argument(
        "--lookback_ms",
        type=float,
        default=None,
        help="Sample reply frames at request_time - lookback_ms",
    )
    parser.add_argument(
        "--retarget_buffer_window_s",
        type=float,
        default=None,
        help="How much retarget history to keep for timestamp interpolation",
    )
    parser.add_argument(
        "--log_interval_s",
        type=float,
        default=None,
        help="Periodic debug log interval. Set to 0 to disable.",
    )
    parser.add_argument("--req_bind_addr", type=str, default=None)
    parser.add_argument("--rep_bind_addr", type=str, default=None)
    parser.add_argument("--ctrl_bind_addr", type=str, default=None)
    parser.add_argument("--ctrl_pub_bind_addr", type=str, default="")
    parser.add_argument("--web-visualize", action="store_true")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--web-mujoco-xml", type=str, default="")
    parser.add_argument("--calibration-button", type=str, default=None)
    parser.add_argument("--min_link_height", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--min_link_height_align_strategy",
        type=str,
        choices=["startup_fixed", "per_frame"],
        default="startup_fixed",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--min_link_height_bootstrap_frames", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_runtime_dependencies()
    server = LowLatencyTeleopPoseZMQServer(args)
    server.run()


if __name__ == "__main__":
    main()
