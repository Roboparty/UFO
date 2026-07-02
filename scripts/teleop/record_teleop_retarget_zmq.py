#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_toe_link",
    "pelvis_contour_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_toe_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "head_link",
    "head_mocap",
    "imu_in_torso",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record retargeted teleop frames over ZMQ.")
    parser.add_argument("--req_addr", type=str, default="tcp://127.0.0.1:28701")
    parser.add_argument("--rep_addr", type=str, default="tcp://127.0.0.1:28702")
    parser.add_argument("--ctrl_addr", type=str, default="tcp://127.0.0.1:28703")
    parser.add_argument("--period_ms", type=float, default=20.0, help="Request period in milliseconds")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory for teleop_retarget_*.npz (default: current working directory)",
    )
    return parser.parse_args()


def next_output_path(output_dir: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    existing = sorted(output_dir.glob(f"teleop_retarget_{ts}_*.npz"))
    next_idx = len(existing)
    return output_dir / f"teleop_retarget_{ts}_{next_idx:08d}.npz"


def save_recording(output_path: Path, fps: float, root_pos_list, root_quat_wxyz_list, dof_pos_list) -> None:
    root_pos = np.asarray(root_pos_list, dtype=np.float64).reshape(-1, 3)
    root_quat_wxyz = np.asarray(root_quat_wxyz_list, dtype=np.float64).reshape(-1, 4)
    root_rot_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    dof_pos = np.asarray(dof_pos_list, dtype=np.float64).reshape(-1, len(JOINT_NAMES))

    np.savez(
        output_path,
        fps=np.float32(fps),
        root_pos=root_pos,
        root_rot=root_rot_xyzw,
        dof_pos=dof_pos,
        joint_names=np.asarray(JOINT_NAMES),
        body_names=np.asarray(BODY_NAMES),
    )


def main() -> None:
    args = parse_args()

    try:
        import zmq
    except ImportError as exc:
        raise ImportError("pyzmq is required.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    period_s = float(args.period_ms) / 1000.0
    fps = 1.0 / period_s

    ctx = zmq.Context.instance()
    req_sock = ctx.socket(zmq.PUSH)
    req_sock.setsockopt(zmq.LINGER, 0)
    req_sock.setsockopt(zmq.SNDHWM, 100)
    req_sock.connect(args.req_addr)

    rep_sock = ctx.socket(zmq.PULL)
    rep_sock.setsockopt(zmq.LINGER, 0)
    rep_sock.setsockopt(zmq.RCVHWM, 200)
    rep_sock.connect(args.rep_addr)

    ctrl_sock = ctx.socket(zmq.PULL)
    ctrl_sock.setsockopt(zmq.LINGER, 0)
    ctrl_sock.setsockopt(zmq.RCVHWM, 200)
    ctrl_sock.connect(args.ctrl_addr)

    root_pos_list: list[list[float]] = []
    root_quat_list: list[list[float]] = []
    dof_pos_list: list[list[float]] = []

    recording = False
    pending_start_flag = False
    prev_a = False
    prev_x = False
    req_count = 0
    saved_count = 0
    latest_buttons: Optional[dict] = None

    print(
        f"Recorder connected req->{args.req_addr}, rep<-{args.rep_addr}, ctrl<-{args.ctrl_addr}, "
        f"period_ms={args.period_ms:.1f}, fps={fps:.1f}"
    )
    print("Press A to start recording, X to stop and save.")

    try:
        next_send = time.monotonic()
        while True:
            while True:
                try:
                    raw = ctrl_sock.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                payload = json.loads(raw)
                buttons = payload.get("controller_buttons", None)
                if isinstance(buttons, dict):
                    latest_buttons = buttons

            if latest_buttons is not None:
                a_pressed = bool(latest_buttons.get("right_key_one", False))
                x_pressed = bool(latest_buttons.get("left_key_one", False))

                if a_pressed and not prev_a and not recording:
                    recording = True
                    pending_start_flag = True
                    root_pos_list.clear()
                    root_quat_list.clear()
                    dof_pos_list.clear()
                    print("[Recorder] start recording")

                if x_pressed and not prev_x and recording:
                    if root_pos_list:
                        output_path = next_output_path(output_dir)
                        save_recording(
                            output_path=output_path,
                            fps=fps,
                            root_pos_list=root_pos_list,
                            root_quat_wxyz_list=root_quat_list,
                            dof_pos_list=dof_pos_list,
                        )
                        saved_count += 1
                        print(
                            f"[Recorder] stop recording | frames={len(root_pos_list)} | "
                            f"saved={output_path}"
                        )
                    else:
                        print("[Recorder] stop recording | no frames captured, skip save")
                    recording = False
                    pending_start_flag = False
                    root_pos_list.clear()
                    root_quat_list.clear()
                    dof_pos_list.clear()

                prev_a = a_pressed
                prev_x = x_pressed

            now = time.monotonic()
            if now >= next_send:
                req = {
                    "start": bool(pending_start_flag),
                    "t_req_ms": int(time.time() * 1000),
                }
                try:
                    req_sock.send_string(json.dumps(req), flags=zmq.NOBLOCK)
                    req_count += 1
                    pending_start_flag = False
                except zmq.Again:
                    pass
                next_send += period_s
                if next_send < now - period_s:
                    next_send = now + period_s

            latest_reply = None
            while True:
                try:
                    raw = rep_sock.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                latest_reply = raw

            if latest_reply is not None:
                payload = json.loads(latest_reply)
                frames = payload.get("frames", [])
                if recording and isinstance(frames, list) and len(frames) > 0:
                    frame = frames[-1]
                    root_pos = frame.get("root_pos", None)
                    root_quat = frame.get("root_quat", None)
                    dof_pos = frame.get("dof_pos", None)
                    if (
                        isinstance(root_pos, list)
                        and len(root_pos) == 3
                        and isinstance(root_quat, list)
                        and len(root_quat) == 4
                        and isinstance(dof_pos, list)
                        and len(dof_pos) == len(JOINT_NAMES)
                    ):
                        root_pos_list.append([float(x) for x in root_pos])
                        root_quat_list.append([float(x) for x in root_quat])
                        dof_pos_list.append([float(x) for x in dof_pos])

            time.sleep(0.001)

    except KeyboardInterrupt:
        print(
            f"\nRecorder exiting | requests={req_count}, recorded_frames={len(root_pos_list)}, saves={saved_count}"
        )
    finally:
        req_sock.close(0)
        rep_sock.close(0)
        ctrl_sock.close(0)


if __name__ == "__main__":
    main()
