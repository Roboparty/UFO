from pathlib import Path
import sys
import threading

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "teleop"))

from motion_tracking_retarget.helper import default_controller_buttons, parse_xrobot_motion_snapshot  # noqa: E402
from motion_tracking_retarget.joint_mapping import (  # noqa: E402
    UFO_EXPECTED_G1_JOINT_NAMES,
    build_joint_permutation,
    canonical_joint_names,
    qpos_size,
)
from motion_tracking_retarget.params import XR_BODY_JOINT_NAMES, resolve_robot_xml_path  # noqa: E402
from motion_tracking_retarget.robot_config import load_teleop_robot_config  # noqa: E402
from motion_tracking_retarget.xrobot_retarget import XRobotRetargetWorkerRuntime  # noqa: E402
from scripts.teleop.xrobot_teleop_to_pose_zmq_server import LowLatencyTeleopPoseZMQServer  # noqa: E402


def _neutral_xr_poses():
    positions = {
        "Pelvis": (0.0, 1.0, 0.0),
        "Left_Hip": (-0.12, 0.92, 0.0),
        "Right_Hip": (0.12, 0.92, 0.0),
        "Spine1": (0.0, 1.12, 0.0),
        "Spine2": (0.0, 1.28, 0.0),
        "Spine3": (0.0, 1.45, 0.0),
        "Left_Knee": (-0.12, 0.52, 0.02),
        "Right_Knee": (0.12, 0.52, 0.02),
        "Left_Ankle": (-0.12, 0.12, 0.0),
        "Right_Ankle": (0.12, 0.12, 0.0),
        "Left_Foot": (-0.12, 0.02, 0.12),
        "Right_Foot": (0.12, 0.02, 0.12),
        "Neck": (0.0, 1.62, 0.0),
        "Left_Collar": (-0.12, 1.52, 0.0),
        "Right_Collar": (0.12, 1.52, 0.0),
        "Head": (0.0, 1.78, 0.0),
        "Left_Shoulder": (-0.28, 1.50, 0.0),
        "Right_Shoulder": (0.28, 1.50, 0.0),
        "Left_Elbow": (-0.52, 1.18, 0.0),
        "Right_Elbow": (0.52, 1.18, 0.0),
        "Left_Wrist": (-0.62, 0.88, 0.0),
        "Right_Wrist": (0.62, 0.88, 0.0),
        "Left_Hand": (-0.66, 0.78, 0.0),
        "Right_Hand": (0.66, 0.78, 0.0),
    }
    poses = np.zeros((len(XR_BODY_JOINT_NAMES), 7), dtype=np.float32)
    for idx, name in enumerate(XR_BODY_JOINT_NAMES):
        poses[idx, :3] = positions[name]
        poses[idx, 3:7] = [0.0, 0.0, 0.0, 1.0]
    return poses


def _snapshot(timestamp_ns=1000, *, right_a=False, left_x=False, poses=None):
    if poses is None:
        poses = _neutral_xr_poses()
    return {
        "timestamp_ns": timestamp_ns,
        "controllers": {
            "left": {
                "primary_button": left_x,
                "secondary_button": False,
                "axis_click": False,
                "trigger": 0.0,
                "grip": 0.0,
                "axis": [0.0, 0.0],
            },
            "right": {
                "primary_button": right_a,
                "secondary_button": False,
                "axis_click": False,
                "trigger": 0.0,
                "grip": 0.0,
                "axis": [0.0, 0.0],
            },
        },
        "body": {
            "available": True,
            "timestamp_ns": timestamp_ns,
            "poses": poses,
        },
    }


def test_teleop_config_loads():
    cfg = load_teleop_robot_config("g1")
    assert cfg.robot_key == "g1"
    assert cfg.dof_count == 29
    assert cfg.qpos_size == 36
    assert cfg.calibration_button is None
    assert cfg.height_alignment_foot_body_names == ("left_toe_link", "right_toe_link")
    assert cfg.max_iter == 5
    assert cfg.lookback_ms == 50.0
    assert not cfg.visualize


def test_retarget_xml_loads_without_external_motion_tracking():
    import mujoco as mj

    xml_path = resolve_robot_xml_path("g1")
    assert ROOT in xml_path.parents
    assert "motion_tracking_sim2real" not in str(xml_path)
    model = mj.MjModel.from_xml_path(str(xml_path))
    assert model.nq == 36
    assert len(canonical_joint_names("g1")) == 29
    body_names = {
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(model.nbody)
    }
    assert "left_toe_link" in body_names
    assert "right_toe_link" in body_names


def test_xr_parser_callback_and_polling_snapshots_match():
    callback_snapshot = _snapshot(timestamp_ns=1234, right_a=True, left_x=True)
    polling_snapshot = _snapshot(timestamp_ns=1234, right_a=True, left_x=True)
    parsed_callback = parse_xrobot_motion_snapshot(callback_snapshot, joint_count=len(XR_BODY_JOINT_NAMES))
    parsed_polling = parse_xrobot_motion_snapshot(polling_snapshot, joint_count=len(XR_BODY_JOINT_NAMES))
    assert parsed_callback is not None
    assert parsed_polling is not None
    np.testing.assert_array_equal(parsed_callback.poses, parsed_polling.poses)
    assert parsed_callback.poses.shape == (24, 7)
    assert parsed_callback.motion_timestamp_ns == parsed_polling.motion_timestamp_ns == 1234
    assert parsed_callback.controller_buttons["right_key_one"]
    assert parsed_callback.controller_buttons["left_key_one"]


def test_xr_parser_rejects_nonfinite_pose():
    poses = _neutral_xr_poses()
    poses[3, 0] = np.nan
    assert parse_xrobot_motion_snapshot(_snapshot(poses=poses), joint_count=24) is None


def test_timestamp_dedup_happens_before_worker_wakeup():
    server = LowLatencyTeleopPoseZMQServer.__new__(LowLatencyTeleopPoseZMQServer)
    server.latest_vr_lock = threading.Lock()
    server.last_controller_buttons = default_controller_buttons()
    server.callback_count = 0
    server.latest_vr_poses = None
    server.latest_vr_recv_ns = 0
    server.latest_vr_seq = 0
    server.latest_vr_motion_timestamp_ns = None
    server.latest_vr_calibration_requested = False
    server.prev_calibration_button_pressed = False
    server.calibration_button = None
    server.vr_frame_event = threading.Event()

    server._on_vr_frame(_snapshot(timestamp_ns=42))
    server._on_vr_frame(_snapshot(timestamp_ns=42))

    assert server.callback_count == 2
    assert server.latest_vr_seq == 1
    assert server.latest_vr_motion_timestamp_ns == 42


def test_synthetic_neutral_retarget_outputs_finite_qpos():
    runtime = XRobotRetargetWorkerRuntime(
        {
            "qpos_size": 36,
            "target_robot": "g1",
            "actual_human_height": 1.75,
            "max_iter": 5,
            "send_human_motion": False,
            "enable_height_alignment": True,
            "height_alignment_xrobot_body_min_each_frame": False,
            "height_alignment_target_z": 0.0,
            "height_bootstrap_frames": 10,
            "height_alignment_foot_body_names": ("left_toe_link", "right_toe_link"),
        }
    )
    out = runtime.process_packet({"seq": 1, "recv_ns": 1, "poses": _neutral_xr_poses()})
    assert out is not None
    qpos = out["qpos"]
    assert qpos.shape == (36,)
    assert np.all(np.isfinite(qpos))
    assert np.isfinite(qpos[2])
    assert 0.9 <= np.linalg.norm(qpos[3:7]) <= 1.1
    assert qpos[7:].shape == (29,)


def test_joint_permutation_validation_and_serialization():
    canonical = tuple(UFO_EXPECTED_G1_JOINT_NAMES)
    identity = build_joint_permutation(canonical, UFO_EXPECTED_G1_JOINT_NAMES)
    np.testing.assert_array_equal(identity, np.arange(29))

    shuffled = (canonical[1], canonical[0], *canonical[2:])
    perm = build_joint_permutation(shuffled, canonical)
    np.testing.assert_array_equal(perm[:2], np.array([1, 0]))

    for bad in (
        canonical[:-1],
        (canonical[0], *canonical),
        (*canonical, "extra_joint"),
    ):
        try:
            build_joint_permutation(bad, canonical)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid joint set accepted")

    server = LowLatencyTeleopPoseZMQServer.__new__(LowLatencyTeleopPoseZMQServer)
    server.canonical_qpos_size = 36
    server.canonical_dof_count = 29
    server.joint_permutation = np.array([1, 0, *range(2, 29)], dtype=np.int64)
    qpos = np.arange(36, dtype=np.float32)
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    frame = server._serialize_qpos_frame(qpos)
    assert frame["dof_pos"][0] == 8.0
    assert frame["dof_pos"][1] == 7.0


def test_pico_policy_control_default_disabled_and_realtime_lookback_zero():
    policy_launcher = (ROOT / "run_g1_teleop_policy_onboard.sh").read_text()
    teleop_launcher = (ROOT / "scripts/teleop/teleop_pose_50hz_onboard.sh").read_text()
    teleop_server = (ROOT / "scripts/teleop/xrobot_teleop_to_pose_zmq_server.py").read_text()
    realtime_server = (ROOT / "scripts/realtime/realtime_z_server.py").read_text()
    realtime_launcher = (ROOT / "scripts/realtime/run_realtime_z_server_onboard.sh").read_text()
    teleop_cfg = yaml.safe_load((ROOT / "config/teleop/g1.yaml").read_text())

    assert 'ENABLE_PICO_POLICY_CONTROL="${ENABLE_PICO_POLICY_CONTROL:-0}"' in policy_launcher
    assert 'cmd+=(--pico-control --pico-control-addr "${PICO_CONTROL_ADDR}")' in policy_launcher
    assert '"${ENABLE_PICO_POLICY_CONTROL}" == "1"' in policy_launcher
    assert 'CTRL_PUB_BIND_ADDR="${CTRL_PUB_BIND_ADDR:-}"' in teleop_launcher
    assert "pico_policy_control:" in teleop_server
    assert '--pico-follow-button", type=str, default="right_key_one"' in realtime_server
    assert '--pico-freeze-button", type=str, default="left_key_one"' in realtime_server
    assert '_set_mode_from_input(mode_state, "follow", args.pico_follow_button)' in realtime_server
    assert '_set_mode_from_input(mode_state, "freeze", args.pico_freeze_button)' in realtime_server
    assert '--pose-buffer-lookback-ms "${POSE_BUFFER_LOOKBACK_MS:-0}"' in realtime_launcher
    assert "--enable-pose-buffer" in realtime_launcher
    assert teleop_cfg["retarget"]["calibration"]["button"] is None


if __name__ == "__main__":
    test_teleop_config_loads()
    test_retarget_xml_loads_without_external_motion_tracking()
    test_xr_parser_callback_and_polling_snapshots_match()
    test_xr_parser_rejects_nonfinite_pose()
    test_timestamp_dedup_happens_before_worker_wakeup()
    test_synthetic_neutral_retarget_outputs_finite_qpos()
    test_joint_permutation_validation_and_serialization()
    test_pico_policy_control_default_disabled_and_realtime_lookback_zero()
    print("motion_tracking_retarget tests ok")
