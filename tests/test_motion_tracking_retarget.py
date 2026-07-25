from pathlib import Path
import argparse
import sys
import tempfile
import threading

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "teleop"))

from motion_tracking_retarget.helper import default_controller_buttons, parse_xrobot_motion_snapshot  # noqa: E402
from motion_tracking_retarget import robot_config as robot_config_module  # noqa: E402
from motion_tracking_retarget.joint_mapping import (  # noqa: E402
    DEFAULT_POLICY_CONFIG,
    UFO_EXPECTED_G1_JOINT_NAMES,
    build_joint_permutation,
    canonical_joint_names,
    policy_joint_names,
    qpos_size,
)
from motion_tracking_retarget.params import XR_BODY_JOINT_NAMES, resolve_robot_xml_path  # noqa: E402
from motion_tracking_retarget.robot_config import load_teleop_robot_config  # noqa: E402
from motion_tracking_retarget.xrobot_retarget import XRobotRetargetWorkerRuntime  # noqa: E402
from scripts.teleop.xrobot_teleop_to_pose_zmq_server import LowLatencyTeleopPoseZMQServer  # noqa: E402


def _server_args(**overrides):
    args = argparse.Namespace(
        robot="unitree_g1",
        actual_human_height=None,
        vis_fps=None,
        ctrl_fps=None,
        lookback_ms=None,
        retarget_buffer_window_s=None,
        log_interval_s=None,
        req_bind_addr=None,
        rep_bind_addr=None,
        ctrl_bind_addr=None,
        ctrl_pub_bind_addr="",
        policy_config=None,
        web_visualize=False,
        calibration_button=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


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
    server.latest_vr_calibration_request_id = 0
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
    canonical = tuple(policy_joint_names(DEFAULT_POLICY_CONFIG))
    assert canonical == tuple(UFO_EXPECTED_G1_JOINT_NAMES)
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
    assert 'ENABLE_PICO_POLICY_CONTROL=1 requires CTRL_PUB_BIND_ADDR=tcp://*:28704' in policy_launcher
    assert 'cmd+=(--pico-control --pico-control-addr "${PICO_CONTROL_ADDR}")' in policy_launcher
    assert '"${ENABLE_PICO_POLICY_CONTROL}" == "1"' in policy_launcher
    assert 'CTRL_PUB_BIND_ADDR="${CTRL_PUB_BIND_ADDR:-}"' in teleop_launcher
    assert 'CTRL_PUB_BIND_ADDR was set but ENABLE_PICO_POLICY_CONTROL is not 1' in teleop_launcher
    assert '--actual_human_height "${ACTUAL_HUMAN_HEIGHT:-1.75}"' not in teleop_launcher
    assert '--lookback_ms "${LOOKBACK_MS:-50}"' not in teleop_launcher
    assert '--policy-config "${TELEOP_POLICY_CONFIG}"' in teleop_launcher
    assert '[[ -n "${ACTUAL_HUMAN_HEIGHT:-}" ]]' in teleop_launcher
    assert '[[ -n "${LOOKBACK_MS:-}" ]]' in teleop_launcher
    assert "pico_policy_control:" in teleop_server
    assert '--pico-follow-button", type=str, default="right_key_one"' in realtime_server
    assert '--pico-freeze-button", type=str, default="left_key_one"' in realtime_server
    assert '_set_mode_from_input(mode_state, "follow", args.pico_follow_button)' in realtime_server
    assert '_set_mode_from_input(mode_state, "freeze", args.pico_freeze_button)' in realtime_server
    assert '--pose-buffer-lookback-ms "${POSE_BUFFER_LOOKBACK_MS:-0}"' in realtime_launcher
    assert "--enable-pose-buffer" in realtime_launcher
    assert teleop_cfg["retarget"]["calibration"]["button"] is None


def test_server_uses_teleop_yaml_defaults_when_cli_omits_values():
    args = _server_args()
    server = LowLatencyTeleopPoseZMQServer(args)
    cfg = server.teleop_config

    assert server.actual_human_height == cfg.actual_human_height == 1.75
    assert server.ctrl_fps == cfg.ctrl_fps == 50
    assert server.vis_fps == cfg.vis_fps == 5
    assert server.lookback_ns == int(cfg.lookback_ms * 1e6)
    assert server.retarget_buffer_window_ns == int(cfg.retarget_buffer_window_s * 1e9)
    assert server.log_interval_s == cfg.log_interval_s
    assert server.req_bind_addr == cfg.req_bind_addr
    assert server.rep_bind_addr == cfg.rep_bind_addr
    assert server.ctrl_bind_addr == cfg.ctrl_bind_addr
    assert server.ufo_output_joint_names == policy_joint_names(DEFAULT_POLICY_CONFIG)
    assert not server.web_visualize


def test_server_reads_explicit_policy_config_for_joint_order():
    default_names = tuple(policy_joint_names(DEFAULT_POLICY_CONFIG))
    reversed_names = tuple(reversed(default_names))
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "policy.yaml"
        cfg_path.write_text(
            yaml.safe_dump({"policy_joint_names": list(reversed_names)}),
            encoding="utf-8",
        )
        server = LowLatencyTeleopPoseZMQServer(_server_args(policy_config=str(cfg_path)))

    assert server.policy_config_path == str(cfg_path)
    assert server.ufo_output_joint_names == reversed_names
    np.testing.assert_array_equal(
        server.joint_permutation,
        build_joint_permutation(server.canonical_joint_names, reversed_names),
    )


def test_server_visualize_yaml_sets_web_viewer_default():
    original_config_root = robot_config_module.CONFIG_ROOT
    raw = yaml.safe_load((original_config_root / "g1.yaml").read_text(encoding="utf-8"))
    raw["server"]["visualize"] = True

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_config_root = Path(tmpdir)
        (temp_config_root / "g1.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
        robot_config_module.CONFIG_ROOT = temp_config_root
        try:
            server = LowLatencyTeleopPoseZMQServer(_server_args(web_visualize=False))
        finally:
            robot_config_module.CONFIG_ROOT = original_config_root

    assert server.web_visualize


def test_calibration_edge_advances_seq_even_when_timestamp_repeats():
    server = LowLatencyTeleopPoseZMQServer.__new__(LowLatencyTeleopPoseZMQServer)
    server.latest_vr_lock = threading.Lock()
    server.last_controller_buttons = default_controller_buttons()
    server.callback_count = 0
    server.latest_vr_poses = None
    server.latest_vr_recv_ns = 0
    server.latest_vr_seq = 0
    server.latest_vr_motion_timestamp_ns = None
    server.latest_vr_calibration_request_id = 0
    server.prev_calibration_button_pressed = False
    server.calibration_button = "right_key_one"
    server.vr_frame_event = threading.Event()

    server._on_vr_frame(_snapshot(timestamp_ns=42, right_a=False))
    server._on_vr_frame(_snapshot(timestamp_ns=42, right_a=True))

    assert server.callback_count == 2
    assert server.latest_vr_seq == 2
    assert server.latest_vr_motion_timestamp_ns == 42
    assert server.latest_vr_calibration_request_id == 1


def test_worker_calibration_request_id_is_monotonic():
    runtime = XRobotRetargetWorkerRuntime.__new__(XRobotRetargetWorkerRuntime)
    runtime.last_processed_calibration_request_id = 0

    assert runtime._packet_requests_calibration({"calibration_request_id": 1})
    assert not runtime._packet_requests_calibration({"calibration_request_id": 1})
    assert runtime._packet_requests_calibration({"calibration_request_id": 2})
    assert not runtime._packet_requests_calibration({"calibration_request_id": 0})


if __name__ == "__main__":
    test_teleop_config_loads()
    test_retarget_xml_loads_without_external_motion_tracking()
    test_xr_parser_callback_and_polling_snapshots_match()
    test_xr_parser_rejects_nonfinite_pose()
    test_timestamp_dedup_happens_before_worker_wakeup()
    test_synthetic_neutral_retarget_outputs_finite_qpos()
    test_joint_permutation_validation_and_serialization()
    test_pico_policy_control_default_disabled_and_realtime_lookback_zero()
    test_server_uses_teleop_yaml_defaults_when_cli_omits_values()
    test_server_reads_explicit_policy_config_for_joint_order()
    test_server_visualize_yaml_sets_web_viewer_default()
    test_calibration_edge_advances_seq_even_when_timestamp_repeats()
    test_worker_calibration_request_id_is_monotonic()
    print("motion_tracking_retarget tests ok")
