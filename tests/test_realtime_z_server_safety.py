from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.realtime.realtime_z_server import (  # noqa: E402
    ModeState,
    OnlineZInferer,
    _blend_z,
    _extract_latest_frame,
    _is_pose_stale,
    parse_args,
)


def _valid_payload():
    return {
        "frames": [
            {
                "root_pos": [0.0, 0.0, 0.8],
                "root_quat": [1.0, 0.0, 0.0, 0.0],
                "dof_pos": [0.0] * 29,
            }
        ]
    }


def test_pose_stale_check_blocks_publish_path():
    assert _is_pose_stale(last_valid_pose_monotonic=1.0, max_pose_stale_s=0.2, now=1.21)
    assert not _is_pose_stale(last_valid_pose_monotonic=1.0, max_pose_stale_s=0.2, now=1.19)
    assert not _is_pose_stale(last_valid_pose_monotonic=1.0, max_pose_stale_s=None, now=100.0)


def test_extract_latest_frame_rejects_nonfinite_pose_values():
    payload = _valid_payload()
    payload["frames"][0]["root_pos"][0] = float("nan")
    assert _extract_latest_frame(payload) is None

    payload = _valid_payload()
    payload["frames"][0]["root_quat"][1] = float("inf")
    assert _extract_latest_frame(payload) is None

    payload = _valid_payload()
    payload["frames"][0]["dof_pos"][3] = float("-inf")
    assert _extract_latest_frame(payload) is None


def test_extract_latest_frame_accepts_and_normalizes_valid_pose():
    frame = _extract_latest_frame(_valid_payload())
    assert frame is not None
    np.testing.assert_allclose(frame.root_quat_wxyz, np.array([1.0, 0.0, 0.0, 0.0]))
    assert frame.dof_pos.shape == (29,)


def test_online_inferer_rejects_invalid_z_without_touching_last_z():
    inferer = OnlineZInferer.__new__(OnlineZInferer)
    inferer._last_invalid_z_warning = 0.0
    inferer.last_z = np.ones(256, dtype=np.float32)

    assert inferer._validate_z_output(np.zeros(255, dtype=np.float32)) is None
    np.testing.assert_array_equal(inferer.last_z, np.ones(256, dtype=np.float32))

    assert inferer._validate_z_output(np.full(256, np.nan, dtype=np.float32)) is None
    np.testing.assert_array_equal(inferer.last_z, np.ones(256, dtype=np.float32))


def test_realtime_z_starts_frozen_and_supports_explicit_follow():
    old_argv = sys.argv
    try:
        sys.argv = ["realtime_z_server.py"]
        args = parse_args()
    finally:
        sys.argv = old_argv

    assert args.initial_mode == "freeze"
    mode_state = ModeState(args.initial_mode)
    assert mode_state.get() == "freeze"
    assert mode_state.set("follow")
    assert mode_state.get() == "follow"


def test_resume_reset_clears_velocity_history_and_seeds_last_z():
    inferer = OnlineZInferer.__new__(OnlineZInferer)
    inferer.prev_root_pos = np.ones(3, dtype=np.float32)
    inferer.prev_root_quat_xyzw = np.ones(4, dtype=np.float32)
    inferer.prev_dof_pos = np.ones(29, dtype=np.float32)
    inferer.prev_all_body_pos = np.ones((31, 3), dtype=np.float32)
    inferer.prev_all_body_rot_xyzw = np.ones((31, 4), dtype=np.float32)
    inferer._step_count = 10
    inferer._wall_step_t0 = 123.0
    inferer._prev_z_dbg = np.ones(256, dtype=np.float32)
    inferer.last_z = np.zeros(256, dtype=np.float32)

    seed_z = np.full(256, 2.0, dtype=np.float32)
    inferer.reset_history(seed_z=seed_z)

    assert inferer.prev_root_pos is None
    assert inferer.prev_root_quat_xyzw is None
    assert inferer.prev_dof_pos is None
    assert inferer.prev_all_body_pos is None
    assert inferer.prev_all_body_rot_xyzw is None
    assert inferer._step_count == 0
    assert inferer._wall_step_t0 is None
    assert inferer._prev_z_dbg is None
    np.testing.assert_array_equal(inferer.last_z, seed_z)


def test_resume_ramp_blends_old_z_to_live_z():
    start_z = np.zeros(256, dtype=np.float32)
    live_z = np.ones(256, dtype=np.float32)
    np.testing.assert_allclose(_blend_z(start_z, live_z, 0.0), start_z)
    np.testing.assert_allclose(_blend_z(start_z, live_z, 0.5), np.full(256, 0.5, dtype=np.float32))
    np.testing.assert_allclose(_blend_z(start_z, live_z, 1.0), live_z)


if __name__ == "__main__":
    test_pose_stale_check_blocks_publish_path()
    test_extract_latest_frame_rejects_nonfinite_pose_values()
    test_extract_latest_frame_accepts_and_normalizes_valid_pose()
    test_online_inferer_rejects_invalid_z_without_touching_last_z()
    test_realtime_z_starts_frozen_and_supports_explicit_follow()
    test_resume_reset_clears_velocity_history_and_seeds_last_z()
    test_resume_ramp_blends_old_z_to_live_z()
    print("realtime_z_server safety tests ok")
