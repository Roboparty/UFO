from pathlib import Path
import sys

import numpy as np
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.realtime.realtime_z_server import (  # noqa: E402
    ModeState,
    OnlineZInferer,
    ResumeRampState,
    _blend_z,
    _extract_latest_frame,
    _is_pose_stale,
    parse_args,
    _quat_mul_xyzw,
    _quat_to_ang_vel_xyzw,
    _quat_rotate_xyzw,
    _compute_humanoid_observations_max_np,
    _quat_rotate_inverse_xyzw,
    PoseFrame,
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


def test_angular_velocity_world_privileged_and_local_state_agree():
    previous = np.array([[0., 0., np.sin(np.pi/4), np.cos(np.pi/4)]])
    current = _quat_mul_xyzw(previous, np.array([[np.sin(.01), 0., 0., np.cos(.01)]]))
    world = _quat_to_ang_vel_xyzw(previous, current, .02, delta_frame='world')
    local = _quat_to_ang_vel_xyzw(previous, current, .02, delta_frame='local')
    np.testing.assert_allclose(_quat_rotate_xyzw(previous, local), world, atol=1e-6)
    obs = _compute_humanoid_observations_max_np(np.zeros((1,1,3)), current[:,None,:], np.zeros((1,1,3)), world[:,None,:], True, False)
    np.testing.assert_allclose(obs['local_body_ang_vel'], [[1,0,0]], atol=1e-6)
    np.testing.assert_allclose(_quat_rotate_inverse_xyzw(current, world), [[1,0,0]], atol=1e-6)


@pytest.mark.parametrize('frame', ['local', 'world'])
def test_online_step_sends_training_coordinates_and_scales_to_encoder(frame):
    inferer = OnlineZInferer.__new__(OnlineZInferer)
    inferer.dt = .02
    inferer.drop_first_n = 0
    inferer.use_wall_clock_dt = False
    inferer.freeze_z_on_static_pose = False
    inferer.fix_quat_continuity = True
    inferer.angvel_delta_frame = frame
    inferer.root_height_obs = False
    inferer.default_pose_29 = np.zeros(29)
    inferer.obs_scales = {'dof_pos': 2., 'dof_vel': 3., 'projected_gravity': 1., 'base_ang_vel': .25, 'max_local_self': 2.}
    inferer.quat_flip_total = inferer.quat_flip_step_total = 0
    inferer.max_z_delta = 0
    inferer.debug_z = False
    inferer.last_z = np.ones(256, dtype=np.float32)
    inferer.reset_history()
    previous = np.array([[0., 0., np.sin(np.pi/4), np.cos(np.pi/4)]])
    current = _quat_mul_xyzw(previous, np.array([[np.sin(.01), 0., 0., np.cos(.01)]]))
    rotations = iter([previous, current])
    inferer._fk_all_bodies = lambda *args: (np.zeros((1,3)), next(rotations))
    feeds = []
    inferer.session = SimpleNamespace(run=lambda outputs, feed: feeds.append(feed) or [np.ones((1,256), np.float32)])
    inferer.session_input_names = {'state', 'privileged_state', 'last_action'}
    inferer.session_output_name = 'z'
    pose = PoseFrame(np.zeros(3), np.array([1.,0.,0.,0.]), np.ones(29))
    inferer.step(pose)
    inferer.step(pose)
    assert len(feeds) == 1
    np.testing.assert_allclose(feeds[0]['state'][0, -3:], [.25,0,0], atol=1e-6)
    np.testing.assert_allclose(feeds[0]['privileged_state'][0, -3:], [2,0,0], atol=1e-6)
    np.testing.assert_allclose(feeds[0]['state'][0, :29], 2)
    np.testing.assert_array_equal(feeds[0]['last_action'], np.zeros((1,29)))

    from scripts.realtime import realtime_z_server as runtime
    from tools.check_realtime_z_obs_alignment import RealtimeObsBuilder, _run_encoder
    inferer.base_body_names, inferer.extend_cfg = ['root'], []
    rotations = iter([previous, current])
    replay = RealtimeObsBuilder(runtime, inferer, .02).build_sequence(
        np.zeros((2, 3)), np.tile([0., 0., 0., 1.], (2, 1)), np.ones((2, 29)),
    )
    for name in ['state', 'privileged_state', 'last_action']:
        np.testing.assert_allclose(replay[name][1:2], feeds[0][name], atol=1e-6)
    _run_encoder(inferer.session, inferer.session_input_names, 'z', replay['state'], replay['privileged_state'])
    np.testing.assert_array_equal(feeds[-1]['last_action'], np.zeros((2, 29)))


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


def test_resume_ramp_waits_for_first_live_z_before_timing():
    seed_z = np.zeros(256, dtype=np.float32)
    live_z = np.ones(256, dtype=np.float32)
    ramp = ResumeRampState(duration_s=0.5)
    ramp.start_pending(seed_z)

    np.testing.assert_allclose(
        ramp.apply(live_z, produced_live_z=False, now=100.0),
        seed_z,
    )
    assert ramp.pending
    assert ramp.ramp_start_z is None

    np.testing.assert_allclose(
        ramp.apply(live_z, produced_live_z=True, now=100.0),
        seed_z,
    )
    assert not ramp.pending
    assert ramp.ramp_start_z is not None
    assert ramp.ramp_start_t == 100.0

    np.testing.assert_allclose(
        ramp.apply(live_z, produced_live_z=True, now=100.25),
        np.full(256, 0.5, dtype=np.float32),
    )


if __name__ == "__main__":
    test_pose_stale_check_blocks_publish_path()
    test_extract_latest_frame_rejects_nonfinite_pose_values()
    test_extract_latest_frame_accepts_and_normalizes_valid_pose()
    test_online_inferer_rejects_invalid_z_without_touching_last_z()
    test_realtime_z_starts_frozen_and_supports_explicit_follow()
    test_resume_reset_clears_velocity_history_and_seeds_last_z()
    test_resume_ramp_blends_old_z_to_live_z()
    test_resume_ramp_waits_for_first_live_z_before_timing()
    print("realtime_z_server safety tests ok")
