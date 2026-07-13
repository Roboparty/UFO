from collections import deque
from pathlib import Path
import sys
import time

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_policy.bfm_zero import BFMZeroPolicy


class _FakeStateProcessor:
    def __init__(self, joint_pos):
        self.joint_pos = np.asarray(joint_pos, dtype=np.float64)


class _FakeCommandSender:
    def __init__(self):
        self.commands = []

    def send_command(self, cmd_q, cmd_dq, cmd_tau):
        self.commands.append(
            (
                np.asarray(cmd_q, dtype=np.float64).copy(),
                np.asarray(cmd_dq, dtype=np.float64).copy(),
                np.asarray(cmd_tau, dtype=np.float64).copy(),
            )
        )


class _FakeZmqSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    def recv(self, flags=0):
        if self.messages:
            return self.messages.pop(0)
        raise zmq.Again()


def _policy(num_dofs=3):
    policy = BFMZeroPolicy.__new__(BFMZeroPolicy)
    policy.num_dofs = num_dofs
    policy.num_actions = num_dofs
    policy.use_policy_action = True
    policy.start_motion = True
    policy.get_ready_state = True
    policy.task_type = "tracking"
    policy.t = 5
    policy.t_start = 0
    policy.t_stop = 0
    policy.last_action = np.ones(num_dofs, dtype=np.float32)
    policy.joint_pos_lower_limit = np.full(num_dofs, -10.0)
    policy.joint_pos_upper_limit = np.full(num_dofs, 10.0)
    policy.joint_velocity_limit = np.ones(num_dofs, dtype=np.float64)
    policy.q_target_slew_safety_factor = 0.5
    policy.rl_dt = 0.1
    policy.last_cmd_q = None
    policy.last_pico_buttons = {}
    policy.stop_latched = False
    policy.pico_enable_released_after_stop = True
    policy._last_safe_stop_warning = 0.0
    policy._last_invalid_z_warning = 0.0
    policy._last_z_timeout_warning = 0.0
    policy._last_stop_latch_warning = 0.0
    policy.state_processor = _FakeStateProcessor(np.zeros(num_dofs))
    policy.command_sender = _FakeCommandSender()
    return policy


def test_stale_realtime_z_enters_safe_stop():
    policy = _policy()
    policy.ctx_source = "zmq"
    policy.ctx_norm_ref = 16.0
    policy.ctx_latest = np.ones(256, dtype=np.float32)
    policy.ctx_window = deque([np.ones(256, dtype=np.float32)], maxlen=3)
    policy.ctx_zmq_timeout_s = 0.001
    policy.ctx_zmq_start_monotonic = time.monotonic() - 1.0
    policy.ctx_last_z_monotonic = time.monotonic() - 1.0

    policy._check_realtime_z_watchdog()

    assert not policy.use_policy_action
    assert not policy.start_motion
    assert not policy.get_ready_state
    assert policy.t == policy.t_stop
    assert len(policy.ctx_window) == 0
    assert policy.ctx_latest.shape == (256,)
    assert policy.ctx_latest[0] == 16.0


def test_nan_realtime_z_is_ignored():
    policy = _policy()
    policy.ctx_zmq = _FakeZmqSocket([np.full(256, np.nan, dtype=np.float32).tobytes()])
    policy.ctx_latest = np.arange(256, dtype=np.float32)
    policy.ctx_last_z_monotonic = None

    received = policy._poll_realtime_z()

    assert not received
    np.testing.assert_array_equal(policy.ctx_latest, np.arange(256, dtype=np.float32))
    assert policy.ctx_last_z_monotonic is None


def test_r2_latch_blocks_pico_enable_until_release_and_repress():
    policy = _policy()
    policy.enter_stop_latch("test R2")

    policy.handle_pico_buttons(
        {"right_key_one": True, "right_key_two": True, "left_key_one": False}
    )
    assert not policy.use_policy_action
    assert not policy.start_motion

    policy.handle_pico_buttons(
        {"right_key_one": False, "right_key_two": False, "left_key_one": False}
    )
    policy.handle_pico_buttons(
        {"right_key_one": True, "right_key_two": True, "left_key_one": False}
    )
    assert policy.use_policy_action
    assert policy.start_motion


def test_q_target_slew_limit_uses_joint_velocity_limit():
    policy = _policy(num_dofs=2)
    policy.last_cmd_q = np.array([0.0, 0.0])
    policy.joint_velocity_limit = np.array([10.0, 2.0])
    policy.rl_dt = 0.1
    policy.q_target_slew_safety_factor = 0.5

    limited = policy._apply_q_target_slew_limit(np.array([10.0, -10.0]))

    np.testing.assert_allclose(limited, np.array([0.5, -0.1]))


if __name__ == "__main__":
    test_stale_realtime_z_enters_safe_stop()
    test_nan_realtime_z_is_ignored()
    test_r2_latch_blocks_pico_enable_until_release_and_repress()
    test_q_target_slew_limit_uses_joint_velocity_limit()
    print("bfm_zero safety tests ok")
