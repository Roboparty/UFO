from collections import deque
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import zmq
import yaml
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_policy.ufo_policy import UFODeployPolicy
from rl_policy.observations import ObsGroup
from rl_policy.utils.state_processor import StateProcessor
from utils.common import LowStateMessage


class _FakeStateProcessor:
    def __init__(self, joint_pos):
        self.joint_pos = np.asarray(joint_pos, dtype=np.float64)

    def _prepare_low_state(self):
        return True


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
    policy = UFODeployPolicy.__new__(UFODeployPolicy)
    policy.num_dofs = num_dofs
    policy.num_actions = num_dofs
    policy.policy_armed = True
    policy.start_motion = True
    policy.hold_init_pose = True
    policy.init_count = 0
    policy.task_type = "tracking"
    policy.t = 5
    policy.t_start = 0
    policy.t_stop = 0
    policy.last_action = np.ones(num_dofs, dtype=np.float32)
    policy.observations = {}
    policy.joint_pos_lower_limit = np.full(num_dofs, -10.0)
    policy.joint_pos_upper_limit = np.full(num_dofs, 10.0)
    policy.joint_velocity_limit = np.ones(num_dofs, dtype=np.float64)
    policy.q_target_slew_safety_factor = 0.5
    policy.rl_dt = 0.1
    policy.last_cmd_q = None
    policy.last_pico_buttons = {}
    policy.stop_latched = False
    policy.stop_latch_enable_released_after_stop = True
    policy._last_safe_stop_warning = 0.0
    policy._last_invalid_z_warning = 0.0
    policy._last_stop_latch_warning = 0.0
    policy.state_processor = _FakeStateProcessor(np.zeros(num_dofs))
    policy.command_sender = _FakeCommandSender()
    return policy


def _loop_policy():
    policy = _policy()
    policy.use_joystick = False
    policy.pico_control_enabled = False
    policy.state_processor._prepare_low_state = lambda: True
    policy.prepare_obs_for_rl = lambda: ({}, np.zeros((1, 1)))
    policy.policy = lambda _: np.ones((1, 3))
    policy.action_rescale = 5
    policy.default_dof_angles = np.ones(3)
    policy.action_scale = np.ones(3)
    policy.controlled_joint_indices = np.arange(3)
    policy.observations = {}
    return policy


@pytest.mark.parametrize("key", ["A", "i"])
@pytest.mark.parametrize("valid_state", [True, False])
def test_initialize_after_stop_requires_valid_state_and_keeps_policy_disabled(key, valid_state):
    policy = _loop_policy()
    policy.enter_stop_latch("test R2")
    policy.state_processor._prepare_low_state = lambda: valid_state
    if key == "A":
        policy.handle_control_button(key)
    else:
        policy.handle_keyboard_button(key)
    assert policy.stop_latched is not valid_state
    assert policy.hold_init_pose is valid_state
    assert not policy.policy_armed
    assert not policy.start_motion
    assert policy.t == policy.t_stop
    assert policy.init_count == 0
    np.testing.assert_array_equal(policy.last_action, np.zeros(3))


def test_stop_latch_overrides_inconsistent_initialization_flag():
    policy = _loop_policy()
    policy.enter_stop_latch("test")
    policy.hold_init_pose = True
    policy.init_count = 500
    policy._rl_step_scheduled()
    np.testing.assert_array_equal(policy.command_sender.commands[-1][0], np.zeros(3))


def test_wireless_r2_release_then_a_initializes_before_r1_and_b():
    policy = _policy()
    buttons = ("A", "B", "X", "Y", "L1", "L2", "R1", "R2")
    policy.last_wc_msg = SimpleNamespace(**dict.fromkeys(buttons, False))

    def press(*pressed):
        state = SimpleNamespace(**{key: key in pressed for key in buttons})
        policy.robot = SimpleNamespace(read_wireless_controller=lambda: state)
        return policy.process_joystick_input()

    assert press("R2", "A")  # R2 wins even if A is pressed at the same time.
    assert policy.stop_latched and not policy.hold_init_pose
    press("A")  # Holding A through R2 release must not initialize automatically.
    assert policy.stop_latched and not policy.hold_init_pose
    press()
    press("A")
    assert not policy.stop_latched and policy.hold_init_pose
    assert not policy.policy_armed and not policy.start_motion
    press()
    press("R1")
    assert policy.policy_armed and not policy.hold_init_pose
    assert not policy.start_motion
    press()
    press("B")
    assert policy.policy_armed and policy.start_motion


@pytest.mark.parametrize("keyboard", [False, True])
def test_reward_holds_initial_pose_until_b_and_can_reinitialize_after_stop(keyboard):
    policy = _loop_policy()
    policy.task_type = "reward"
    policy.selected_z_names = ["move-ego-0-0.7"]
    policy.selected_z = np.ones((1, 256))
    policy.z_index = 0
    policy.joint_velocity_limit = np.full(3, 200.)

    def press(button):
        if keyboard and button != "R2":
            policy.handle_keyboard_button({"A": "i", "R1": "]", "B": "["}[button])
        else:
            policy.handle_control_button(button)

    press("R2")
    for _ in range(2):
        press("A")
        policy.init_count = 500
        press("B")  # Initialization alone must not arm the reward policy.
        assert not policy.policy_armed
        assert policy.hold_init_pose
        press("R1")
        for _ in range(3):
            policy._rl_step_scheduled()
            np.testing.assert_array_equal(policy.command_sender.commands[-1][0], np.ones(3))
            np.testing.assert_array_equal(policy.last_action, np.zeros(3))
        press("B")
        policy._rl_step_scheduled()
        np.testing.assert_array_equal(policy.command_sender.commands[-1][0], np.full(3, 6.))
        np.testing.assert_array_equal(policy.last_action, np.full((1, 3), 5.))
        press("R1")  # Returning to the waiting pose must stop walking again.
        policy._rl_step_scheduled()
        np.testing.assert_array_equal(policy.command_sender.commands[-1][0], np.ones(3))
        press("R2")
        press("B")
        policy._rl_step_scheduled()
        assert policy.stop_latched and not policy.policy_armed
        np.testing.assert_array_equal(policy.command_sender.commands[-1][0], np.zeros(3))


def test_reward_initialization_requires_r1_again_after_walking():
    policy = _loop_policy()
    policy.task_type = "reward"
    policy.hold_init_pose = False
    policy.handle_control_button("A")
    policy.handle_control_button("B")
    assert policy.hold_init_pose and not policy.policy_armed


def test_missing_state_latches_and_recovery_does_not_resume():
    policy = _loop_policy()
    policy.has_valid_low_state = True
    policy.state_processor._prepare_low_state = lambda: False
    policy._rl_step_scheduled()
    assert policy.stop_latched and not policy.policy_armed
    np.testing.assert_array_equal(policy.command_sender.commands[-1][0], np.zeros(3))
    policy.handle_keyboard_button('r')
    assert policy.stop_latched
    policy.state_processor._prepare_low_state = lambda: True
    policy._rl_step_scheduled()
    assert policy.stop_latched and not policy.policy_armed
    np.testing.assert_array_equal(policy.command_sender.commands[-1][0], np.zeros(3))
    policy.handle_keyboard_button('r')
    assert not policy.stop_latched and not policy.policy_armed


@pytest.mark.parametrize('failure', ['nan', 'wrong_shape', 'exception'])
def test_inference_failure_requires_explicit_rearm(failure):
    policy = _loop_policy()
    if failure == 'exception':
        def fail(_):
            raise RuntimeError('inference failed')
        policy.policy = fail
    else:
        policy.policy = lambda _: np.full((1, 3), np.nan) if failure == 'nan' else np.ones((1, 1))
    policy._rl_step_scheduled()
    assert policy.stop_latched and not policy.policy_armed
    policy.policy = lambda _: np.zeros((1, 3))
    policy.handle_control_button('R1')
    assert not policy.policy_armed
    policy.handle_keyboard_button('r')
    assert not policy.stop_latched and not policy.policy_armed


@pytest.mark.parametrize('initializing', [False, True])
def test_inactive_policy_does_not_record_unexecuted_actions(initializing):
    policy = _loop_policy()
    policy.policy_armed = initializing
    policy.hold_init_pose = initializing
    policy._rl_step_scheduled()
    np.testing.assert_array_equal(policy.last_action, np.zeros(3))
    policy.hold_init_pose = False
    policy.policy_armed = True
    policy._rl_step_scheduled()
    np.testing.assert_array_equal(policy.last_action, np.full((1, 3), 5.))


def test_enabling_policy_clears_inactive_action_and_observation_history():
    policy = _loop_policy()
    policy.policy_armed = False
    cleared = []
    policy.observations = {'policy': SimpleNamespace(reset=lambda: cleared.append(True))}
    policy.handle_control_button('R1')
    np.testing.assert_array_equal(policy.last_action, np.zeros(3))
    assert cleared == [True]
    policy.handle_control_button('R1')
    assert cleared == [True]


@pytest.mark.parametrize('duration, latched', [(.025, False), (.12, True)])
def test_control_overrun_does_not_burst_commands(monkeypatch, duration, latched):
    from rl_policy import ufo_policy
    policy = _loop_policy()
    policy.rl_dt = .02
    clock, calls = [0.], []
    monkeypatch.setattr(ufo_policy.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(ufo_policy.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    policy._install_signal_handlers = lambda: None

    def step():
        calls.append(clock[0])
        if len(calls) == 1:
            clock[0] += duration
        if len(calls) == 8:
            policy._exit_requested = True

    policy._rl_step_scheduled = step
    policy.run()
    assert policy.stop_latched == latched
    assert np.all(np.diff(calls) >= .02 - 1e-9)


def test_disabling_sim_support_clears_applied_force():
    import mujoco
    from sim_env.simulator import Simulator
    sim = Simulator.__new__(Simulator)
    sim.mj_model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    sim.mj_data = mujoco.MjData(sim.mj_model)
    sim.scene_config = {'ENABLE_ELASTIC_BAND': True}
    sim.band_attached_link = 1
    sim.elastic_band = SimpleNamespace(enable=True, Advance=lambda *_: np.array([0., 0., 300.]))
    sim.sim_bridge = SimpleNamespace(publish_low_state=lambda: None, compute_torques=lambda: None,
                                     torques=np.zeros(sim.mj_model.nu))
    sim.sim_step()
    assert sim.mj_data.xfrc_applied[1, 2] == 300.
    sim.elastic_band.enable = False
    sim.sim_step()
    np.testing.assert_array_equal(sim.mj_data.xfrc_applied[1, :3], np.zeros(3))


def test_observation_history_excludes_current_frame_and_keeps_previous_actions():
    policy = _policy(29)
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/policy/g1_policy.yaml').read_text())
    policy.obs_cfg = config['observation']
    processor = policy.state_processor
    processor.num_dof = 29
    processor.root_quat_b = np.array([1., 0., 0., 0.])
    processor.root_ang_vel_b = np.zeros(3)
    processor.joint_vel = np.zeros(29)
    default = np.array(config['observation']['policy']['dof_pos_minus_default']['default_pos'])
    policy.ctx = np.ones((10, 256), dtype=np.float32)
    policy.window_size, policy.gamma = 1, .8
    policy.policy_armed = False
    policy.setup_observations()
    group = policy.observations['policy']
    offsets, cursor = {}, 0
    for name, func in group.funcs.items():
        count = func.compute().size
        offsets[name] = slice(cursor, cursor + count)
        cursor += count
    for frame in range(1, 6):
        processor.joint_pos = default + frame
        policy.last_action = np.full(29, 10 * frame)
        obs, _ = policy.prepare_obs_for_rl()
        if frame == 1:
            np.testing.assert_array_equal(obs['policy'][0, offsets['dof_pos_minus_default_history']], 0)
    values = obs['policy'][0]
    np.testing.assert_allclose(values[offsets['dof_pos_minus_default']], 5)
    np.testing.assert_allclose(values[offsets['dof_pos_minus_default_history']].reshape(4, 29)[:, 0], [4, 3, 2, 1])
    np.testing.assert_allclose(values[offsets['prev_actions']], 50)
    np.testing.assert_allclose(values[offsets['prev_actions_history']].reshape(4, 29)[:, 0], [40, 30, 20, 10])
    policy.reset()
    for name, func in group.funcs.items():
        if name.endswith('_history'):
            np.testing.assert_array_equal(func.compute(), 0)


def test_all_observations_keep_scales_history_lengths_and_reset():
    policy = _policy(2)
    state = policy.state_processor
    state.num_dof = 2
    state.root_quat_b = np.array([2 ** -.5, 0., 2 ** -.5, 0.])
    names = ["base_ang_vel", "projected_gravity", "dof_pos_minus_default", "dof_vel", "prev_actions"]
    config, past = {}, {name: [] for name in names}
    for index, name in enumerate(names):
        options = {"scale": (index + 1) / 10}
        if name == "dof_pos_minus_default":
            options["default_pos"] = [1., 2.]
        config[name] = options
        config[name + "_history"] = {**options, "steps": index + 1}
    group = ObsGroup("policy", config, policy)
    for frame in range(1, 8):
        state.root_ang_vel_b = np.arange(3) + frame
        state.joint_pos = np.array([1., 2.]) + frame
        state.joint_vel = np.full(2, 2 * frame)
        policy.last_action = np.full((1, 2), 3 * frame)
        current = [np.arange(3) + frame, np.array([1., 0., 0.]),
                   np.full(2, frame), np.full(2, 2 * frame), np.full(2, 3 * frame)]
        expected = []
        for index, (name, value) in enumerate(zip(names, current)):
            scale, steps = (index + 1) / 10, index + 1
            history = past[name][:steps] + [np.zeros_like(value)] * max(0, steps - len(past[name]))
            expected.extend([value * scale, np.concatenate(history) * scale])
        np.testing.assert_allclose(group.compute(), np.concatenate(expected), atol=1e-15)
        group.update()
        for name, value in zip(names, current):
            past[name].insert(0, value.copy())
    group.reset()
    for name in names:
        np.testing.assert_array_equal(group.funcs[name + "_history"].compute(), 0)


@pytest.mark.parametrize("config", [
    {"unknown": {}}, {"dof_vel_history": {"steps": 0}},
    {"dof_pos_minus_default": {"default_pos": [0]}},
])
def test_invalid_observation_config_fails_at_setup(config):
    policy = _policy(2)
    policy.state_processor.num_dof = 2
    with pytest.raises(ValueError):
        ObsGroup("policy", config, policy)


@pytest.mark.parametrize("source", ["pkl", "zmq"])
def test_tracking_smoothing_keeps_offline_future_and_live_past_windows(source):
    policy = _policy()
    policy.ctx_source = source
    policy.policy_armed = False
    policy.observations = {"policy": SimpleNamespace(compute=lambda: np.zeros(2), update=lambda: None)}
    policy.ctx = np.eye(256, dtype=np.float32)[:5] * 16
    policy.window_size, policy.gamma = 3, .8
    policy.ctx_window = deque(maxlen=3)
    policy.ctx_norm_ref = 16.
    policy._poll_realtime_z = lambda: None
    policy._check_realtime_z_watchdog = lambda: None
    for frame in range(5):
        policy.t = frame
        policy.ctx_latest = policy.ctx[frame]
        _, inputs = policy.prepare_obs_for_rl()
        indices = list(range(frame, min(frame + 3, 5))) if source == "pkl" else list(range(frame, max(frame - 3, -1), -1))
        expected = np.zeros(256)
        for offset, index in enumerate(indices):
            expected[index] = 16 * .8 ** offset
        expected /= sum(.8 ** offset for offset in range(len(indices)))
        expected = expected / (np.linalg.norm(expected) + 1e-8) * 16
        np.testing.assert_allclose(inputs[0, 2:], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("task", ["reward", "goal"])
def test_context_switch_and_reset_use_matching_vector_and_label(task):
    policy = _policy()
    policy.task_type = task
    policy.selected_z = np.eye(3)
    policy.selected_z_names = ["a", "b", "c"]
    policy.z_index = 0
    policy.observations = {"policy": SimpleNamespace(compute=lambda: np.zeros(2), update=lambda: None)}
    for expected_index in [1, 2, 0]:
        policy.handle_keyboard_button("n")
        assert policy.z_index == expected_index
        _, inputs = policy.prepare_obs_for_rl()
        np.testing.assert_array_equal(inputs[0, 2:], policy.selected_z[expected_index])
    policy.handle_keyboard_button("n")
    policy.handle_keyboard_button("p")
    assert policy.z_index == 0


def _state_processor():
    processor = StateProcessor.__new__(StateProcessor)
    processor.state_timeout_s = .2
    processor.last_state_time = None
    processor.last_state_tick = None
    processor.latest_low_state = None
    processor.joint_indices_in_source = [0, 1, 2]
    processor.root_quat_b = np.zeros(4)
    processor.root_ang_vel_b = np.zeros(3)
    processor.joint_pos = np.zeros(3)
    processor.joint_vel = np.zeros(3)
    return processor


def test_cached_zmq_state_expires_even_if_same_packet_is_resent(monkeypatch):
    now = [1.]
    monkeypatch.setattr('rl_policy.utils.state_processor.time.monotonic', lambda: now[0])
    processor = _state_processor()
    packet = LowStateMessage([1, 0, 0, 0], [0, 0, 0], np.zeros(3), np.zeros(3), tick=1).to_bytes()
    processor.low_state_socket = _FakeZmqSocket([packet])
    assert processor._prepare_low_state()
    now[0] = 1.19
    assert processor._prepare_low_state()
    now[0] = 1.21
    processor.low_state_socket.messages.append(packet)
    assert not processor._prepare_low_state()
    processor.low_state_socket.messages.append(LowStateMessage([1,0,0,0], [0,0,0], np.ones(3), np.zeros(3), tick=2).to_bytes())
    assert processor._prepare_low_state()
    np.testing.assert_array_equal(processor.joint_pos, np.ones(3))


def test_sdk_tick_detects_cached_state_and_rejects_invalid_packets(monkeypatch):
    now = [1.]
    monkeypatch.setattr('rl_policy.utils.state_processor.time.monotonic', lambda: now[0])
    processor = _state_processor()
    state = SimpleNamespace(tick=1, imu=SimpleNamespace(quat=[1,0,0,0], omega=[0,0,0]), motor=SimpleNamespace(q=np.zeros(3), dq=np.zeros(3)))
    processor.robot = SimpleNamespace(read_low_state=lambda: state)
    assert processor._prepare_low_state()
    now[0] = 1.21
    assert not processor._prepare_low_state()
    state.tick = 2
    state.motor.q[0] = np.nan
    assert not processor._prepare_low_state()
    assert processor.last_state_tick == 1
    state.motor.q[0] = 1.
    assert processor._prepare_low_state()


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

    assert not policy.policy_armed
    assert not policy.start_motion
    assert not policy.hold_init_pose
    assert policy.t == policy.t_stop
    assert len(policy.ctx_window) == 0
    assert policy.ctx_latest.shape == (256,)
    assert policy.ctx_latest[0] == 16.0
    assert policy.stop_latched
    policy.handle_control_button('R1')
    assert not policy.policy_armed


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
    assert not policy.policy_armed
    assert not policy.start_motion

    policy.handle_pico_buttons(
        {"right_key_one": False, "right_key_two": False, "left_key_one": False}
    )
    policy.handle_pico_buttons(
        {"right_key_one": True, "right_key_two": True, "left_key_one": False}
    )
    assert policy.policy_armed
    assert policy.start_motion


def test_r2_latch_blocks_wireless_r1_enable():
    policy = _policy()
    policy.enter_stop_latch("test R2")

    policy.handle_control_button("R1")

    assert not policy.policy_armed
    assert policy.stop_latched


def test_q_target_slew_limit_uses_joint_velocity_limit():
    policy = _policy(num_dofs=2)
    policy.last_cmd_q = np.array([0.0, 0.0])
    policy.joint_velocity_limit = np.array([10.0, 2.0])
    policy.rl_dt = 0.1
    policy.q_target_slew_safety_factor = 0.5

    limited = policy._apply_q_target_slew_limit(np.array([10.0, -10.0]))

    np.testing.assert_allclose(limited, np.array([0.5, -0.1]))


def test_g1_real_requires_explicit_real_robot_env():
    old = os.environ.pop("UFO_REAL_ROBOT_OK", None)
    try:
        try:
            UFODeployPolicy(
                robot_config={"ROBOT_TYPE": "g1_real"},
                policy_config={},
                exp_config={},
                model_path="",
            )
        except RuntimeError as exc:
            assert "UFO_REAL_ROBOT_OK=1" in str(exc)
        else:
            raise AssertionError("g1_real policy started without UFO_REAL_ROBOT_OK=1")
    finally:
        if old is not None:
            os.environ["UFO_REAL_ROBOT_OK"] = old


if __name__ == "__main__":
    test_stale_realtime_z_enters_safe_stop()
    test_nan_realtime_z_is_ignored()
    test_r2_latch_blocks_pico_enable_until_release_and_repress()
    test_r2_latch_blocks_wireless_r1_enable()
    test_q_target_slew_limit_uses_joint_velocity_limit()
    test_g1_real_requires_explicit_real_robot_env()
    print("ufo_policy safety tests ok")
