from __future__ import annotations

import json
import os
import signal
import sys
import time
from typing import Any

import numpy as np
from loguru import logger
from sshkeyboard import listen_keyboard
from termcolor import colored


from utils.strings import joint_values, resolve_matching_names_values
from utils.policy import resolve_context_path, load_context, select_contexts, validate_policy_config, validate_task_config, validate_context
from rl_policy.observations import ObsGroup
from rl_policy.utils.state_processor import StateProcessor
from rl_policy.utils.command_sender import CommandSender


def _default_realtime_z(norm_ref: float) -> np.ndarray:
    z = np.zeros(256, dtype=np.float32)
    z[0] = float(norm_ref)
    return z


class UFODeployPolicy:
    def __init__(
        self,
        robot_config: dict[str, Any],
        policy_config: dict[str, Any],
        exp_config: dict[str, Any],
        model_path: str,
        rl_rate: int = 50,
        pico_control: bool = False,
        pico_control_addr: str = "tcp://127.0.0.1:28704",
    ) -> None:
        robot_type = robot_config["ROBOT_TYPE"]
        if robot_type == "g1_real" and os.environ.get("UFO_REAL_ROBOT_OK") != "1":
            raise RuntimeError("Refusing to start real robot control without UFO_REAL_ROBOT_OK=1.")
        validate_task_config(exp_config, robot_type)
        self.policy_metadata = validate_policy_config(model_path, policy_config)
        self.z_dim = int(self.policy_metadata["z_dim"])
        if not np.isfinite(rl_rate) or rl_rate <= 0:
            raise ValueError("rl_rate must be finite and positive")
        self.rl_dt = 1.0 / rl_rate
        self.t = 0

        self.policy_config = policy_config

        self.setup_policy(model_path)
        self.obs_cfg = policy_config["observation"]

        self.isaac_joint_names = policy_config["isaac_joint_names"]
        self.num_dofs = len(self.isaac_joint_names)

        self.default_dof_angles = joint_values(policy_config["default_joint_pos"], self.isaac_joint_names)
        self.last_action = np.zeros(self.num_dofs)

        action_scale_cfg = policy_config["action_scale"]
        self.action_scale = np.ones((self.num_dofs))
        self.action_rescale = policy_config["action_rescale"]
        if isinstance(action_scale_cfg, float):
            self.action_scale *= action_scale_cfg
        elif isinstance(action_scale_cfg, dict):
            joint_ids, _, action_scales = resolve_matching_names_values(
                action_scale_cfg, self.isaac_joint_names, preserve_order=True
            )
            self.action_scale[joint_ids] = action_scales
        else:
            raise ValueError(f"Invalid action scale type: {type(action_scale_cfg)}")

        self.policy_joint_names = policy_config["policy_joint_names"]
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [
            self.isaac_joint_names.index(name)
            for name in self.policy_joint_names
        ]

        # Keypress/control state
        self.policy_armed = False
        self.pico_control_enabled = bool(pico_control)
        self.pico_control_addr = str(pico_control_addr)
        self.pico_control_sock = None
        self.last_pico_buttons: dict[str, bool] = {}
        self.stop_latched = False
        self.stop_latch_enable_released_after_stop = True
        self._last_safe_stop_warning = 0.0
        self._last_invalid_z_warning = 0.0
        self._last_stop_latch_warning = 0.0
        self._exit_requested = False
        self._exit_signal_name: str | None = None

        self.init_count = 0
        self.hold_init_pose = False
        self.has_valid_low_state = False

        self.joint_pos_lower_limit = joint_values(robot_config["joint_pos_lower_limit"], self.isaac_joint_names)
        self.joint_pos_upper_limit = joint_values(robot_config["joint_pos_upper_limit"], self.isaac_joint_names)
        self.joint_velocity_limit = joint_values(robot_config["joint_velocity_limit"], self.isaac_joint_names, default=np.inf)
        self.q_target_slew_safety_factor = float(
            policy_config.get(
                "q_target_slew_safety_factor",
                robot_config.get("Q_TARGET_SLEW_SAFETY_FACTOR", 0.5),
            )
        )
        if not np.isfinite(self.q_target_slew_safety_factor) or self.q_target_slew_safety_factor < 0.0:
            raise ValueError("q_target_slew_safety_factor must be a finite non-negative value")
        self.last_cmd_q: np.ndarray | None = None

        # Load and validate artifacts before creating the hardware interface.
        self.setup_task(model_path, exp_config)
        if robot_type == "g1_real":
            sys.path.append("/home/unitree/unitree_sdk2_bfm/build/lib")
            import g1_interface
            if not hasattr(g1_interface.LowState(), "tick"):
                raise RuntimeError("G1 SDK lacks low-state tick; run scripts/onboard/patch_g1_sdk.py and rebuild g1_interface")
            network_interface = robot_config.get("INTERFACE", None)
            self.robot = g1_interface.G1Interface(network_interface)
            try:
                self.robot.set_control_mode(g1_interface.ControlMode.PR)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to set G1 control mode to PR; refusing to start real robot control."
                ) from exc
            robot_config["robot"] = self.robot
        self.state_processor = StateProcessor(robot_config, policy_config["isaac_joint_names"])
        self.command_sender = CommandSender(robot_config, policy_config)

        self.setup_observations()
        self.setup_controls(robot_config)

    def setup_controls(self, robot_config):
        # Start input handlers only after observations and task state are ready.
        if robot_config.get("USE_JOYSTICK", False):
            print("Using joystick")
            self.use_joystick = True
            self.wc_msg = None  # type: ignore
            self.last_wc_msg = self.robot.read_wireless_controller()
            print("Wireless Controller Initialized")
        else:
            import threading
            print("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(
                target=self.start_key_listener, daemon=True
            )
            self.key_listener_thread.start()

        if self.pico_control_enabled:
            self.setup_pico_control()

    def setup_task(self, model_path, exp_config):
        self.task_type = exp_config['type']
        self.start_motion = False
        logger.info(f"task_type={self.task_type}")

        # Task-specific setup
        if self.task_type == "tracking":
            self.gamma = float(exp_config['gamma'])
            self.window_size = exp_config['window_size']
            if type(self.window_size) is not int or self.window_size <= 0:
                raise ValueError("window_size must be a positive integer")
            if not np.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
                raise ValueError("gamma must be finite and between 0 and 1")
            self.t_start, self.t_end, self.t_stop = (exp_config[key] for key in ('start', 'end', 'stop'))
            if any(type(value) is not int or value < 0 for value in (self.t_start, self.t_end, self.t_stop)):
                raise ValueError("tracking start/end/stop must be non-negative integers")
            if self.t_end < self.t_start:
                raise ValueError("tracking end must not precede start")
            self.ctx_source = str(exp_config.get("ctx_source", "pkl")).lower()
            if self.ctx_source not in {"pkl", "zmq"}:
                raise ValueError(f"Invalid ctx_source={self.ctx_source}, expected 'pkl' or 'zmq'")

            if self.ctx_source == "pkl":
                ctx_path = resolve_context_path(
                    model_path,
                    exp_config,
                    default_ctx_dir="tracking_inference",
                    fallback_ctx_dirs=("tracking_inference_mjlab",),
                )
                logger.info(f"Loading tracking context from {ctx_path}")
                self.ctx = np.asarray(load_context(ctx_path))
                if validate_context(self.ctx) != self.z_dim or self.ctx.ndim != 2:
                    raise ValueError("Tracking context must have shape [frames, model z_dim]")
                if max(self.t_start, self.t_stop) >= len(self.ctx):
                    raise ValueError("tracking start/stop must index an existing context frame")
                self.ctx_zmq = None
                self.ctx_window = None
                self.ctx_norm_ref = float(np.linalg.norm(self.ctx[0]))
            else:
                import zmq
                from collections import deque

                if self.z_dim != 256:
                    raise ValueError("Live z currently requires a 256-dimensional encoder and actor")
                self.ctx = None
                self.ctx_norm_ref = float(exp_config.get("ctx_norm_ref", 16.0))
                self.ctx_window = deque(maxlen=self.window_size)
                self.ctx_zmq_timeout_s = float(exp_config.get("ctx_zmq_timeout_ms", 200.0)) / 1000.0
                if not np.isfinite(self.ctx_norm_ref) or self.ctx_norm_ref <= 0:
                    raise ValueError("ctx_norm_ref must be finite and positive")
                if not np.isfinite(self.ctx_zmq_timeout_s) or self.ctx_zmq_timeout_s <= 0:
                    raise ValueError("ctx_zmq_timeout_ms must be finite and positive")
                self.ctx_zmq_start_monotonic = time.monotonic()
                self.ctx_last_z_monotonic: float | None = None

                zmq_addr = str(exp_config.get("ctx_zmq_addr", "tcp://127.0.0.1:28711"))
                logger.info(
                    f"Tracking context from ZMQ: {zmq_addr} "
                    f"(norm_ref={self.ctx_norm_ref}, timeout={self.ctx_zmq_timeout_s * 1000:.0f} ms)"
                )
                zctx = zmq.Context.instance()
                sock = zctx.socket(zmq.SUB)
                sock.setsockopt(zmq.SUBSCRIBE, b"")
                sock.setsockopt(zmq.CONFLATE, 1)
                sock.setsockopt(zmq.RCVTIMEO, 0)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(zmq_addr)
                self.ctx_zmq = sock
                self.ctx_latest = _default_realtime_z(self.ctx_norm_ref)

            logger.info(f"t_start={exp_config['start']}, t_end={exp_config['end']}, t_stop={exp_config['stop']}")
            logger.info(f"gamma={exp_config['gamma']}, window_size={exp_config['window_size']}")

        elif self.task_type in {"reward", "goal"}:
            path = resolve_context_path(
                model_path, exp_config,
                default_ctx_dir=f"{self.task_type}_inference",
                fallback_ctx_dirs=(f"{self.task_type}_inference_mjlab",),
            )
            logger.info(f"Loading {self.task_type} context from {path}")
            self.selected_z, self.selected_z_names = select_contexts(load_context(path), exp_config)
            if self.selected_z.shape[1] != self.z_dim:
                raise ValueError("Selected contexts do not match the model z_dim")
            self.z_index = 0
            logger.info(f"Selected {self.task_type} contexts: {self.selected_z_names}")
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def setup_policy(self, model_path):
        # load onnx policy
        import onnxruntime
        logger.info(f"Loading onnx policy from {model_path}")
        providers = self.policy_config.get("onnx_providers", ["CPUExecutionProvider"])
        if isinstance(providers, str):
            providers = [providers]
        if not isinstance(providers, list) or not all(isinstance(item, str) for item in providers):
            raise ValueError("policy_config['onnx_providers'] must be a string or list of strings")
        if len(providers) == 0:
            providers = ["CPUExecutionProvider"]
        logger.info(f"ONNX Runtime providers: {providers}")
        self.onnx_policy_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )
        inputs = self.onnx_policy_session.get_inputs()
        outputs = self.onnx_policy_session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Policy ONNX must have one input and one output")
        for tensor, width in ((inputs[0], self.policy_metadata['actor_obs_dim']),
                              (outputs[0], self.policy_metadata['output_action_dim'])):
            if tensor.type != 'tensor(float)' or len(tensor.shape) != 2 or tensor.shape[-1] != width:
                raise ValueError(f"ONNX tensor {tensor.name} differs from its metadata")
        self.onnx_input_name = inputs[0].name
        self.onnx_output_name = outputs[0].name

    def policy(self, obs):
        return self.onnx_policy_session.run(
            [self.onnx_output_name], {self.onnx_input_name: obs}
        )[0]

    def setup_observations(self):
        self.observations = {
            name: ObsGroup(name, config, self) for name, config in self.obs_cfg.items()
        }

    def reset(self):
        for group in self.observations.values():
            group.reset()

    def _warn_throttled(self, attr_name: str, message: str, interval_s: float = 1.0) -> None:
        now = time.monotonic()
        last = float(getattr(self, attr_name, 0.0))
        if now - last >= interval_s:
            logger.warning(message)
            setattr(self, attr_name, now)

    def _reset_tracking_state_to_stop(self, reset_realtime_z: bool = False) -> None:
        self.start_motion = False
        if hasattr(self, "t_stop"):
            self.t = self.t_stop
        if reset_realtime_z and getattr(self, "ctx_source", None) == "zmq":
            self.ctx_latest = _default_realtime_z(float(self.ctx_norm_ref))
            if self.ctx_window is not None:
                self.ctx_window.clear()

    def enter_safe_stop(
        self,
        reason: str,
        *,
        reset_realtime_z: bool = False,
        warning_attr: str = "_last_safe_stop_warning",
    ) -> None:
        self.policy_armed = False
        self.hold_init_pose = False
        self._reset_tracking_state_to_stop(reset_realtime_z=reset_realtime_z)
        self.last_action = np.zeros_like(self.last_action)
        self._warn_throttled(warning_attr, f"Safe stop: {reason}")

    def enter_stop_latch(self, reason: str) -> None:
        self.stop_latched = True
        self.stop_latch_enable_released_after_stop = False
        self.enter_safe_stop(reason)
        self.reset()

    def _request_exit(self, signum, _frame) -> None:
        self._exit_requested = True
        try:
            self._exit_signal_name = signal.Signals(signum).name
        except ValueError:
            self._exit_signal_name = str(signum)

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._request_exit)
            except (ValueError, RuntimeError):
                pass

    def _warn_stop_latch_ignored(self, source: str) -> None:
        self._warn_throttled(
            "_last_stop_latch_warning",
            f"{source} ignored while global stop latch is active; release enable/start inputs, then re-arm explicitly.",
        )

    def _clear_stop_latch_for_rearm(self, source: str) -> None:
        if not self.state_processor._prepare_low_state():
            self._warn_stop_latch_ignored(f"{source}: low state unavailable")
            return
        self.stop_latched = False
        self.stop_latch_enable_released_after_stop = True
        logger.info(f"{source}: global stop latch cleared")

    def _poll_realtime_z(self) -> bool:
        if getattr(self, "ctx_zmq", None) is None:
            return False

        import zmq

        received_valid_z = False
        while True:
            try:
                raw = self.ctx_zmq.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            try:
                z = np.frombuffer(raw, dtype=np.float32) if raw else np.array([], dtype=np.float32)
            except ValueError:
                self._warn_throttled(
                    "_last_invalid_z_warning",
                    "Ignoring invalid realtime z: malformed float32 payload",
                )
                continue
            if z.size != 256 or not np.all(np.isfinite(z)):
                self._warn_throttled(
                    "_last_invalid_z_warning",
                    f"Ignoring invalid realtime z: size={z.size}, finite={bool(np.all(np.isfinite(z)))}",
                )
                continue

            self.ctx_latest = z.copy()
            self.ctx_last_z_monotonic = time.monotonic()
            received_valid_z = True

        return received_valid_z

    def _check_realtime_z_watchdog(self) -> None:
        timeout_s = float(getattr(self, "ctx_zmq_timeout_s", 0.0))
        if timeout_s <= 0.0:
            return

        now = time.monotonic()
        last_z_time = getattr(self, "ctx_last_z_monotonic", None)
        reference_time = (
            float(last_z_time)
            if last_z_time is not None
            else float(getattr(self, "ctx_zmq_start_monotonic", now))
        )
        age_s = now - reference_time
        if age_s > timeout_s:
            self.enter_stop_latch(f"no valid realtime z for {age_s * 1000:.0f} ms")
            self._reset_tracking_state_to_stop(reset_realtime_z=True)

    def _apply_q_target_slew_limit(self, q_target: np.ndarray) -> np.ndarray:
        q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
        if self.last_cmd_q is None:
            baseline = np.asarray(self.state_processor.joint_pos, dtype=np.float64).reshape(-1)
            if baseline.shape != q_target.shape or not np.all(np.isfinite(baseline)):
                baseline = q_target.copy()
        else:
            baseline = np.asarray(self.last_cmd_q, dtype=np.float64).reshape(-1)

        max_delta = self.joint_velocity_limit * self.rl_dt * self.q_target_slew_safety_factor
        if max_delta.shape != q_target.shape:
            raise ValueError(
                f"q_target slew limit shape mismatch: {max_delta.shape} vs {q_target.shape}"
            )
        return baseline + np.clip(q_target - baseline, -max_delta, max_delta)

    def _send_safe_hold_command(self) -> bool:
        if not bool(getattr(self, "has_valid_low_state", False)):
            return False
        if not hasattr(self, "state_processor") or not hasattr(self.state_processor, "joint_pos"):
            return False

        hold_q = np.asarray(self.state_processor.joint_pos, dtype=np.float64).reshape(-1)
        if hold_q.shape[0] != self.num_dofs or not np.all(np.isfinite(hold_q)):
            return False

        hold_q = np.clip(hold_q, self.joint_pos_lower_limit, self.joint_pos_upper_limit)
        try:
            self.command_sender.send_command(
                hold_q,
                np.zeros(self.num_dofs),
                np.zeros(self.num_dofs),
            )
            self.last_cmd_q = hold_q.copy()
            return True
        except Exception as exc:
            self._warn_throttled(
                "_last_safe_stop_warning",
                f"Failed to send safe hold command: {exc}",
            )
            return False

    def prepare_obs_for_rl(self):
        obs_dict = {
            name: group.compute()[None, :].astype(np.float32)
            for name, group in self.observations.items()
        }
        # Training observes t-1..t-4 before inserting the current sensor frame.
        for group in self.observations.values():
            group.update()

        if self.task_type == "tracking":
            if getattr(self, "ctx_source", "pkl") == "pkl":
                window = self.ctx[self.t:self.t + self.window_size]
                norm_ref = float(np.linalg.norm(self.ctx[0]))
            else:
                self._poll_realtime_z()
                self._check_realtime_z_watchdog()
                self.ctx_window.appendleft(self.ctx_latest)
                window = np.stack(self.ctx_window)
                norm_ref = self.ctx_norm_ref

            discounts = self.gamma ** np.arange(len(window))
            discounts = discounts / np.sum(discounts)
            z = np.sum(window * discounts[:, None], axis=0)
            z = z / (np.linalg.norm(z) + 1e-8) * norm_ref

            if self.policy_armed:
                if self.start_motion and self.t < self.t_end:
                    self.t += 1
                    if getattr(self, "ctx_source", "pkl") == "pkl":
                        self.t %= len(self.ctx)
                    if self.t % 100 == 0:
                        logger.info(f"step={self.t}")
                else:
                    self.t = self.t_stop
                    self.start_motion = False
        else:
            z = self.selected_z[self.z_index]

        inputs = np.concatenate([obs_dict["policy"], z[None, :]], axis=-1).astype(np.float32)
        return obs_dict, inputs

    def get_init_target(self):
        if self.init_count > 500:
            self.init_count = 500

        # interpolate from current dof_pos to default angles
        dof_pos = self.state_processor.joint_pos
        progress = self.init_count / 500
        q_target = dof_pos + (self.default_dof_angles - dof_pos) * progress
        self.init_count += 1
        return q_target

    def run(self):
        self._install_signal_handlers()
        self._exit_requested = False
        self._exit_signal_name = None

        try:
            next_run_time = time.perf_counter()

            while not self._exit_requested:
                time.sleep(max(0.0, next_run_time - time.perf_counter()))
                if self._exit_requested:
                    break
                self._rl_step_scheduled()
                next_run_time += self.rl_dt
                now = time.perf_counter()
                if now > next_run_time:
                    if now - next_run_time >= self.rl_dt:
                        self.enter_stop_latch("control loop missed more than one period")
                        self._send_safe_hold_command()
                    # Expired commands must not be replayed against the same sensor frame.
                    next_run_time = now + self.rl_dt
            if self._exit_requested:
                logger.info(f"{self._exit_signal_name or 'signal'} received, exiting policy loop.")
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt, exiting policy loop.")
        finally:
            self.enter_safe_stop("policy loop exiting")
            self._send_safe_hold_command()

    def _rl_step_scheduled(self):
        loop_start = time.perf_counter()

        wireless_r2_stop = False
        if self.use_joystick:
            wireless_r2_stop = self.process_joystick_input()
        if self.pico_control_enabled and not wireless_r2_stop:
            self.process_pico_control_input()

        if not self.state_processor._prepare_low_state():
            self.enter_stop_latch("low state unavailable or stale")
            self._send_safe_hold_command()
            return
        self.has_valid_low_state = True

        try:
            # Prepare observations
            _, observations = self.prepare_obs_for_rl()

            # Inference
            action = np.asarray(self.policy(observations), dtype=np.float32)
            if action.shape != (1, self.num_actions) or not np.all(np.isfinite(action)):
                self.enter_stop_latch("invalid policy action shape or non-finite value")
                self._send_safe_hold_command()
                return
            # Clip policy action
            action = action.clip(-1, 1)
            action_scaled = self.action_rescale * action
        except Exception as e:
            print(f"Error in policy inference: {e}")
            self.enter_stop_latch(f"policy inference failed: {e}")
            self._send_safe_hold_command()
            return

        if self.stop_latched:
            q_target = self.state_processor.joint_pos
        elif self.hold_init_pose:
            q_target = self.get_init_target()
        elif not self.policy_armed:
            q_target = self.state_processor.joint_pos
        else:
            policy_action = np.zeros((self.num_dofs))
            policy_action[self.controlled_joint_indices] = action_scaled
            policy_action = policy_action * self.action_scale
            q_target = policy_action + self.default_dof_angles

        # Clip q target
        q_target = np.clip(
            q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
        )
        if not np.all(np.isfinite(q_target)):
            self.enter_stop_latch("non-finite q_target before slew limit")
            self._send_safe_hold_command()
            return

        q_target = self._apply_q_target_slew_limit(q_target)
        q_target = np.clip(
            q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
        )
        if not np.all(np.isfinite(q_target)):
            self.enter_stop_latch("non-finite q_target after slew limit")
            self._send_safe_hold_command()
            return

        # Send command
        cmd_q = q_target
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)
        self.last_cmd_q = np.asarray(cmd_q, dtype=np.float64).copy()
        if self.policy_armed and not self.hold_init_pose and not self.stop_latched:
            self.last_action = action_scaled
        else:
            self.last_action = np.zeros(self.num_actions)

        elapsed = time.perf_counter() - loop_start
        if elapsed > self.rl_dt:
            logger.warning(f"RL step took {elapsed:.6f} seconds, expected {self.rl_dt} seconds")


    def setup_pico_control(self):
        """Subscribe to Pico controller button JSON from the onboard teleop server."""
        try:
            import zmq
        except ImportError as exc:
            logger.warning(f"Pico control disabled because pyzmq is unavailable: {exc}")
            self.pico_control_enabled = False
            return

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, 0)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.pico_control_addr)
        self.pico_control_sock = sock
        logger.info(f"Pico policy control enabled: {self.pico_control_addr}")
        logger.info("Pico controls: A=init, A+B=enable policy/start tracking, B=stop, X=reset")

    def process_pico_control_input(self):
        if self.pico_control_sock is None:
            return

        try:
            import zmq
        except ImportError:
            return

        latest_buttons = None
        while True:
            try:
                raw = self.pico_control_sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                logger.debug(f"Pico control receive failed: {exc}")
                return

            try:
                payload = json.loads(raw)
            except Exception:
                continue
            buttons = payload.get("controller_buttons") if isinstance(payload, dict) else None
            if isinstance(buttons, dict):
                latest_buttons = {str(k): bool(v) for k, v in buttons.items()}

        if latest_buttons is None:
            return

        self.handle_pico_buttons(latest_buttons)

    def handle_pico_buttons(self, buttons: dict[str, bool]):
        prev = self.last_pico_buttons
        a = bool(buttons.get("right_key_one", False))
        b = bool(buttons.get("right_key_two", False))
        x = bool(buttons.get("left_key_one", False))

        prev_a = bool(prev.get("right_key_one", False))
        prev_b = bool(prev.get("right_key_two", False))
        prev_x = bool(prev.get("left_key_one", False))

        combo = a and b
        prev_combo = prev_a and prev_b
        if self.stop_latched and not a and not b:
            self.stop_latch_enable_released_after_stop = True

        if combo and not prev_combo:
            if self.stop_latched:
                if not self.stop_latch_enable_released_after_stop:
                    self._warn_stop_latch_ignored("Pico A+B")
                    self.last_pico_buttons = buttons
                    return
                self._clear_stop_latch_for_rearm("Pico A+B")
            logger.info("Pico A+B: enable policy and start tracking")
            self.handle_control_button("R1")
            self.handle_control_button("B")
        elif a and not prev_a and not b:
            logger.info("Pico A: init")
            self.handle_control_button("A")
        elif b and not prev_b and not a:
            logger.info("Pico B: stop policy")
            self.handle_control_button("R2")
        elif x and not prev_x:
            logger.info("Pico X: reset")
            self.handle_control_button("X")

        self.last_pico_buttons = buttons


    def process_joystick_input(self):
        """Poll current wireless controller state and translate to high-level key events."""
        try:
            self.wc_msg = self.robot.read_wireless_controller()
        except Exception:
            return False

        if self.wc_msg is None:
            return False

        r2_stop_pressed = bool(self.wc_msg.R2)
        if r2_stop_pressed:
            self.enter_stop_latch("G1 wireless R2 stop")
            self.last_wc_msg = self.wc_msg
            return True

        if self.stop_latched:
            enable_or_start_pressed = bool(self.wc_msg.R1 or self.wc_msg.B)
            if not enable_or_start_pressed:
                self.stop_latch_enable_released_after_stop = True

            combo = bool(self.wc_msg.R1 and self.wc_msg.B)
            prev_combo = bool(self.last_wc_msg.R1 and self.last_wc_msg.B)
            if combo and not prev_combo:
                if self.stop_latch_enable_released_after_stop:
                    self._clear_stop_latch_for_rearm("Wireless R1+B")
                    self.handle_control_button("R1")
                    self.handle_control_button("B")
                else:
                    self._warn_stop_latch_ignored("Wireless R1+B")
                self.last_wc_msg = self.wc_msg
                return False

            if (
                (self.wc_msg.R1 and not self.last_wc_msg.R1)
                or (self.wc_msg.B and not self.last_wc_msg.B)
            ):
                self._warn_stop_latch_ignored("Wireless enable/start")
                self.last_wc_msg = self.wc_msg
                return False

        for button in ("A", "B", "X", "Y", "R1"):
            if getattr(self.wc_msg, button) and not getattr(self.last_wc_msg, button):
                self.handle_control_button(button)

        self.last_wc_msg = self.wc_msg
        return False


    def handle_control_button(self, cur_key):
        if cur_key == "R1":
            if self.stop_latched:
                self._warn_stop_latch_ignored("R1 enable")
                return
            if not self.policy_armed:
                self.last_action = np.zeros(self.num_actions)
                self.reset()
            if self.task_type == "reward":
                logger.info("Reward ready: holding initial pose; press B to start")
            else:
                logger.info("Using policy actions")
            self.policy_armed = True
            self.hold_init_pose = self.task_type == "reward"
            if self.task_type in {"reward", "goal"}:
                self._log_selected_context()
            if self.task_type == "tracking":
                self.t = self.t_stop

        elif cur_key == "R2":
            self.enter_stop_latch("R2 stop command")
            logger.info("Policy stopped; holding current joint positions")
        elif cur_key == "A":
            if self.stop_latched:
                self._clear_stop_latch_for_rearm("A initialize")
                if self.stop_latched:
                    return
            if self.task_type == "reward":
                self.policy_armed = False
            self.hold_init_pose = True
            self.init_count = 0
            logger.info("Moving to initialization pose")
        elif cur_key == "B":
            if self.stop_latched:
                self._warn_stop_latch_ignored("B start")
                return
            if self.task_type == "tracking":
                logger.info("Starting motion")
                self.start_motion = True
                self.t = self.t_start
            elif self.task_type == "reward":
                if not self.policy_armed:
                    logger.warning("Press R1 before B to start the reward policy")
                    return
                self.hold_init_pose = False
                logger.info("Starting reward policy")
            else:
                logger.info(f"B / keyboard [ is unused in {self.task_type} mode")
        elif cur_key == "X":
            self.z_index = 0
            self.start_motion = False
            if self.task_type == "tracking":
                self.t = self.t_stop
            logger.info("Resetting to stop state")
        elif cur_key == "Y" and self.task_type in {"reward", "goal"}:
            self.z_index = (self.z_index + 1) % len(self.selected_z)
            self._log_selected_context()

    def _log_selected_context(self):
        name = self.selected_z_names[self.z_index]
        logger.info(f"Selected {self.task_type} context ({self.z_index + 1}/{len(self.selected_z)}): {name}")

    # ----------------------------- Keyboard handling -----------------------------
    def _warn_unsupported_keyboard_key(self, keycode: str) -> None:
        logger.warning(f"Keyboard key '{keycode}' is not supported by this release runtime")

    def _set_debug_kp_level(self, *, delta: float | None = None, value: float | None = None) -> None:
        if value is None:
            value = self.command_sender.kp_level + float(delta or 0.0)
        value = max(0.0, float(value))
        self.command_sender.kp_level = value
        logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
        logger.info(
            colored(
                f"Debug kp: {np.array2string(self.command_sender.joint_kp_unitree, precision=4)}",
                "green",
            )
        )

    def start_key_listener(self):
        """Listen for keyboard commands until the process exits."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except Exception as e:
                logger.warning(f"Keyboard key {keycode}. Error: {e}")

        listen_keyboard(on_press=on_press)

    def handle_keyboard_button(self, keycode):
        command = {"]": "R1", "[": "B", "n": "Y", "p": "X", "i": "A"}.get(keycode)
        if command is not None:
            self.handle_control_button(command)
        elif keycode == "o":
            self.enter_safe_stop("keyboard stop")
        elif keycode == "r":
            self._clear_stop_latch_for_rearm("Keyboard r")
        elif keycode in {"w", "s", "a", "d", "q", "e", "z"}:
            self._warn_unsupported_keyboard_key(keycode)
        elif keycode == "5":
            self._set_debug_kp_level(delta=-0.01)
        elif keycode == "6":
            self._set_debug_kp_level(delta=0.01)
        elif keycode == "4":
            self._set_debug_kp_level(delta=-0.1)
        elif keycode == "7":
            self._set_debug_kp_level(delta=0.1)
        elif keycode == "0":
            self._set_debug_kp_level(value=1.0)
        else:
            self._warn_unsupported_keyboard_key(keycode)
