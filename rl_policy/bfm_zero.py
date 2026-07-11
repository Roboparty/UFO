from __future__ import annotations

import sys
import time
import numpy as np
from typing import Dict, Any, Type
import sched
from termcolor import colored
from sshkeyboard import listen_keyboard
from pathlib import Path
import copy
import joblib
import json
import pickle


sys.path.append(".")


from loguru import logger

# Re-use utilities from the existing sim2real package
from utils.strings import resolve_matching_names_values
from utils.onnx_module import Timer
from rl_policy.observations import Observation, ObsGroup
from rl_policy.utils.state_processor import StateProcessor
from rl_policy.utils.command_sender import CommandSender


def _default_realtime_z(norm_ref: float) -> np.ndarray:
    z = np.zeros(256, dtype=np.float32)
    z[0] = float(norm_ref)
    return z


def _resolve_model_relative_context_path(
    model_path: str,
    exp_config: Dict[str, Any],
    default_ctx_dir: str,
    fallback_ctx_dirs: tuple[str, ...] = (),
) -> Path:
    ctx_path = Path(str(exp_config["ctx_path"])).expanduser()
    if ctx_path.is_absolute():
        return ctx_path

    model_export_dir = Path(model_path).expanduser().resolve(strict=False).parent
    model_root = model_export_dir.parent
    candidates: list[Path] = []

    if "ctx_dir" in exp_config:
        candidates.append(model_root / str(exp_config["ctx_dir"]) / ctx_path)
    else:
        candidates.append(model_root / default_ctx_dir / ctx_path)
        for ctx_dir in fallback_ctx_dirs:
            candidates.append(model_root / ctx_dir / ctx_path)

    # Backward compatibility: older configs used paths relative to either the
    # model root or the exported/ directory.
    candidates.append(model_root / ctx_path)
    candidates.append(model_export_dir / ctx_path)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduped.append(resolved)

    for candidate in deduped:
        if candidate.is_file():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in deduped)
    raise FileNotFoundError(
        f"Could not resolve context file ctx_path={ctx_path!s} for model_path={model_path!s}.\n"
        f"Tried:\n{tried}"
    )

# -------------------------------------------------------------------------------------------------
# High-level RL policy that plugs into the existing framework
# -------------------------------------------------------------------------------------------------
class BFMZeroPolicy:
    def __init__(
        self,
        robot_config: Dict[str, Any],
        policy_config: Dict[str, Any],
        exp_config: Dict[str, Any],
        model_path: str,
        rl_rate: int = 50,
        pico_control: bool = False,
        pico_control_addr: str = "tcp://127.0.0.1:28704",
    ) -> None:
        robot_type = robot_config["ROBOT_TYPE"]
        if robot_type == "g1_real":
            # example: sys.path.append("/home/unitree/User/unitree_sdk2/build/lib")
            sys.path.append("/home/unitree/unitree_sdk2_bfm/build/lib")
            import g1_interface
            network_interface = robot_config.get("INTERFACE", None)
            self.robot = g1_interface.G1Interface(network_interface)
            try:
                self.robot.set_control_mode(g1_interface.ControlMode.PR)
            except Exception:
                pass  # Ignore if firmware already in the correct mode
            robot_config["robot"] = self.robot
        # Plug-in our custom state processor & command sender
        self.state_processor = StateProcessor(robot_config, policy_config["isaac_joint_names"])
        self.command_sender = CommandSender(robot_config, policy_config)

        self.rl_dt = 1.0 / rl_rate
        self.t = 0

        self.policy_config = policy_config

        self.setup_policy(model_path)
        self.obs_cfg = policy_config["observation"]

        self.isaac_joint_names = policy_config["isaac_joint_names"]
        self.num_dofs = len(self.isaac_joint_names)

        default_joint_pos_dict = policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            self.isaac_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_dof_angles = np.zeros(len(self.isaac_joint_names))
        self.last_action  = np.zeros(len(self.isaac_joint_names))
        self.default_dof_angles[joint_indices] = default_joint_pos

        action_scale_cfg = policy_config["action_scale"]
        self.action_scale = np.ones((self.num_dofs))
        self.action_rescale = policy_config["action_rescale"]
        if isinstance(action_scale_cfg, float):
            self.action_scale *= action_scale_cfg
        elif isinstance(action_scale_cfg, dict):
            joint_ids, joint_names, action_scales = resolve_matching_names_values(
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
        self.use_policy_action = False
        self.pico_control_enabled = bool(pico_control)
        self.pico_control_addr = str(pico_control_addr)
        self.pico_control_sock = None
        self.last_pico_buttons: Dict[str, bool] = {}

        self.first_time_init = True
        self.init_count = 0
        self.get_ready_state = False

        # Joint limits
        joint_indices, joint_names, joint_pos_lower_limit = (
            resolve_matching_names_values(
                robot_config["joint_pos_lower_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_lower_limit = np.zeros(self.num_dofs)
        self.joint_pos_lower_limit[joint_indices] = joint_pos_lower_limit

        joint_indices, joint_names, joint_pos_upper_limit = (
            resolve_matching_names_values(
                robot_config["joint_pos_upper_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_upper_limit = np.zeros(self.num_dofs)
        self.joint_pos_upper_limit[joint_indices] = joint_pos_upper_limit

        # ------------------------------------------------------
        # Joystick / keyboard setup (mirrors base_policy logic)
        # ------------------------------------------------------
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

        # Setup observations after all processors are ready
        self.setup_observations()

        # Initialize variables
        self.exp_config = exp_config
        self.task_type = exp_config['type']
        self.start_motion = False
        logger.info(f"task_type={self.task_type}")

        # Task-specific setup
        if self.task_type == "tracking":
            self.ctx_source = str(exp_config.get("ctx_source", "pkl")).lower()
            if self.ctx_source not in {"pkl", "zmq"}:
                raise ValueError(f"Invalid ctx_source={self.ctx_source}, expected 'pkl' or 'zmq'")

            if self.ctx_source == "pkl":
                ctx_path = _resolve_model_relative_context_path(
                    model_path,
                    exp_config,
                    default_ctx_dir="tracking_inference_mjlab",
                    fallback_ctx_dirs=("tracking_inference",),
                )
                logger.info(f"Loading tracking context from {ctx_path}")
                self.ctx = joblib.load(ctx_path)
                self.ctx_zmq = None
                self.ctx_window = None
                self.ctx_norm_ref = float(np.linalg.norm(self.ctx[0]))
            else:
                import zmq
                from collections import deque

                self.ctx = None
                self.ctx_norm_ref = float(exp_config.get("ctx_norm_ref", 16.0))
                self.ctx_window = deque(maxlen=int(exp_config.get("window_size", 1)))

                zmq_addr = str(exp_config.get("ctx_zmq_addr", "tcp://127.0.0.1:28711"))
                logger.info(f"Tracking context from ZMQ: {zmq_addr} (norm_ref={self.ctx_norm_ref})")
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
            self.t_start = exp_config['start']
            self.t_end = exp_config['end']
            self.t_stop = exp_config['stop']

            logger.info(f"gamma={exp_config['gamma']}, window_size={exp_config['window_size']}")
            self.gamma = exp_config['gamma']  # discount factor
            self.window_size = exp_config['window_size']  # context window size

        elif self.task_type == "reward":
            self.z_index = 0
            reward_path = _resolve_model_relative_context_path(
                model_path,
                exp_config,
                default_ctx_dir="reward_inference_mjlab",
                fallback_ctx_dirs=("reward_inference",),
            )
            logger.info(f"Loading reward context from {reward_path}")
            with open(reward_path, "rb") as f:
                self.z_dict = pickle.load(f)
                self.z_dict_raw = copy.deepcopy(self.z_dict)
                logger.info(colored(f"\n\nAvailable z_dict={list(self.z_dict_raw.keys())}", "green"))
            
            if "selected_rewards_filter_z" in exp_config:
                selected_rewards_filter_z = exp_config['selected_rewards_filter_z'] 
                logger.info(colored(f"\nSelected_rewards_filter_z read from config: {selected_rewards_filter_z}", "green"))
            else:
                raise ValueError("For task_type=reward-multiple-z-selection, selected_rewards_filter_z must be provided in the config file")
            
            self.z_dict = {}
            self.selected_z_names = []
            
            # Iterate in the order of selected_rewards_filter_z
            if isinstance(selected_rewards_filter_z, list):
                for dct in selected_rewards_filter_z:
                    k = dct['reward']
                    selected_z_ids = dct['z_ids']
                    if k in self.z_dict_raw:
                        v = self.z_dict_raw[k]
                        self.z_dict[k] = []
                        for z_id in selected_z_ids:
                            if z_id < len(v):
                                self.z_dict[k].append(v[z_id])
                                self.selected_z_names.append(f"""Reward="{k}"__Z_id={z_id}""")
                                logger.info(f"""Added Reward="{k}"__Z_id={z_id} to self.z_dict""")
            elif isinstance(selected_rewards_filter_z, dict):
                for k in selected_rewards_filter_z.keys():
                    if k in self.z_dict_raw:
                        v = self.z_dict_raw[k]
                        self.z_dict[k] = []
                        for z_id in selected_rewards_filter_z[k]:
                            if z_id < len(v):
                                self.z_dict[k].append(v[z_id])
                                self.selected_z_names.append(f"""Reward="{k}"__Z_id={z_id}""")
                                logger.info(f"""Added Reward="{k}"__Z_id={z_id} to self.z_dict""")
            
            logger.info(colored(f"\n\nValid z_dict contains: {list(self.z_dict.keys())}", "blue"))
            if len(self.z_dict) == 0:
                raise ValueError("After filtering, self.z_dict is empty. Please check your selected_rewards_filter_z and available z_dict")
            self.num_selected_rewards = len(self.z_dict.keys())
            self.num_selected_z = len(self.selected_z_names)
            self.selected_z = np.concatenate([val for val in self.z_dict.values()], axis=0)
            
            logger.info(f"self.num_selected_z={self.num_selected_z}, self.selected_z.shape={self.selected_z.shape}")
            if self.num_selected_rewards == 1:
                logger.info(colored(f"Only one reward is selected, make sure that is what you want", "red"))

        # new goal reaching code
        elif self.task_type == "goal":
            self.z_index = 0
            goal_path = _resolve_model_relative_context_path(
                model_path,
                exp_config,
                default_ctx_dir="goal_inference_mjlab",
                fallback_ctx_dirs=("goal_inference",),
            )
            logger.info(f"Loading goal context from {goal_path}")
            with open(goal_path, "rb") as f:
                self.z_dict = joblib.load(f)
                self.z_dict_raw = copy.deepcopy(self.z_dict)
                logger.info(colored(f"\n\nAvailable z_dict={list(self.z_dict_raw.keys())}", "green"))
            if "selected_goals" in exp_config:
               selected_goals = exp_config['selected_goals'] 
               logger.info(colored(f"\nSelected_goals read from config: {selected_goals}", "green"))
            else:
                selected_goals = self.z_dict.keys()
            self.z_dict = {
                k: self.z_dict[k] for k in selected_goals if k in self.z_dict
            }

            logger.info(colored(f"\n\nValid z_dict contains: {list(self.z_dict.keys())} (Total = {len(self.z_dict)})", "blue"))

            self.num_selected_goals = len(self.z_dict.keys())
            if self.num_selected_goals ==1:
                logger.info(colored(f"Only one goal is selected, make sure that is what you want", "red"))


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
        self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name

        def policy_act(obs):
            return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        self.policy = policy_act

    def setup_observations(self):
        """Setup observations for policy inference"""
        self.observations: Dict[str, ObsGroup] = {}
        self.reset_callbacks = []
        self.update_callbacks = []
        
        # Create observation instances based on config
        for obs_group, obs_items in self.obs_cfg.items():
            print(f"obs_group: {obs_group}")
            obs_funcs = {}
            for obs_name, obs_config in obs_items.items():
                obs_class: Type[Observation] = Observation.registry[obs_name]
                obs_func = obs_class(env=self, **obs_config)
                obs_funcs[obs_name] = obs_func
                self.reset_callbacks.append(obs_func.reset)
                self.update_callbacks.append(obs_func.update)
                print(f"\t{obs_name}: {obs_config}")
            self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)

    def reset(self):
        for reset_callback in self.reset_callbacks:
            reset_callback()

    def update(self):
        self.state_dict["action"] = self.last_action
        for update_callback in self.update_callbacks:
            update_callback(self.state_dict)

    def prepare_obs_for_rl(self):
        """Prepare observation for policy inference using observation classes"""
        obs_dict: Dict[str, np.ndarray] = {}
        self.update()
        
        for obs_group in self.observations.values():
            obs = obs_group.compute()
            obs_dict[obs_group.name] = obs[None, :].astype(np.float32)
        
        obs = obs_dict[obs_group.name]

        if self.task_type == "tracking":
            if getattr(self, "ctx_source", "pkl") == "pkl":
                window = self.ctx[self.t:self.t + self.window_size]
                discounts = self.gamma ** np.arange(len(window))
                discounts = discounts / np.sum(discounts)
                discounted_avg = np.sum(window * discounts[:, np.newaxis], axis=0)
                discounted_avg = (
                    discounted_avg
                    / (np.linalg.norm(discounted_avg, axis=-1) + 1e-8)
                    * float(np.linalg.norm(self.ctx[0]))
                )
            else:
                import zmq
                from collections import deque

                while True:
                    try:
                        raw = self.ctx_zmq.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if raw:
                        z = np.frombuffer(raw, dtype=np.float32)
                        if z.size == 256:
                            self.ctx_latest = z.copy()

                if self.ctx_window.maxlen != int(self.window_size):
                    old = list(self.ctx_window)
                    self.ctx_window = deque(old[-int(self.window_size):], maxlen=int(self.window_size))

                self.ctx_window.appendleft(self.ctx_latest)
                if len(self.ctx_window) > 0:
                    window = np.stack(list(self.ctx_window), axis=0)
                else:
                    window = self.ctx_latest[None, :]
                discounts = self.gamma ** np.arange(window.shape[0])
                discounts = discounts / np.sum(discounts)
                discounted_avg = np.sum(window * discounts[:, np.newaxis], axis=0)
                discounted_avg = (
                    discounted_avg
                    / (np.linalg.norm(discounted_avg) + 1e-8)
                    * float(self.ctx_norm_ref)
                )

            inputs = np.concatenate([obs, discounted_avg[np.newaxis, :]], axis=-1).astype(np.float32)
            
            if self.use_policy_action:
                if self.start_motion and self.t < self.t_end:
                    self.t += 1
                    if getattr(self, "ctx_source", "pkl") == "pkl":
                        self.t = self.t % self.ctx.shape[0]
                    if self.t % 100 == 0:
                        logger.info(f"step={self.t}")
                else:
                    self.t = self.t_stop
                    self.start_motion = False
        elif self.task_type == "reward":
            try:
                inputs = np.concatenate([obs, self.selected_z[self.z_index]], axis=-1).astype(np.float32)
            except Exception as e:
                print(f"obs={obs.shape}")
                print(f"self.selected_z[self.z_index]={self.selected_z[self.z_index].shape}")
                raise e
        # new goal reaching code
        elif self.task_type == "goal":
            try:
                inputs = np.concatenate([obs, list(self.z_dict.values())[self.z_index]], axis=-1).astype(np.float32)
            except Exception as e:
                print(f"obs={obs.shape}")
                print(f"list(self.z_dict.values())[self.z_index]]={list(self.z_dict.values())[self.z_index].shape}")
                raise e

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
        total_inference_cnt = 0
        state_dict = {}
        state_dict["action"] = np.zeros(self.num_actions)
        self.state_dict = state_dict
        self.total_inference_cnt = total_inference_cnt
        self.perf_dict = {}

        try:
            scheduler = sched.scheduler(time.perf_counter, time.sleep)
            next_run_time = time.perf_counter()
            
            while True:
                scheduler.enterabs(next_run_time, 1, self._rl_step_scheduled, ())
                scheduler.run()
                
                next_run_time += self.rl_dt
                self.total_inference_cnt += 1

                if self.total_inference_cnt % 100 == 0:
                    self.perf_dict = {}
        except KeyboardInterrupt:
            pass

    def _rl_step_scheduled(self):
        loop_start = time.perf_counter()

        with Timer(self.perf_dict, "prepare_low_state"):
            if self.use_joystick:
                # print(f"Debug::process_joystick:")
                self.process_joystick_input()
            if self.pico_control_enabled:
                self.process_pico_control_input()

            if not self.state_processor._prepare_low_state():
                print("low state not ready.")
                return
            
        try:
            with Timer(self.perf_dict, "prepare_obs"):
                # Prepare observations
                obs_dict, observations = self.prepare_obs_for_rl()
                self.state_dict["is_init"] = np.zeros(1, dtype=bool)

            with Timer(self.perf_dict, "policy"):  
                # Inference
                action = self.policy(observations)
                # Clip policy action
                action = action.clip(-1, 1)
                action_scaled = self.action_rescale * action
                self.last_action = action_scaled
        except Exception as e:
            print(f"Error in policy inference: {e}")
            self.state_dict["action"] = np.zeros(self.num_actions)
            return

        with Timer(self.perf_dict, "rule_based_control_flow"):
            # rule based control flow
            if self.get_ready_state:
                q_target = self.get_init_target()
            elif not self.use_policy_action:
                q_target = self.state_processor.joint_pos
            else:
                policy_action = np.zeros((self.num_dofs))
                policy_action[self.controlled_joint_indices] = action_scaled
                policy_action = policy_action * self.action_scale
                q_target = policy_action + self.default_dof_angles

            # Clip q target
            # print(self.joint_pos_lower_limit)
            q_target = np.clip(
                q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
            )

            # Send command
            cmd_q = q_target
            cmd_dq = np.zeros(self.num_dofs)
            cmd_tau = np.zeros(self.num_dofs)
            self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)

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

    def handle_pico_buttons(self, buttons: Dict[str, bool]):
        prev = self.last_pico_buttons
        a = bool(buttons.get("right_key_one", False))
        b = bool(buttons.get("right_key_two", False))
        x = bool(buttons.get("left_key_one", False))

        prev_a = bool(prev.get("right_key_one", False))
        prev_b = bool(prev.get("right_key_two", False))
        prev_x = bool(prev.get("left_key_one", False))

        combo = a and b
        prev_combo = prev_a and prev_b
        if combo and not prev_combo:
            logger.info("Pico A+B: enable policy and start tracking")
            self.handle_joystick_button("R1")
            self.handle_joystick_button("B")
        elif a and not prev_a and not b:
            logger.info("Pico A: init")
            self.handle_joystick_button("A")
        elif b and not prev_b and not a:
            logger.info("Pico B: stop policy")
            self.handle_joystick_button("R2")
        elif x and not prev_x:
            logger.info("Pico X: reset")
            self.handle_joystick_button("X")

        self.last_pico_buttons = buttons


    def process_joystick_input(self):
        """Poll current wireless controller state and translate to high-level key events."""
        try:
            self.wc_msg = self.robot.read_wireless_controller()
        except Exception:
            return

        if self.wc_msg is None:
            return

        # print(f"wc_msg.A: {self.wc_msg.A}")
        if self.wc_msg.A and not self.last_wc_msg.A:
            self.handle_joystick_button("A")
        if self.wc_msg.B and not self.last_wc_msg.B:
            self.handle_joystick_button("B")
        if self.wc_msg.X and not self.last_wc_msg.X:
            self.handle_joystick_button("X")
        if self.wc_msg.Y and not self.last_wc_msg.Y:
            self.handle_joystick_button("Y")
        if self.wc_msg.L1 and not self.last_wc_msg.L1:
            self.handle_joystick_button("L1")
        if self.wc_msg.L2 and not self.last_wc_msg.L2:
            self.handle_joystick_button("L2")
        if self.wc_msg.R1 and not self.last_wc_msg.R1:
            self.handle_joystick_button("R1")
        if self.wc_msg.R2 and not self.last_wc_msg.R2:
            self.handle_joystick_button("R2")
        
        self.last_wc_msg = self.wc_msg


    def handle_joystick_button(self, cur_key):
        if cur_key == "R1":
            logger.info("Using policy actions")
            self.use_policy_action = True
            self.get_ready_state = False    
            if self.task_type == "reward":
                total_z = self.num_selected_z
                logger.info(colored(f"Switch to reward={self.selected_z_names[self.z_index]} (Count: {self.z_index+1}/{total_z})", "blue"))
            if self.task_type == "goal":
                logger.info(colored(f"Switch to goal={list(self.z_dict.keys())[self.z_index]} (Count: {self.z_index+1}/{len(self.z_dict)})", "blue"))
            if self.task_type == "tracking":
                self.t = self.t_stop

        elif cur_key == "R2":
            self.use_policy_action = False
            self.get_ready_state = False
            logger.info(colored("Actions set to zero", "blue"))
        elif cur_key == "A":
            self.get_ready_state = True
            self.init_count = 0
            logger.info(colored("Setting to init state (do this when robot was in a bad shape)", "blue"))
        elif cur_key == "B":
            if self.task_type == "tracking":
                logger.info("Starting motion")
                self.start_motion = True
                self.t = self.t_start
            else:
                logger.info(colored(f"Commmand [ is undefined in current task type {self.task_type}!", "red"))
                pass
        elif cur_key == "X":
            self.z_index = 0
            self.start_motion = False
            if self.task_type == "tracking":
                self.t = self.t_stop
            logger.info("Resetting to stop state")
        elif cur_key == "Y":
            if self.task_type == "reward":
                if self.z_index >= self.num_selected_z - 1:
                    self.z_index = 0
                else:
                    self.z_index += 1
                total_z = self.num_selected_z if self.task_type.startswith("reward-multiple-z-selection-duplicate") else self.num_selected_rewards
                logger.info(colored(f"Switch to reward={self.selected_z_names[self.z_index]} (Count: {self.z_index+1}/{total_z})", "blue"))
            elif self.task_type == "goal":
                if self.z_index >= self.num_selected_goals - 1:
                    self.z_index = 0
                else:
                    self.z_index += 1
                logger.info(colored(f"Switch to goal {list(self.z_dict.keys())[self.z_index]} ({self.z_index+1}/{self.num_selected_goals})", "blue"))

        # Debug print for kp level tuning
        if cur_key in ["Y+left", "Y+right", "A+left", "A+right"]:
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))

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
        """Start a key listener using sshkeyboard (same as BasePolicy)."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except Exception as e:
                logger.warning(f"Keyboard key {keycode}. Error: {e}")

        listener = listen_keyboard(on_press=on_press)
        listener.start()
        listener.join()

    def handle_keyboard_button(self, keycode):
        if keycode == "]":
            logger.info("Using policy actions")
            self.use_policy_action = True
            self.get_ready_state = False    
            if self.task_type == "reward":
                total_z = self.num_selected_z
                logger.info(colored(f"Switch to reward={self.selected_z_names[self.z_index]} (Count: {self.z_index+1}/{total_z})", "blue"))
            if self.task_type == "goal":
                logger.info(colored(f"Switch to goal={list(self.z_dict.keys())[self.z_index]} (Count: {self.z_index+1}/{len(self.z_dict)})", "blue"))
            if self.task_type == "tracking":
                self.t = self.t_stop
        elif keycode == "[":
            if self.task_type == "tracking":
                logger.info("Starting motion")
                self.start_motion = True
                self.t = self.t_start
            else:
                logger.info(colored(f"Commmand [ is undefined in current task type {self.task_type}!", "red"))
                pass
        elif keycode == "n":
            if self.task_type == "reward":
                if self.z_index >= self.num_selected_z - 1:
                    self.z_index = 0
                else:
                    self.z_index += 1
                total_z = self.num_selected_z if self.task_type.startswith("reward-multiple-z-selection-duplicate") else self.num_selected_rewards
                logger.info(colored(f"Switch to reward={self.selected_z_names[self.z_index]} (Count: {self.z_index+1}/{total_z})", "blue"))
            elif self.task_type == "goal":
                if self.z_index >= self.num_selected_goals - 1:
                    self.z_index = 0
                else:
                    self.z_index += 1
                logger.info(colored(f"Switch to goal {list(self.z_dict.keys())[self.z_index]} ({self.z_index+1}/{self.num_selected_goals})", "blue"))
        elif keycode == "p":
            self.z_index = 0
            self.start_motion = False
            if self.task_type == "tracking":
                self.t = self.t_stop
            logger.info("Resetting to stop state")
        elif keycode == "o":
            self.use_policy_action = False
            self.get_ready_state = False
            logger.info("Actions set to zero")
        elif keycode == "i":
            self.get_ready_state = True
            self.init_count = 0
            logger.info("Setting to init state")
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
        

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--policy_config", type=str, help="policy config file"
    )
    parser.add_argument(
        "--model_path", type=str, help="model path"
    )
    parser.add_argument(
        "--task", type=str, help="task type: tracking or reward or single or stiching", default="track-walk"
    )
    parser.add_argument(
        "--pico-control", action="store_true",
        help="Enable Pico controller buttons from the onboard teleop server for policy mode control.",
    )
    parser.add_argument(
        "--pico-control-addr", type=str, default="tcp://127.0.0.1:28704",
        help="ZMQ SUB address for Pico controller button JSON.",
    )

    args = parser.parse_args()

    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
    
    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)

    with open(args.task, 'r') as file:
        exp_config = yaml.load(file, Loader=yaml.FullLoader)
    model_path = args.model_path

    policy = BFMZeroPolicy(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=model_path,
        exp_config=exp_config,
        rl_rate=50,
        pico_control=bool(args.pico_control),
        pico_control_addr=str(args.pico_control_addr),
    )
    policy.run()
