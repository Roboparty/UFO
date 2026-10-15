import time
import numpy as np
import zmq

from utils.strings import joint_values, unitree_joint_names
from utils.common import LowCmdMessage, PORTS


class CommandSender:
    def __init__(self, robot_config, policy_config):
        self.robot_type = robot_config["ROBOT_TYPE"]
        if self.robot_type == "g1_real":
            self.robot = robot_config["robot"]
        else:
            supported_types = {"g1_29dof"}
            if self.robot_type not in supported_types:
                raise NotImplementedError(
                    f"Robot type {self.robot_type} is not supported by the UFO-Deploy release runtime"
                )

        self._kp_level = 1.0
        self.policy_config = policy_config
        self.joint_kp_unitree_default = joint_values(policy_config["joint_kp"], unitree_joint_names)
        self.joint_kp_unitree = self.joint_kp_unitree_default.copy()
        self.joint_kd_unitree = joint_values(policy_config["joint_kd"], unitree_joint_names)
        self.default_joint_pos_unitree = joint_values(policy_config["default_joint_pos"], unitree_joint_names)

        joint_names_isaac = self.policy_config["isaac_joint_names"]
        self.joint_indices_unitree = [unitree_joint_names.index(name) for name in joint_names_isaac]

        # init low cmd publisher
        if self.robot_type != "g1_real":
            self.zmq_context = zmq.Context.instance()
            self.low_cmd_port = robot_config.get(
                "LOW_CMD_PORT", PORTS["low_cmd"]
            )
            bind_addr = robot_config.get("LOW_CMD_BIND_ADDR", "*")
            bind_endpoint = f"tcp://{bind_addr}:{self.low_cmd_port}"

            self.lowcmd_socket: zmq.Socket = self.zmq_context.socket(zmq.PUB)
            self.lowcmd_socket.setsockopt(zmq.SNDHWM, 1)
            self.lowcmd_socket.setsockopt(zmq.LINGER, 0)
            self.lowcmd_socket.bind(bind_endpoint)
            # Give subscribers time to connect before sending commands
            time.sleep(0.1)
        else:
            self.lowcmd_socket = None

        self.InitLowCmd()

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = value
        self.joint_kp_unitree[:] = self.joint_kp_unitree_default * self._kp_level

    def InitLowCmd(self):
        self.cmd_q = np.zeros(len(unitree_joint_names))
        self.cmd_dq = np.zeros(len(unitree_joint_names))
        self.cmd_tau = np.zeros(len(unitree_joint_names))

        self.cmd_q[:] = self.default_joint_pos_unitree

    def send_command(self, cmd_q, cmd_dq, cmd_tau):
        if self.robot_type != "g1_real":
            self.cmd_q[self.joint_indices_unitree] = cmd_q
            self.cmd_dq[self.joint_indices_unitree] = cmd_dq
            self.cmd_tau[self.joint_indices_unitree] = cmd_tau
            
            message = LowCmdMessage(
                q_target=self.cmd_q,
                dq_target=self.cmd_dq,
                tau_ff=self.cmd_tau,
                kp=self.joint_kp_unitree,
                kd=self.joint_kd_unitree,
            )
            try:
                self.lowcmd_socket.send(message.to_bytes(), flags=zmq.DONTWAIT)
            except zmq.Again:
                pass
        else:
            cmd = self.robot.create_zero_command()

            # kp_level setter maintains joint_kp_unitree; kd remains constant.
            kp_scaled = self.joint_kp_unitree
            kd_scaled = self.joint_kd_unitree

            q_target = list(cmd.q_target)
            dq_target = list(cmd.dq_target)
            tau_ff = list(cmd.tau_ff)
            kp = list(cmd.kp)
            kd = list(cmd.kd)
            for i_policy, idx_unitree in enumerate(self.joint_indices_unitree):
                q_target[idx_unitree] = float(cmd_q[i_policy])
                dq_target[idx_unitree] = float(cmd_dq[i_policy])
                tau_ff[idx_unitree] = float(cmd_tau[i_policy])
                kp[idx_unitree] = float(kp_scaled[idx_unitree])
                kd[idx_unitree] = float(kd_scaled[idx_unitree])

            cmd.q_target = q_target
            cmd.dq_target = dq_target
            cmd.tau_ff = tau_ff
            cmd.kp = kp
            cmd.kd = kd

            self.robot.write_low_command(cmd)
