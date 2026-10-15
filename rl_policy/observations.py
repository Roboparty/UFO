"""Actor observations and previous-frame history, in configured order."""

import numpy as np

from utils.math import quat_rotate_inverse_numpy


class Observation:
    def __init__(self, name, env, scale=1.0, steps=None, default_pos=None):
        self.env = env
        self.state_processor = env.state_processor
        self.scale = scale
        self.name = name.removesuffix("_history")
        sizes = {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "dof_pos_minus_default": env.state_processor.num_dof,
            "dof_vel": env.state_processor.num_dof,
            "prev_actions": env.num_actions,
        }
        if self.name not in sizes:
            raise ValueError(f"Unknown observation: {name}")
        if self.name == "dof_pos_minus_default":
            self.default_pos = np.asarray(default_pos)
            if self.default_pos.shape != (sizes[self.name],):
                raise ValueError(f"{name}: default_pos must have {sizes[self.name]} entries")
        self.history = None
        if name.endswith("_history"):
            if type(steps) is not int or steps <= 0:
                raise ValueError(f"{name}: steps must be a positive integer")
            self.history = np.zeros((steps, sizes[self.name]))

    def _current_value(self):
        state = self.state_processor
        if self.name == "base_ang_vel":
            return state.root_ang_vel_b
        if self.name == "projected_gravity":
            return quat_rotate_inverse_numpy(
                state.root_quat_b[None, :], np.array([[0, 0, -1]])
            ).squeeze(0)
        if self.name == "dof_pos_minus_default":
            return state.joint_pos - self.default_pos
        if self.name == "dof_vel":
            return state.joint_vel
        return self.env.last_action.reshape(-1)

    def compute(self, scale=1.0):
        value = self._current_value() if self.history is None else self.history.reshape(-1)
        return value * scale

    def update(self):
        if self.history is not None:
            self.history = np.roll(self.history, 1, axis=0)
            self.history[0] = self._current_value()

    def reset(self):
        if self.history is not None:
            self.history.fill(0)


class ObsGroup:
    def __init__(self, name, config, env):
        if not isinstance(config, dict) or not config:
            raise ValueError(f"Observation group {name!r} must be a nonempty mapping")
        self.name = name
        self.funcs = {}
        for obs_name, options in config.items():
            if options is None:
                options = {}
            if not isinstance(options, dict):
                raise ValueError(f"Observation config for {obs_name!r} must be a mapping")
            self.funcs[obs_name] = Observation(obs_name, env, **options)

    def compute(self):
        return np.concatenate([func.compute(scale=func.scale) for func in self.funcs.values()])

    def update(self):
        for func in self.funcs.values():
            func.update()

    def reset(self):
        for func in self.funcs.values():
            func.reset()
