import sys

from ufo.tools.robot_config import builtin_robot_config, load_robot_training_spec
from ufo.train import build_mjlab_config, parse_args


def _config(**changes):
    return build_mjlab_config(**{
        "device": "cpu", "work_dir": "/tmp/rpo_config", "num_envs": 8,
        "num_env_steps": 32, "seed": 52, "use_wandb": False, "wandb_run_name": None,
        "robot_config": "configs/robots/rpo.yaml", **changes,
    })


def test_rpo_robot_and_fb_baseline():
    training = load_robot_training_spec(builtin_robot_config("rpo"))
    assert training.robot.name == "rpo_23dof"
    assert len(training.robot.control_joint_names) == 23
    assert (training.action_scale, training.action_rescale, training.action_clip_value) == (.4, False, 8.)
    assert training.actuator["model"] == "roboparty"
    assert (training.actuator["min_delay"], training.actuator["max_delay"]) == (0, 4)

    config = _config()
    assert config.agent.model.archi.shared_critic
    assert config.agent.model.archi.f.hidden_dim == 1024
    assert config.agent.train.use_bf16_amp
    assert config.agent.train.lr_discriminator == 2e-6
    assert config.agent.train.reg_coeff == config.agent.train.reg_coeff_aux == .03
    assert len(config.agent.aux_rewards) == 13
    assert config.prioritization_scale == 1.5


def test_rpo_cli_selects_reproduced_baseline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--robot", "rpo"])
    args = parse_args()
    assert args.robot_config == builtin_robot_config("rpo")
    assert args.seed == 52
    assert args.num_env_steps == 768_000_000
    assert args.data_path.endswith("rpo_23dof_lafan_grounded_10s-clipped.pkl")
