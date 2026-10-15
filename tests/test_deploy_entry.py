"""Exercise mode/task routing without constructing a hardware interface."""

import os
import json
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import deploy
import download_hf_model
from rl_policy import ufo_policy
from scripts.onboard import interface_config
from utils.policy import validate_task_config


def test_public_model_download_selects_matching_deployment_files(monkeypatch, tmp_path):
    captured = {}

    def download(**kwargs):
        captured.update(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr(download_hf_model, "snapshot_download", download)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    download_hf_model.main(["--output-dir", str(tmp_path), "--revision", "test-release"])
    assert captured["repo_id"] == "xuewang/ufo-g1-policy"
    assert captured["revision"] == "test-release"
    assert captured["local_dir"] == str(tmp_path)
    from fnmatch import fnmatch

    patterns = captured["allow_patterns"]
    for name in ("exported/FBcprAuxModel.onnx", "exported/FBcprAuxModel.meta.json",
                 "exported/FBcprAuxModel.deploy.yaml", "tasks/reward.yaml",
                 "tracking_inference/zs_7.pkl", "goal_inference/goal_reaching.pkl",
                 "reward_inference/reward_locomotion.pkl"):
        assert any(fnmatch(f"g1_policy/{name}", pattern) for pattern in patterns)
    for name in ("checkpoint/model/model.safetensors", "tracking_inference/tracking_7.mp4"):
        assert not any(fnmatch(f"g1_policy/{name}", pattern) for pattern in patterns)


@pytest.fixture
def entry(monkeypatch, tmp_path):
    captured = {}
    model = tmp_path / "custom.onnx"
    model.with_suffix(".deploy.yaml").write_text("export_marker: selected-model\n")
    monkeypatch.delenv("G1_INTERFACE", raising=False)
    monkeypatch.setenv("UFO_REAL_ROBOT_OK", "1")

    def construct(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(run=lambda: captured.update(ran=True))

    def validate(name, *args, **kwargs):
        captured["interface_checked"] = name
        return name == "robot-dds"

    monkeypatch.setattr(ufo_policy, "UFODeployPolicy", construct)
    monkeypatch.setattr(interface_config, "validate_interface", validate)
    return model, captured


@pytest.mark.parametrize("mode", ["sim2sim", "sim2real"])
@pytest.mark.parametrize("task", ["tracking", "goal", "reward", "teleop"])
def test_eight_modes_select_matching_robot_task_and_export(entry, mode, task):
    model, captured = entry
    argv = ["--mode", mode, "--task", task, "--model", str(model)]
    if mode == "sim2real":
        argv += ["--interface", "robot-dds"]
    deploy.main(argv)
    assert captured["ran"]
    assert captured["policy_config"] == {"export_marker": "selected-model"}
    assert captured["model_path"] == str(model)
    assert captured["robot_config"]["ROBOT_TYPE"] == ("g1_real" if mode == "sim2real" else "g1_29dof")
    assert captured["exp_config"]["type"] == ("tracking" if task == "teleop" else task)
    assert captured["exp_config"].get("ctx_source", "pkl") == ("zmq" if task == "teleop" else "pkl")
    assert ("interface_checked" in captured) == (mode == "sim2real")
    assert not captured["pico_control"]


@pytest.mark.parametrize("task, config_task", [("goal", "reward"), ("tracking", "teleop"), ("teleop", "tracking")])
def test_task_override_cannot_silently_change_reference_source(entry, task, config_task):
    model, captured = entry
    with pytest.raises(SystemExit):
        deploy.main(["--mode", "sim2sim", "--task", task, "--model", str(model),
                     "--task-config", str(ROOT / f"config/tasks/{config_task}.yaml")])
    assert not captured


@pytest.mark.parametrize("mode, robot", [("sim2sim", "g1_real"), ("sim2real", "g1")])
def test_robot_override_cannot_change_execution_mode(entry, mode, robot):
    model, captured = entry
    with pytest.raises(SystemExit):
        deploy.main(["--mode", mode, "--task", "tracking", "--model", str(model),
                     "--robot-config", str(ROOT / f"config/robot/{robot}.yaml")])
    assert not captured


def test_real_gate_runs_before_loading_files_or_sdk(entry, monkeypatch):
    _, captured = entry
    monkeypatch.delenv("UFO_REAL_ROBOT_OK")
    with pytest.raises(SystemExit):
        deploy.main(["--mode", "sim2real", "--task", "teleop", "--model", "/missing/export.onnx"])
    assert not captured


def test_bad_interface_never_constructs_runtime(entry):
    model, captured = entry
    with pytest.raises(SystemExit):
        deploy.main(["--mode", "sim2real", "--task", "tracking", "--model", str(model), "--interface", "wifi"])
    assert captured == {"interface_checked": "wifi"}


@pytest.mark.parametrize('failure', ['onnx_load', 'onnx_shape', 'missing_z', 'nan_z', 'wrong_z_dim', 'bad_start', 'bad_window'])
def test_invalid_artifacts_fail_before_opening_g1(monkeypatch, tmp_path, failure):
    model = tmp_path / 'Policy.onnx'
    model.write_bytes(b'fake model')
    config = yaml.safe_load((ROOT / 'config/policy/g1_policy.yaml').read_text())
    metadata = {'deployment': config, 'observation_order': list(config['observation']['policy']),
                'history_includes_current': False, 'history_oldest_first': False,
                'onnx_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
                'z_dim': 256, 'actor_obs_dim': 721, 'output_action_dim': 29}
    model.with_suffix('.meta.json').write_text(json.dumps(metadata))
    context = np.ones((2, 255 if failure == 'wrong_z_dim' else 256), np.float32)
    if failure == 'nan_z':
        context[-1, 0] = np.nan
    z_path = tmp_path / 'z.pkl'
    if failure != 'missing_z':
        joblib.dump(context, z_path)
    task = {'type': 'tracking', 'ctx_path': str(z_path), 'start': 0, 'end': 100, 'stop': 0,
            'window_size': 1, 'gamma': .8}
    if failure == 'bad_start':
        task['start'] = 5
    if failure == 'bad_window':
        task['window_size'] = 0
    opened = []
    sdk = SimpleNamespace(LowState=lambda: SimpleNamespace(tick=0), G1Interface=lambda *_: opened.append(True))
    monkeypatch.setitem(sys.modules, 'g1_interface', sdk)
    monkeypatch.setenv('UFO_REAL_ROBOT_OK', '1')
    import onnxruntime

    def session(*args, **kwargs):
        if failure == 'onnx_load':
            raise ValueError('bad ONNX')
        tensor = lambda name, width: SimpleNamespace(name=name, shape=[None, width], type='tensor(float)')
        return SimpleNamespace(get_inputs=lambda: [tensor('actor_obs', 720 if failure == 'onnx_shape' else 721)],
                               get_outputs=lambda: [tensor('action', 29)])

    monkeypatch.setattr(onnxruntime, 'InferenceSession', session)
    from utils.policy import load_robot_config
    robot = load_robot_config(ROOT / 'config/robot/g1_real.yaml')
    with pytest.raises((ValueError, FileNotFoundError)):
        ufo_policy.UFODeployPolicy(robot, config, task, str(model))
    assert not opened


@pytest.mark.parametrize("address", ["tcp://10.1.2.3:28711", "tcp://*:28711", "tcp://localhost", "ipc:///tmp/remote"])
def test_real_teleop_rejects_remote_or_invalid_z_before_sdk(monkeypatch, address):
    monkeypatch.setenv("UFO_REAL_ROBOT_OK", "1")
    with pytest.raises(ValueError, match="onboard-only"):
        ufo_policy.UFODeployPolicy(robot_config={"ROBOT_TYPE": "g1_real"}, policy_config={},
                                  exp_config={"type": "tracking", "ctx_source": "zmq", "ctx_zmq_addr": address},
                                  model_path="/missing/model.onnx")


@pytest.mark.parametrize("address", ["tcp://127.0.0.1:28711", "tcp://localhost:28711", "tcp://[::1]:28711"])
def test_onboard_loopback_sources_are_accepted(address):
    validate_task_config({"type": "tracking", "ctx_source": "zmq", "ctx_zmq_addr": address}, "g1_real")


def test_custom_task_and_policy_paths_work_outside_repo(entry, monkeypatch, tmp_path):
    model, captured = entry
    config = tmp_path / "goal.yaml"
    config.write_text(yaml.safe_dump({"type": "goal", "ctx_path": "/export/z.pkl", "selected_goals": ["stand"]}))
    monkeypatch.chdir(tmp_path)
    deploy.main(["--mode", "sim2sim", "--task", "goal", "--model", str(model), "--task-config", str(config)])
    assert captured["exp_config"]["selected_goals"] == ["stand"]


def test_help_and_onboard_gate_do_not_require_artifacts_or_sdk(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", UFO_REAL_ROBOT_OK="0")
    env.pop("PYTHONPATH", None)
    help_result = subprocess.run([sys.executable, str(ROOT / "deploy.py"), "--help"],
                                 cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert help_result.returncode == 0, help_result.stderr
    gate = subprocess.run(["bash", str(ROOT / "scripts/onboard/run_policy.sh")],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert gate.returncode == 2 and "UFO_REAL_ROBOT_OK=1" in gate.stderr


@pytest.mark.parametrize("task", ["tracking", "goal", "reward", "teleop"])
def test_onboard_launcher_forwards_custom_export_without_forcing_default_config(tmp_path, task):
    # Substitute Python so the shell's native import probe and exec cannot touch hardware.
    python = tmp_path / "python"
    python.write_text(f"#!{sys.executable}\nimport json, sys\n"
                      "if sys.argv[1] == '-':\n    sys.stdin.read()\n"
                      "else:\n    print('ARGV=' + json.dumps(sys.argv[1:]))\n")
    python.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", UFO_REAL_ROBOT_OK="1", UFO_ROOT=str(ROOT))
    for key in ("POLICY_CONFIG", "TASK_CONFIG", "VENV_PATH", "ENABLE_PICO_POLICY_CONTROL"):
        env.pop(key, None)
    model = str(tmp_path / "custom export.onnx")
    result = subprocess.run(["bash", str(ROOT / "scripts/onboard/run_policy.sh"),
                             "--task", task, "--model", model],
                            env=env, cwd=tmp_path, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    argv = json.loads(next(line[5:] for line in result.stdout.splitlines() if line.startswith("ARGV=")))
    assert argv[:3] == ["deploy.py", "--mode", "sim2real"]
    assert argv[-4:] == ["--task", task, "--model", model]
    assert "--policy-config" not in argv
