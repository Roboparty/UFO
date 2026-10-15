from pathlib import Path
from types import SimpleNamespace
import os
import sys
import tempfile
import hashlib
import json
import copy
import subprocess

import yaml
import numpy as np
import joblib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.onboard import deploy_helpers
from scripts.onboard.check_deploy_artifacts import Reporter, check_manifest, load_manifest
from scripts.onboard.interface_config import (
    InterfaceInfo,
    parse_default_route_interfaces,
    parse_ip_br_addr,
    validate_interface,
)
from scripts.teleop import check_teleop_env
from utils.policy import load_context, resolve_context_path, select_contexts, validate_policy_config
from scripts.onboard.patch_g1_sdk import patch_sdk
from utils.strings import joint_values, unitree_joint_names
from scripts.onboard.check_policy_preflight import _load_context_dim
from utils.policy import load_robot_config


@pytest.mark.parametrize('entry', ['scripts/onboard/interface_config.py', 'scripts/teleop/check_teleop_env.py'])
def test_shared_helper_cli_imports_work_outside_repo(tmp_path, entry):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(root/entry), '--help'], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('task', ['tracking', 'reward', 'goal'])
def test_artifact_and_policy_checks_reject_bad_later_rows(tmp_path, task):
    from scripts.onboard.check_deploy_artifacts import _check_context
    values = np.ones((2, 256), np.float32)
    context = values if task == 'tracking' else {'a': values[:1], 'b': values[1:]}
    path = tmp_path / 'z.pkl'
    config = {'type': task, 'ctx_path': str(path)}
    if task == 'reward':
        config['selected_rewards_filter_z'] = {'a': [0]}
    for invalid in [False, True]:
        values[1, 0] = np.nan if invalid else 1.
        joblib.dump(context, path)
        artifact_report, policy_report = Reporter(), Reporter()
        _check_context(path, {'expected_last_dim': 256}, artifact_report)
        _load_context_dim(config, tmp_path/'Policy.onnx', policy_report)
        assert artifact_report.failures == policy_report.failures == int(invalid)


@pytest.mark.parametrize('apis, expected_failures', [
    ([], 1),
    (['init', 'is_body_data_available', 'get_body_joints_pose', 'get_body_timestamp_ns'], 1),
    (['init', *deploy_helpers.POLLING_APIS], 0),
    (['init', *deploy_helpers.CALLBACK_APIS], 0),
])
def test_all_sdk_preflights_apply_the_same_capability_rule(monkeypatch, apis, expected_failures):
    from scripts.onboard import check_g1_onboard_env as onboard, check_xrobot_sdk
    module = SimpleNamespace(**{name: lambda: None for name in apis})
    monkeypatch.setattr(onboard, '_check_import', lambda *args: module)
    monkeypatch.setitem(sys.modules, 'xrobotoolkit_sdk', module)
    monkeypatch.setattr(check_xrobot_sdk, '_add_sdk_paths', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['check_xrobot_sdk.py', '--no-init'])
    reports = Reporter(), Reporter()
    onboard._check_xrobotoolkit_sdk(reports[0])
    check_teleop_env._check_xrobotoolkit_api(module, reports[1])
    assert [report.failures for report in reports] == [expected_failures] * 2
    assert check_xrobot_sdk.main() == expected_failures


def test_robot_base_and_joint_override_survive_flattened_overlay(tmp_path):
    from scripts.onboard.interface_config import _write_yaml
    (tmp_path/'base.yaml').write_text('ROBOT_TYPE: g1_29dof\njoint_velocity_limit: {a: 1, b: 2}\n')
    path = tmp_path/'robot.yaml'
    path.write_text('base: base.yaml\nROBOT_TYPE: g1_real\njoint_velocity_limit: {a: 3}\n')
    config = load_robot_config(path)
    assert config == {'ROBOT_TYPE': 'g1_real', 'joint_velocity_limit': {'a': 3, 'b': 2}}
    standalone = tmp_path/'standalone.yaml'
    _write_yaml(standalone, config)
    (tmp_path/'base.yaml').unlink()
    assert load_robot_config(standalone) == config
    with pytest.raises(FileNotFoundError):
        load_robot_config(path)
    path.write_text('base: robot.yaml\n')
    with pytest.raises(ValueError, match='nested/cyclic'):
        load_robot_config(path)


def test_real_and_sim_g1_share_joint_limits():
    root = Path(__file__).resolve().parents[1]
    sim = load_robot_config(root/'config/robot/g1.yaml')
    real = load_robot_config(root/'config/robot/g1_real.yaml')
    for key in ['joint_pos_lower_limit', 'joint_pos_upper_limit', 'joint_velocity_limit', 'joint_effort_limit']:
        assert len(sim[key]) == 29
        assert real[key] == sim[key]
    assert real['ROBOT_TYPE'] == 'g1_real' and real['USE_JOYSTICK']


def test_simulation_pd_and_xml_use_current_g1_torque_caps():
    import mujoco
    from sim_env.utils.simulation_bridge import SimulationBridge
    root = Path(__file__).resolve().parents[1]
    bridge = SimulationBridge.__new__(SimulationBridge)
    bridge.robot_config = load_robot_config(root / 'config/robot/g1.yaml')
    bridge.mj_model = mujoco.MjModel.from_xml_path(str(root / 'data/robots/g1/scene_29dof_freebase.xml'))
    bridge.mj_data = mujoco.MjData(bridge.mj_model)
    bridge.init_joint_indices()
    bridge.torques = np.zeros(bridge.mj_model.nu)
    bridge.low_cmd_sub = None
    bridge.has_received_command = True
    bridge.cmd_q = np.full(29, 10.)
    bridge.cmd_dq = bridge.cmd_tau = bridge.cmd_kd = np.zeros(29)
    bridge.cmd_kp = np.full(29, 100.)
    bridge.compute_torques()
    for name, limit in bridge.robot_config['joint_effort_limit'].items():
        actuator = bridge.mj_model.actuator(name.removesuffix('_joint')).id
        assert bridge.torques[actuator] == limit
        np.testing.assert_allclose(bridge.mj_model.actuator_ctrlrange[actuator], [-limit, limit])
    assert bridge.robot_config['joint_effort_limit']['left_hip_pitch_joint'] == 139.
    assert bridge.robot_config['joint_effort_limit']['left_wrist_pitch_joint'] == 13.4


def test_legacy_viewer_defaults_resolve_without_opening_viewer(monkeypatch):
    from scripts.vis import mujoco_mocap_viewer as viewer
    from utils.common import PORTS
    captured = {}
    def fake_viewer(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(main_loop=lambda: None)
    monkeypatch.setattr(viewer, 'MuJoCoMocapViewer', fake_viewer)
    monkeypatch.setattr(sys, 'argv', ['mujoco_mocap_viewer.py'])
    viewer.main()
    assert (Path(__file__).resolve().parents[1]/captured['mujoco_model_path']).is_file()
    assert captured['root_pose_port'] == PORTS['pelvis_pose']


def test_artifact_manifest_parser_accepts_valid_manifest():
    manifest = load_manifest(Path("model/g1_policy/artifact_manifest.yaml"))
    assert manifest["version"] == 1
    assert any(item["type"] == "policy_onnx" for item in manifest["artifacts"])


def test_context_loader_reads_all_three_numpy_export_formats(tmp_path):
    for value in (np.ones((5,256), np.float32), {'stand': [np.ones((1,256), np.float32)]}, {'goal': np.ones((1,256), np.float32)}):
        path = tmp_path / 'z.pkl'
        joblib.dump(value, path)
        restored = load_context(path)
        if isinstance(value, dict):
            key = next(iter(value))
            np.testing.assert_array_equal(restored[key], value[key])
        else:
            np.testing.assert_array_equal(restored, value)


@pytest.mark.parametrize("as_list", [False, True])
def test_reward_selection_preserves_config_order_and_values(as_list):
    contexts = {"stand": [np.full((1, 4), 10), np.full((1, 4), 20)],
                "walk": [np.full((1, 4), 30)]}
    choices = {"walk": [0], "stand": [1, 0]}
    if as_list:
        choices = [{"reward": key, "z_ids": ids} for key, ids in choices.items()]
    values, names = select_contexts(contexts, {"type": "reward", "selected_rewards_filter_z": choices})
    np.testing.assert_array_equal(values, np.repeat([[30], [20], [10]], 4, axis=1))
    assert names == ['Reward="walk"__Z_id=0', 'Reward="stand"__Z_id=1', 'Reward="stand"__Z_id=0']


def test_goal_selection_preserves_order_and_defaults_to_all():
    contexts = {"a": np.ones((1, 4)), "b": np.full((1, 4), 2)}
    values, names = select_contexts(contexts, {"type": "goal", "selected_goals": ["b", "a"]})
    assert names == ["b", "a"]
    np.testing.assert_array_equal(values[:, 0], [2, 1])
    assert select_contexts(contexts, {"type": "goal"})[1] == ["a", "b"]


@pytest.mark.parametrize("config", [
    {"type": "reward"},
    {"type": "reward", "selected_rewards_filter_z": {}},
    {"type": "reward", "selected_rewards_filter_z": {"missing": [0]}},
    {"type": "reward", "selected_rewards_filter_z": {"stand": [-1]}},
    {"type": "reward", "selected_rewards_filter_z": {"stand": [1]}},
    {"type": "goal", "selected_goals": []},
    {"type": "goal", "selected_goals": ["missing"]},
])
def test_invalid_context_selection_fails_explicitly(config):
    with pytest.raises(ValueError):
        select_contexts({"stand": np.ones((1, 4))}, config)


def test_selected_context_must_be_one_finite_vector():
    for value in (np.ones((2, 4)), np.full((1, 4), np.nan), np.empty((1, 0))):
        with pytest.raises(ValueError, match="finite z vector"):
            select_contexts({"stand": value}, {"type": "goal"})


def test_joint_values_preserve_joint_order_defaults_and_overlap_checks():
    names = ["right_knee", "left_hip", "left_knee", "waist"]
    values = joint_values({"left_.*": 2, "right_knee": 3, "unused": 9}, names, default=np.inf)
    np.testing.assert_array_equal(values, [3, 2, 2, np.inf])
    with pytest.raises(ValueError, match="Multiple matches"):
        joint_values({"left_.*": 2, "left_knee": 3}, names)


@pytest.mark.parametrize('task', ['tracking', 'reward', 'goal'])
def test_preflight_checks_every_context_row_and_all_task_types(tmp_path, task):
    values = np.ones((2,256), np.float32)
    path = tmp_path/'z.pkl'
    context = values if task == 'tracking' else {'stand': values}
    if task == 'goal':
        context = {'first': values[:1], 'second': values[1:]}
    config = {'type': task, 'ctx_path': str(path)}
    if task == 'reward':
        config['selected_rewards_filter_z'] = {'stand': [0]}
    joblib.dump(context, path)
    reporter = Reporter()
    assert _load_context_dim(config, tmp_path/'Policy.onnx', reporter) == 256
    assert reporter.failures == 0
    values[1,0] = np.nan
    joblib.dump(context, path)
    assert _load_context_dim(config, tmp_path/'Policy.onnx', reporter) is None
    assert reporter.failures == 1


def test_metadata_rejects_same_dimension_different_scale_order_and_weights(tmp_path):
    model = tmp_path / 'Policy.onnx'
    model.write_bytes(b'offline fixture')
    config = {'policy_joint_names': list(unitree_joint_names), 'observation': {'policy': {'base_ang_vel': {'scale': 1.}, 'prev_actions': {'scale': 1.}}}}
    metadata = {'deployment': config, 'observation_order': ['base_ang_vel', 'prev_actions'], 'history_includes_current': False, 'history_oldest_first': False, 'onnx_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}
    model.with_suffix('.meta.json').write_text(json.dumps(metadata))
    validate_policy_config(model, config)
    changed = copy.deepcopy(config)
    changed['observation']['policy']['base_ang_vel']['scale'] = .25
    with pytest.raises(ValueError, match='observation'):
        validate_policy_config(model, changed)
    changed = copy.deepcopy(config)
    changed['observation']['policy'] = dict(reversed(list(config['observation']['policy'].items())))
    with pytest.raises(ValueError, match='order'):
        validate_policy_config(model, changed)
    model.write_bytes(b'different weights, same dimensions')
    with pytest.raises(ValueError, match='ONNX does not match'):
        validate_policy_config(model, config)


def test_context_resolution_uses_same_precedence_for_runtime_and_preflight(tmp_path):
    model = tmp_path / 'exported/Policy.onnx'
    model.parent.mkdir()
    preferred = tmp_path / 'tracking_inference/z.pkl'
    preferred.parent.mkdir()
    preferred.write_bytes(b'preferred')
    (model.parent / 'z.pkl').write_bytes(b'fallback')
    assert resolve_context_path(model, {'ctx_path':'z.pkl'}, 'tracking_inference_mjlab', ('tracking_inference',)) == preferred
    with pytest.raises(FileNotFoundError, match='Tried'):
        resolve_context_path(model, {'ctx_path':'missing.pkl'}, 'tracking_inference')


@pytest.mark.parametrize('task', ['tracking', 'reward', 'goal'])
def test_default_tasks_find_current_export_folders_before_legacy(tmp_path, task):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / f'config/tasks/{task}.yaml').read_text())
    current = tmp_path / f'{task}_inference' / config['ctx_path']
    legacy = tmp_path / f'{task}_inference_mjlab' / config['ctx_path']
    for path in [current, legacy]:
        path.parent.mkdir(); path.write_bytes(b'fixture')
    model = tmp_path / 'exported/Policy.onnx'
    assert resolve_context_path(model, config, f'{task}_inference', (f'{task}_inference_mjlab',)) == current
    current.unlink()
    assert resolve_context_path(model, config, f'{task}_inference', (f'{task}_inference_mjlab',)) == legacy


def test_sdk_tick_patch_is_idempotent_and_validates_before_writing(tmp_path):
    binding = tmp_path / 'python_binding'
    binding.mkdir()
    sources = {
        'g1_interface.hpp': 'struct PyLowState {\n};\nstruct MotorState {\n};\n',
        'g1_interface.cpp': '    MotorState ms_tmp;\n        py_state.motor.q = ms->q;\n',
        'python_bindings.cpp': '        .def_readwrite("imu", &PyLowState::imu, "IMU state")\n',
    }
    for name, source in sources.items():
        (binding / name).write_text(source)
    patch_sdk(tmp_path)
    once = {name: (binding/name).read_text() for name in sources}
    patch_sdk(tmp_path)
    assert once == {name: (binding/name).read_text() for name in sources}
    for name, source in sources.items():
        assert (binding/(name+'.before_ufo_tick')).read_text() == source
    (binding/'g1_interface.hpp').write_text(sources['g1_interface.hpp'])
    (binding/'python_bindings.cpp').write_text('incompatible binding')
    with pytest.raises(ValueError, match='Unsupported SDK'):
        patch_sdk(tmp_path)
    assert (binding/'g1_interface.hpp').read_text() == sources['g1_interface.hpp']


def test_artifact_checker_reports_missing_required_file():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "artifacts": [
                        {"path": "missing.onnx", "type": "policy_onnx", "required": True}
                    ],
                }
            ),
            encoding="utf-8",
        )
        reporter = Reporter()
        check_manifest(manifest, root, reporter)
        assert reporter.failures >= 1


def test_artifact_checker_reports_hash_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        artifact = root / "artifact.bin"
        artifact.write_text("abc", encoding="utf-8")
        manifest = root / "manifest.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "artifacts": [
                        {
                            "path": "artifact.bin",
                            "type": "file",
                            "required": True,
                            "sha256": "0" * 64,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        reporter = Reporter()
        check_manifest(manifest, root, reporter)
        assert reporter.failures >= 1


def test_sdk_architecture_helpers_reject_x86_and_accept_aarch64():
    aarch64 = (
        "xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so: ELF 64-bit LSB shared object, "
        "ARM aarch64, version 1"
    )
    x86 = "xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so: ELF 64-bit LSB shared object, x86-64"
    assert deploy_helpers.file_output_is_cpython310_aarch64_extension(aarch64)
    assert not deploy_helpers.file_output_is_cpython310_aarch64_extension(x86)
    assert deploy_helpers.file_output_is_x86_64(x86)


def test_ldd_missing_libraries_parser():
    text = "\tlibssl.so.1.1 => not found\n\tlibc.so.6 => /lib/libc.so.6\n"
    assert deploy_helpers.ldd_missing_libraries(text) == ["libssl.so.1.1 => not found"]


def test_pth_idempotent_line_append():
    assert deploy_helpers.pth_lines_after_ensure("", "/sdk") == "/sdk\n"
    assert deploy_helpers.pth_lines_after_ensure("/sdk\n", "/sdk") == "/sdk\n"


def test_openssl_dir_status():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ok, missing = deploy_helpers.openssl11_dir_status(root)
        assert not ok
        assert set(missing) == {"libssl.so.1.1", "libcrypto.so.1.1"}
        (root / "libssl.so.1.1").write_text("", encoding="utf-8")
        (root / "libcrypto.so.1.1").write_text("", encoding="utf-8")
        ok, missing = deploy_helpers.openssl11_dir_status(root)
        assert ok
        assert missing == []


def test_interface_parser_and_validation():
    parsed = parse_ip_br_addr(
        "lo UNKNOWN 127.0.0.1/8\n"
        "eth0 UP 198.51.100.10/24 fe80::1/64\n"
        "wlan0 UP 203.0.113.5/24\n"
    )
    assert parsed["eth0"].ipv4_addresses == ("198.51.100.10/24",)
    assert parse_default_route_interfaces("default via 203.0.113.1 dev wlan0 proto dhcp") == {
        "wlan0"
    }

    reporter = Reporter()
    assert validate_interface(
        "eth0",
        reporter,
        interfaces=parsed,
        default_route_ifaces={"wlan0"},
    )
    reporter = Reporter()
    assert not validate_interface(
        "wlan0",
        reporter,
        interfaces=parsed,
        default_route_ifaces={"wlan0"},
    )


def test_viewer_yaml_preflight_detects_visualize_true():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = root / "g1.yaml"
        cfg.write_text("server:\n  visualize: true\n", encoding="utf-8")
        old = os.environ.get("TELEOP_CONFIG")
        os.environ["TELEOP_CONFIG"] = str(cfg)
        try:
            reporter = check_teleop_env.Reporter()
            assert check_teleop_env._teleop_yaml_requests_viewer(reporter)
        finally:
            if old is None:
                os.environ.pop("TELEOP_CONFIG", None)
            else:
                os.environ["TELEOP_CONFIG"] = old


if __name__ == "__main__":
    test_artifact_manifest_parser_accepts_valid_manifest()
    test_artifact_checker_reports_missing_required_file()
    test_artifact_checker_reports_hash_mismatch()
    test_sdk_architecture_helpers_reject_x86_and_accept_aarch64()
    test_ldd_missing_libraries_parser()
    test_pth_idempotent_line_append()
    test_openssl_dir_status()
    test_interface_parser_and_validation()
    test_viewer_yaml_preflight_detects_visualize_true()
    print("onboard deployment helper tests ok")
