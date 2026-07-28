from __future__ import annotations

import ast
import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _requirements(path: str, seen: set[Path] | None = None) -> set[str]:
    req_path = ROOT / path
    seen = set() if seen is None else seen
    req_path = req_path.resolve()
    if req_path in seen:
        return set()
    seen.add(req_path)
    lines = req_path.read_text(encoding="utf-8").splitlines()
    result: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-r "):
            include = stripped.split(maxsplit=1)[1]
            result.update(_requirements(str((req_path.parent / include).relative_to(ROOT)), seen))
            continue
        name = stripped.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0]
        result.add(name.lower().replace("_", "-"))
    return result


def _raw_requirements(path: str) -> list[str]:
    return [
        line.strip()
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _imported_modules(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", 1)[0])
    return modules


def test_teleop_requirements_exclude_gmr_and_torch():
    teleop = _requirements("requirements/teleop.txt")
    assert "-r runtime.txt" in _raw_requirements("requirements/teleop.txt")
    assert "general-motion-retargeting" not in teleop
    assert "general_motion_retargeting" not in teleop
    assert "torch" not in teleop
    assert {
        "mink",
        "qpsolvers",
        "daqp",
        "mujoco",
        "numpy",
        "scipy",
        "pyyaml",
        "pyzmq",
        "onnxruntime",
    } <= teleop


def test_runtime_requirements_exclude_teleop_and_training_packages():
    runtime = _requirements("requirements/runtime.txt")
    assert "xrobotoolkit-sdk" not in runtime
    assert "xrobotoolkit_sdk" not in runtime
    assert "general-motion-retargeting" not in runtime
    assert "general_motion_retargeting" not in runtime
    assert "torch" not in runtime
    assert {
        "numpy",
        "mujoco",
        "onnxruntime",
        "pyyaml",
        "scipy",
        "pyzmq",
        "joblib",
        "loguru",
        "termcolor",
        "sshkeyboard",
        "huggingface-hub",
    } <= runtime


def test_training_requirements_inherit_runtime_and_keep_torch_separate():
    training = _requirements("requirements/training.txt")
    assert "-r runtime.txt" in _raw_requirements("requirements/training.txt")
    assert "torch" in training
    assert "onnxruntime" in training
    assert "general-motion-retargeting" not in training
    assert "general_motion_retargeting" not in training


def test_teleop_imports_vendored_motion_tracking_retarget():
    teleop_root = ROOT / "scripts" / "teleop"
    sys.path.insert(0, str(teleop_root))
    try:
        import motion_tracking_retarget
        from motion_tracking_retarget import joint_mapping
    finally:
        if sys.path[0] == str(teleop_root):
            sys.path.pop(0)

    assert Path(motion_tracking_retarget.__file__).resolve().is_relative_to(
        teleop_root / "motion_tracking_retarget"
    )
    assert joint_mapping.qpos_size("g1") == 36


def test_runtime_policy_source_has_no_xrobot_or_gmr_import():
    imports = _imported_modules("rl_policy/ufo_policy.py")
    assert "xrobotoolkit_sdk" not in imports
    assert "general_motion_retargeting" not in imports

    source = (ROOT / "rl_policy" / "ufo_policy.py").read_text(encoding="utf-8")
    assert "xrobotoolkit_sdk" not in source
    assert "general_motion_retargeting" not in source
    assert "GeneralMotionRetargeting" not in source


def test_check_teleop_env_checks_current_retarget_not_gmr():
    source = (ROOT / "scripts" / "teleop" / "check_teleop_env.py").read_text(
        encoding="utf-8"
    )
    assert "motion_tracking_retarget" in source
    assert "xrobotoolkit_sdk" in source
    assert "general_motion_retargeting" not in source
    assert "GeneralMotionRetargeting" not in source


def test_onboard_dependency_profiles_separate_external_sdks():
    source = (ROOT / "scripts" / "onboard" / "check_g1_onboard_env.py").read_text(
        encoding="utf-8"
    )
    assert 'PROFILE_CHOICES = ("ordinary", "teleop", "diagnostic", "all")' in source
    assert 'TARGET_CHOICES = ("workstation", "g1-onboard")' in source
    assert 'CONTROL_PROFILES = ("ordinary", "teleop", "all")' in source
    assert 'TELEOP_PROFILES = ("teleop", "all")' in source
    assert 'DIAGNOSTIC_PROFILES = ("diagnostic", "all")' in source


def test_onboard_python_environment_helpers_classify_venv_and_conda():
    from scripts.onboard import check_g1_onboard_env as onboard_env

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        venv = tmp / "ufo_deploy_venv"
        conda = tmp / "miniconda3" / "envs" / "ufo-deploy"
        (conda / "conda-meta").mkdir(parents=True)

        assert onboard_env._is_venv_python(
            prefix=str(venv),
            base_prefix="/usr",
            real_prefix=None,
        )
        assert not onboard_env._is_venv_python(
            prefix="/usr",
            base_prefix="/usr",
            real_prefix=None,
        )
        assert onboard_env._is_conda_python(
            prefix=str(conda),
            executable=str(conda / "bin" / "python"),
            environ={"CONDA_PREFIX": str(conda)},
        )
        assert not onboard_env._is_conda_python(
            prefix=str(venv),
            executable=str(venv / "bin" / "python"),
            environ={"CONDA_PREFIX": str(conda)},
        )


def test_workstation_target_does_not_reject_conda_or_venv():
    from scripts.onboard import check_g1_onboard_env as onboard_env

    reporter = onboard_env.Reporter()
    with contextlib.redirect_stdout(io.StringIO()):
        onboard_env._check_python_environment(
            reporter,
            target="workstation",
            allow_nondefault=False,
        )
    assert reporter.failures == 0


def test_onboard_env_checker_defines_supported_python_policy():
    source = (ROOT / "scripts" / "onboard" / "check_g1_onboard_env.py").read_text(
        encoding="utf-8"
    )
    assert 'SUPPORTED_WORKSTATION_ENV = "Workstation: Conda with Python 3.10"' in source
    assert 'SUPPORTED_ONBOARD_ENV = "G1 onboard: Python 3.10 venv"' in source
    assert "--target" in source
    assert "--allow-nondefault-python-env" in source
    assert "UFO_ALLOW_NONDEFAULT_ONBOARD_PY" in source
    assert "G1 onboard Python environment is a venv" in source
    assert "workstation target accepts Conda Python" in source


def test_onboard_profile_source_marks_skipped_dependencies():
    source = (ROOT / "scripts" / "onboard" / "check_g1_onboard_env.py").read_text(
        encoding="utf-8"
    )
    assert "Dependency profile: {args.profile}" in source
    assert "[SKIP]" in source
    assert "xrobotoolkit_sdk (teleop only)" in source
    assert "unitree_sdk2py (diagnostic only)" in source
    assert "g1_interface (control only)" in source
    assert "g1_interface.cpython-310-aarch64-linux-gnu.so" in source
    assert "g1_interface.cpython-38-aarch64-linux-gnu.so" not in source


def test_preflight_suite_profiles_gate_optional_checks():
    source = (ROOT / "scripts" / "onboard" / "run_preflight_suite.sh").read_text(
        encoding="utf-8"
    )
    assert "--profile PROFILE" in source
    assert "ordinary|teleop|diagnostic|all" in source
    assert "run_control_checks=1" in source
    assert "run_teleop_checks=1" in source
    assert "run_diagnostic_checks=1" in source
    assert "scripts/onboard/check_xrobot_sdk.py" in source
    assert "scripts/onboard/check_g1_state_readonly.py" in source
    assert 'env_check_args=(--profile "${PROFILE}" --target g1-onboard)' in source
    assert 'scripts/onboard/check_g1_onboard_env.py "${env_check_args[@]}"' in source


def test_preflight_suite_enforces_onboard_venv_by_default():
    source = (ROOT / "scripts" / "onboard" / "run_preflight_suite.sh").read_text(
        encoding="utf-8"
    )
    assert "The validated G1 onboard deployment uses a CPython 3.10 venv." in source
    assert "ONBOARD_ALLOW_NONDEFAULT_PY" in source
    assert "--target g1-onboard" in source
    assert "--allow-nondefault-python-env" in source
    assert 'ONBOARD_ALLOW_NONDEFAULT_PY="${ONBOARD_ALLOW_NONDEFAULT_PY:-0}"' in source


def test_xrobot_onboard_bootstrap_requires_python310_venv():
    source = (ROOT / "scripts" / "onboard" / "install_xrobot_sdk.sh").read_text(
        encoding="utf-8"
    )
    assert "validated G1 onboard CPython 3.10 aarch64 venv" in source
    assert "check_supported_onboard_python" in source
    assert "expected_soabi = \"cpython-310-aarch64-linux-gnu\"" in source
    assert "EXPECTED_XROBOT_EXT=\"xrobotoolkit_sdk.${EXPECTED_SOABI}.so\"" in source
    assert "XRoboToolkit Python extension ABI mismatch" in source
    assert "conda-meta" in source
    assert "selected Python is not a venv" in source
    assert "expected 3.10" in source


def test_xrobot_bootstrap_rejects_cpython38_extension():
    with tempfile.TemporaryDirectory() as tmpdir:
        sdk_root = Path(tmpdir)
        (sdk_root / "xrobotoolkit_sdk.cpython-38-aarch64-linux-gnu.so").write_bytes(b"")
        proc = subprocess.run(
            [
                "scripts/onboard/install_xrobot_sdk.sh",
                "--sdk-root",
                str(sdk_root),
                "--venv",
                sys.executable,
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    assert proc.returncode != 0
    assert "XRoboToolkit Python extension ABI mismatch" in proc.stdout
    assert "cpython-38-aarch64-linux-gnu" in proc.stdout
    assert "cpython-310-aarch64-linux-gnu" in proc.stdout


def test_dependency_matrix_documents_current_flows():
    text = (ROOT / "docs" / "deployment_dependencies.md").read_text(encoding="utf-8")
    assert "| Workstation | Conda with Python 3.10 |" in text
    assert "| G1 onboard | Python 3.10 venv |" in text
    assert "The validated G1 onboard deployment uses venv by default" in text
    assert "| Ordinary onboard Sim2Real | No | No | No | No | No |" in text
    assert (
        "| PICO Teleop Sim2Real | Yes | Yes | `motion_tracking_retarget` + Mink IK | No | No for retarget |"
        in text
    )
    assert "Current deploy does not require the external" in text
    assert "It inherits\n`requirements/runtime.txt`" in text


def test_onboard_docs_spell_out_runtime_plus_teleop_dependencies():
    text = (ROOT / "scripts" / "onboard" / "README.md").read_text(encoding="utf-8")
    assert "Ordinary Sim2Real does not need human pose" in text
    assert "XRoboToolkit, `xrobotoolkit_sdk`, PICO, retargeting, `unitree_sdk2py`, GMR" in text
    assert "`requirements/teleop.txt`\ninherits `requirements/runtime.txt`" in text
    assert "realtime\n`z`, ONNX inference, policy code" in text
    assert "UFO deploy has three different external dependency categories." in text
    assert "g1_interface.cpython-310-aarch64-linux-gnu.so" in text
    assert "g1_interface.cpython-38-aarch64-linux-gnu.so" in text
    assert "xrobotoolkit_sdk`" in text
    assert "unitree_sdk2py" in text
    assert "scripts/onboard/run_preflight_suite.sh --profile ordinary" in text
    assert "scripts/onboard/run_preflight_suite.sh --profile teleop" in text
    assert "scripts/onboard/run_preflight_suite.sh --profile diagnostic" in text
    assert "check_g1_onboard_env.py --target g1-onboard" in text
    assert "check_g1_onboard_env.py --profile teleop --target g1-onboard" in text


def test_readme_documents_official_python_environment_policy():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Official Python Environment Policy" in text
    assert "Workstation:\nConda with Python 3.10" in text
    assert "G1 onboard:\nPython 3.10 venv" in text
    assert "The validated G1 onboard deployment uses Python\n3.10 venv by default" in text
    assert "For the validated G1 onboard path, activate this venv rather than a Conda" in text
    assert "ONBOARD_PY=/home/unitree/ufo_deploy_venv/bin/python" in text


def test_onboard_readme_documents_venv_default_path():
    text = (ROOT / "scripts" / "onboard" / "README.md").read_text(encoding="utf-8")
    assert "## Official Onboard Python Environment" in text
    assert "Workstation:\nConda with Python 3.10" in text
    assert "G1 onboard:\nPython 3.10 venv" in text
    assert "/home/unitree/ufo_deploy_venv" in text
    assert "Onboard preflight validates the venv default when run with the G1 onboard" in text
    assert "Direct PC/CI calls to `check_g1_onboard_env.py` default to\n`--target workstation`" in text


def test_teleop_readme_keeps_conda_workstation_only_for_release_path():
    text = (ROOT / "scripts" / "teleop" / "README.md").read_text(encoding="utf-8")
    assert "On a workstation, the release-supported default is Conda:" in text
    assert "Use this `ufo-teleop` Conda environment for the workstation" in text
    assert "On the G1 onboard Jetson, the validated default is a Python 3.10 venv" in text
    assert "Direct onboard PICO teleop can use the same deployment venv" in text
    assert "/home/unitree/ufo_deploy_venv/bin/python" in text


def test_readme_no_longer_presents_all_sdks_as_required():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## External Runtime Dependencies" in text
    assert "UFO deploy has three different external dependency categories." in text
    assert "Required For Any Real G1 Control" in text
    assert "Required Only For Onboard PICO Teleop" in text
    assert "Optional Diagnostics" in text
    assert "Unitree SDK2 Python binding, including `g1_interface`" not in text
    assert "Install Unitree SDK" not in text
    assert "Missing `unitree_sdk2py` does not block ordinary Sim2Real or PICO teleop." in text
    assert "Ordinary Sim2Real does not need `xrobotoolkit_sdk`" in text


if __name__ == "__main__":
    test_teleop_requirements_exclude_gmr_and_torch()
    test_runtime_requirements_exclude_teleop_and_training_packages()
    test_training_requirements_inherit_runtime_and_keep_torch_separate()
    test_teleop_imports_vendored_motion_tracking_retarget()
    test_runtime_policy_source_has_no_xrobot_or_gmr_import()
    test_check_teleop_env_checks_current_retarget_not_gmr()
    test_onboard_dependency_profiles_separate_external_sdks()
    test_onboard_python_environment_helpers_classify_venv_and_conda()
    test_workstation_target_does_not_reject_conda_or_venv()
    test_onboard_env_checker_defines_supported_python_policy()
    test_onboard_profile_source_marks_skipped_dependencies()
    test_preflight_suite_profiles_gate_optional_checks()
    test_preflight_suite_enforces_onboard_venv_by_default()
    test_xrobot_onboard_bootstrap_requires_python310_venv()
    test_xrobot_bootstrap_rejects_cpython38_extension()
    test_dependency_matrix_documents_current_flows()
    test_onboard_docs_spell_out_runtime_plus_teleop_dependencies()
    test_readme_documents_official_python_environment_policy()
    test_onboard_readme_documents_venv_default_path()
    test_teleop_readme_keeps_conda_workstation_only_for_release_path()
    test_readme_no_longer_presents_all_sdks_as_required()
    print("dependency layout tests ok")
