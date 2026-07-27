from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def test_dependency_matrix_documents_current_flows():
    text = (ROOT / "docs" / "deployment_dependencies.md").read_text(encoding="utf-8")
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
    assert "XRoboToolkit, retargeting, GMR, or torch" in text
    assert "`requirements/teleop.txt` inherits `requirements/runtime.txt`" in text
    assert "realtime `z`, ONNX inference, policy code" in text


if __name__ == "__main__":
    test_teleop_requirements_exclude_gmr_and_torch()
    test_runtime_requirements_exclude_teleop_and_training_packages()
    test_training_requirements_inherit_runtime_and_keep_torch_separate()
    test_teleop_imports_vendored_motion_tracking_retarget()
    test_runtime_policy_source_has_no_xrobot_or_gmr_import()
    test_check_teleop_env_checks_current_retarget_not_gmr()
    test_dependency_matrix_documents_current_flows()
    test_onboard_docs_spell_out_runtime_plus_teleop_dependencies()
    print("dependency layout tests ok")
