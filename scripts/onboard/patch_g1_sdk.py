"""Expose the received low-state tick in EGalahad/unitree_sdk2's Python binding."""

import argparse
from pathlib import Path


def patch_sdk(sdk_root: Path) -> None:
    binding = sdk_root / "python_binding"
    replacements = {
        "g1_interface.hpp": [
            ("struct PyLowState {", "struct PyLowState {\n  uint32_t tick = 0;"),
            ("struct MotorState {", "struct MotorState {\n  uint32_t tick = 0;"),
        ],
        "g1_interface.cpp": [
            ("    MotorState ms_tmp;", "    MotorState ms_tmp;\n    ms_tmp.tick = low_state.tick();"),
            ("        py_state.motor.q = ms->q;", "        py_state.tick = ms->tick;\n        py_state.motor.q = ms->q;"),
        ],
        "python_bindings.cpp": [
            ('        .def_readwrite("imu", &PyLowState::imu, "IMU state")',
             '        .def_readonly("tick", &PyLowState::tick, "Received low-state tick")\n        .def_readwrite("imu", &PyLowState::imu, "IMU state")'),
        ],
    }
    updated = {}
    for name, edits in replacements.items():
        path = binding / name
        original = path.read_text()
        text = original
        for old, new in edits:
            if new in text:
                continue
            if text.count(old) != 1:
                raise ValueError(f"Unsupported SDK source at {path}; expected one occurrence of {old!r}")
            text = text.replace(old, new)
        if text != original:
            updated[path] = (original, text)
    # Validate every source before modifying any file; retain the originals.
    for path, (original, text) in updated.items():
        backup = path.with_suffix(path.suffix + ".before_ufo_tick")
        if not backup.exists():
            backup.write_text(original)
        path.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk_root", type=Path)
    args = parser.parse_args()
    patch_sdk(args.sdk_root)
    print("SDK tick patch applied. Rebuild g1_interface before starting robot control.")
