"""Small pure helpers for onboard deployment diagnostics."""

from __future__ import annotations

from pathlib import Path


class Reporter:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def ok(self, message):
        print(f"[OK] {message}")

    def info(self, message):
        print(f"[INFO] {message}")

    def warn(self, message):
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message):
        self.failures += 1
        print(f"[FAIL] {message}")

    def skip(self, message):
        print(f"[SKIP] {message}")


CALLBACK_APIS = ("register_frame_callback", "clear_frame_callback", "has_frame_callback")
POLLING_APIS = (
    "is_body_data_available", "get_body_joints_pose", "get_body_timestamp_ns",
    "get_A_button", "get_B_button", "get_X_button", "get_Y_button",
)


def check_xrobot_api(module, reporter):
    if not hasattr(module, "init"):
        reporter.fail("xrobotoolkit_sdk installed but missing init()")
        return
    missing_callback = [name for name in CALLBACK_APIS if not hasattr(module, name)]
    missing_polling = [name for name in POLLING_APIS if not hasattr(module, name)]
    if not missing_callback:
        reporter.ok("xrobotoolkit_sdk callback API available")
    if not missing_polling:
        reporter.ok("xrobotoolkit_sdk polling API available")
    if missing_callback and missing_polling:
        reporter.fail(
            "xrobotoolkit_sdk has neither callback nor polling API "
            f"(callback missing: {', '.join(missing_callback)}; polling missing: {', '.join(missing_polling)})"
        )


def file_output_is_aarch64_elf(text: str) -> bool:
    return "ELF 64-bit" in text and "ARM aarch64" in text and "x86-64" not in text


def file_output_is_cpython310_aarch64_extension(text: str) -> bool:
    return file_output_is_aarch64_elf(text) and "cpython-310-aarch64" in text


def file_output_is_x86_64(text: str) -> bool:
    return "x86-64" in text or "x86_64" in text


def ldd_missing_libraries(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if "not found" in line]


def pth_lines_after_ensure(existing_text: str, line: str) -> str:
    lines = [item for item in existing_text.splitlines() if item.strip()]
    if line not in lines:
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def openssl11_dir_status(path: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for name in ("libssl.so.1.1", "libcrypto.so.1.1"):
        if not (path / name).is_file():
            missing.append(name)
    return (not missing, missing)
