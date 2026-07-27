"""Small pure helpers for onboard deployment diagnostics."""

from __future__ import annotations

from pathlib import Path


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
