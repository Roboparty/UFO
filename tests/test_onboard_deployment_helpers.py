from pathlib import Path
import os
import sys
import tempfile

import yaml

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


def test_artifact_manifest_parser_accepts_valid_manifest():
    manifest = load_manifest(Path("model/g1_policy/artifact_manifest.yaml"))
    assert manifest["version"] == 1
    assert any(item["type"] == "policy_onnx" for item in manifest["artifacts"])


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
