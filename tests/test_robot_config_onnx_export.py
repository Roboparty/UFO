from __future__ import annotations

import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from humanoidverse.tracking_inference import _export_policy_model
from humanoidverse.utils.helpers import export_meta_policy_as_onnx


class _FakeSpace:
    def __init__(self, dim: int) -> None:
        self.shape = (int(dim),)


class _FakeInputFilter:
    def __init__(self, dim: int) -> None:
        self.output_space = _FakeSpace(dim)


class _FakeActor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_filter = _FakeInputFilter(input_dim)


class _FakePolicyModel(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        last_action_dim: int,
        history_actor_dim: int = 0,
        z_dim: int = 16,
        action_dim: int = 29,
        actor_input_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        if actor_input_keys is None:
            actor_input_keys = ["state", "last_action"]
        self.obs_space = {
            "state": _FakeSpace(state_dim),
            "last_action": _FakeSpace(last_action_dim),
            "history_actor": _FakeSpace(history_actor_dim),
        }
        self.action_dim = int(action_dim)
        actor_input_dim = sum(int(self.obs_space[key].shape[0]) for key in actor_input_keys)
        self._actor = _FakeActor(actor_input_dim)
        self.cfg = SimpleNamespace(
            archi=SimpleNamespace(
                z_dim=int(z_dim),
                actor=SimpleNamespace(input_filter=SimpleNamespace(key=list(actor_input_keys))),
            )
        )
        self.last_actor_dict_shapes: dict[str, tuple[int, ...]] = {}
        self.last_ctx_shape: tuple[int, ...] | None = None

    def act(self, actor_dict: dict[str, torch.Tensor], ctx: torch.Tensor) -> torch.Tensor:
        self.last_actor_dict_shapes = {key: tuple(value.shape) for key, value in actor_dict.items()}
        self.last_ctx_shape = tuple(ctx.shape)
        return torch.zeros(ctx.shape[0], self.action_dim, dtype=ctx.dtype, device=ctx.device)


def _fake_robot_training(num_dof: int) -> SimpleNamespace:
    return SimpleNamespace(
        config_path=Path("/tmp/fake_robot.yaml"),
        robot=SimpleNamespace(
            name="fake_robot",
            xml_path="/tmp/fake_robot.xml",
            control_joint_names=[f"joint_{idx}" for idx in range(num_dof)],
        ),
    )


class RobotConfigOnnxExportTest(unittest.TestCase):
    def _export_with_mock(self, model: _FakePolicyModel, output_name: str = "FakePolicyModel.onnx"):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        patcher = patch("humanoidverse.utils.helpers.torch.onnx.export")
        onnx_export = patcher.start()
        self.addCleanup(patcher.stop)
        metadata = export_meta_policy_as_onnx(model, Path(tmpdir.name), output_name)
        return metadata, onnx_export

    def test_g1_29dof_wrapper_uses_inferred_dimensions(self) -> None:
        model = _FakePolicyModel(state_dim=64, last_action_dim=29, z_dim=32, action_dim=29)
        metadata, onnx_export = self._export_with_mock(model)

        self.assertEqual(metadata["actor_input_keys"], ["state", "last_action"])
        self.assertEqual(metadata["actor_input_dims"], {"state": 64, "last_action": 29})
        self.assertEqual(metadata["actor_obs_dim"], 125)
        wrapper = onnx_export.call_args.args[0]
        example_input = onnx_export.call_args.args[1]
        self.assertEqual(tuple(example_input.shape), (1, 125))

        action = wrapper(torch.randn(2, 125))
        self.assertEqual(tuple(action.shape), (2, 29))
        self.assertEqual(wrapper.actor.last_actor_dict_shapes["state"], (2, 64))
        self.assertEqual(wrapper.actor.last_actor_dict_shapes["last_action"], (2, 29))
        self.assertEqual(wrapper.actor.last_ctx_shape, (2, 32))

    def test_non_g1_action_dims_export_without_g1_only_error(self) -> None:
        for action_dim in (23, 27):
            model = _FakePolicyModel(state_dim=40 + action_dim, last_action_dim=action_dim, action_dim=action_dim)
            metadata, _onnx_export = self._export_with_mock(model, output_name=f"FakePolicyModel{action_dim}.onnx")
            self.assertEqual(metadata["output_action_dim"], action_dim)
            self.assertEqual(metadata["actor_input_dims"]["last_action"], action_dim)

    def test_export_meta_policy_signature_does_not_accept_use_29dof(self) -> None:
        signature = inspect.signature(export_meta_policy_as_onnx)
        self.assertNotIn("use_29dof", signature.parameters)
        model = _FakePolicyModel(state_dim=64, last_action_dim=29)
        with self.assertRaises(TypeError):
            export_meta_policy_as_onnx(model, "/tmp", "policy.onnx", use_29dof=True)  # type: ignore[call-arg]

    def test_history_actor_key_contributes_to_actor_obs_dim(self) -> None:
        model = _FakePolicyModel(
            state_dim=52,
            last_action_dim=23,
            history_actor_dim=112,
            z_dim=24,
            action_dim=23,
            actor_input_keys=["state", "last_action", "history_actor"],
        )
        metadata, onnx_export = self._export_with_mock(model)

        self.assertEqual(
            metadata["actor_input_dims"],
            {"state": 52, "last_action": 23, "history_actor": 112},
        )
        self.assertEqual(metadata["actor_obs_dim"], 52 + 23 + 112 + 24)
        wrapper = onnx_export.call_args.args[0]
        wrapper(torch.randn(3, metadata["actor_obs_dim"]))
        self.assertEqual(wrapper.actor.last_actor_dict_shapes["history_actor"], (3, 112))

    def test_missing_actor_input_key_errors_clearly(self) -> None:
        model = _FakePolicyModel(
            state_dim=52,
            last_action_dim=23,
            history_actor_dim=112,
            actor_input_keys=["state", "last_action", "history_actor"],
        )
        del model.obs_space["history_actor"]
        with self.assertRaisesRegex(KeyError, "history_actor.*obs_space"):
            export_meta_policy_as_onnx(model, "/tmp", "policy.onnx")

    def test_tracking_inference_has_no_g1_only_num_dof_export_block(self) -> None:
        source = Path(__file__).resolve().parents[1] / "humanoidverse" / "tracking_inference.py"
        text = source.read_text()
        self.assertNotIn("num_dof != 29", text)
        self.assertNotIn("ONNX export currently supports only G1 29-DOF", text)

    def test_goal_and_reward_export_calls_use_current_signature(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative_path in ("humanoidverse/goal_inference.py", "humanoidverse/reward_inference.py"):
            source = root / relative_path
            tree = ast.parse(source.read_text())
            export_model = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "_export_model"
            )
            export_call = next(
                node
                for node in ast.walk(export_model)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "export_meta_policy_as_onnx"
            )

            self.assertEqual(len(export_call.args), 3, relative_path)
            self.assertEqual([keyword.arg for keyword in export_call.keywords], ["z_dim"], relative_path)

    def test_tracking_policy_export_writes_robot_metadata_json(self) -> None:
        model = _FakePolicyModel(
            state_dim=52,
            last_action_dim=27,
            history_actor_dim=100,
            z_dim=12,
            action_dim=27,
            actor_input_keys=["state", "last_action", "history_actor"],
        )
        with tempfile.TemporaryDirectory() as tmpdir, patch("humanoidverse.utils.helpers.torch.onnx.export"):
            metadata = _export_policy_model(model, Path(tmpdir), _fake_robot_training(27))
            metadata_path = Path(tmpdir) / "_FakePolicyModel.meta.json"
            self.assertTrue(metadata_path.exists())
            saved = json.loads(metadata_path.read_text())

        self.assertEqual(metadata["robot_name"], "fake_robot")
        self.assertEqual(saved["robot_name"], "fake_robot")
        self.assertEqual(saved["num_dof"], 27)
        self.assertEqual(saved["control_joint_names"][0], "joint_0")
        self.assertEqual(saved["actor_input_dims"]["last_action"], 27)
        self.assertEqual(saved["z_dim"], 12)
        self.assertEqual(saved["actor_obs_dim"], 52 + 27 + 100 + 12)
        self.assertEqual(saved["output_action_dim"], 27)


if __name__ == "__main__":
    unittest.main()
