from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from humanoidverse.agents.presets import build_agent_preset
from humanoidverse.train import _default_update_z_every_step, build_ufo_mjlab_config, canonical_agent_name, parse_args


ROOT = Path(__file__).resolve().parents[1]


class AgentAliasesTest(unittest.TestCase):
    def _parse(self, *args: str):
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["train.py", *args]), patch("sys.stderr", stderr):
            parsed = parse_args()
        return parsed, stderr.getvalue()

    def test_canonical_agent_names(self) -> None:
        self.assertEqual(canonical_agent_name("tech"), "tech")
        self.assertEqual(canonical_agent_name("tldr"), "tech")
        self.assertEqual(canonical_agent_name("fb"), "fb")

    def test_agent_tech_parses(self) -> None:
        args, warning = self._parse("--agent", "tech")
        self.assertEqual(args.agent, "tech")
        self.assertEqual(args.update_z_every_step, 10)
        self.assertEqual(warning, "")

    def test_agent_tldr_parses_as_deprecated_alias(self) -> None:
        args, warning = self._parse("--agent", "tldr")
        self.assertEqual(args.agent, "tech")
        self.assertEqual(args.update_z_every_step, 10)
        self.assertIn("agent=tldr is deprecated; use agent=tech instead.", warning)

    def test_tech_default_update_z_every_step(self) -> None:
        self.assertEqual(_default_update_z_every_step("tech"), 10)

    def test_tech_preset_uses_tech_wandb_group(self) -> None:
        selected = build_agent_preset(
            agent="tech",
            device="cpu",
            compile=False,
            update_z_every_step=10,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
            wandb_project="test",
        )
        self.assertEqual(selected["wandb_group"], "ufo_tech")
        self.assertEqual(selected["agent_cfg"].name, "TldrDistAuxAgent")
        self.assertEqual(selected["agent_cfg"].train.update_z_every_step, 10)

    def test_tech_build_config_uses_canonical_public_names(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_agent_alias_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            agent="tech",
        )
        self.assertEqual(cfg.wandb_gname, "ufo_tech")
        self.assertEqual(cfg.wandb_run_name, "ufo_tech")
        self.assertEqual(cfg.tags["agent"], "tech")

    def test_readme_exposes_tech_training(self) -> None:
        readme = (ROOT / "README.md").read_text()
        self.assertNotIn("TLDR Training", readme)
        self.assertIn("TeCH Training", readme)
        self.assertIn("--agent tech", readme)
        self.assertNotIn("  --agent tldr", readme)
        self.assertNotIn("ufo_tldr", readme)

    def test_readme_tldr_mentions_are_compatibility_only(self) -> None:
        readme = (ROOT / "README.md").read_text()
        for line in readme.splitlines():
            if "tldr" in line.lower():
                self.assertIn("deprecated compatibility alias", line.lower())


if __name__ == "__main__":
    unittest.main()
