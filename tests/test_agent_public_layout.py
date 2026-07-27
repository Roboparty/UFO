from __future__ import annotations

import unittest


class AgentPublicLayoutTest(unittest.TestCase):
    def test_fb_public_imports(self) -> None:
        from humanoidverse.agents.fb import FBAgentConfig
        from humanoidverse.agents.fb.aux import FBcprAuxAgentConfig
        from humanoidverse.agents.fb.preset import build_fb_agent

        self.assertEqual(FBAgentConfig.model_fields["name"].default, "FBAgent")
        self.assertEqual(FBcprAuxAgentConfig.model_fields["name"].default, "FBcprAuxAgent")
        self.assertTrue(callable(build_fb_agent))

    def test_tech_public_imports(self) -> None:
        from humanoidverse.agents.tech import TeCHAgentConfig, TeCHModelConfig, build_tech_agent
        from humanoidverse.agents.tech.agent import TldrDistAuxAgentConfig

        self.assertIs(TeCHAgentConfig, TldrDistAuxAgentConfig)
        self.assertEqual(TeCHAgentConfig().name, "TldrDistAuxAgent")
        self.assertEqual(TeCHModelConfig().name, "GcrRlDistAuxModel")
        self.assertTrue(callable(build_tech_agent))

    def test_common_public_imports(self) -> None:
        from humanoidverse.agents.common.nn_filters import DictInputFilterConfig
        from humanoidverse.agents.common.normalizers import BatchNormNormalizerConfig

        self.assertEqual(DictInputFilterConfig(key=["state"]).name, "DictInputFilterConfig")
        self.assertEqual(BatchNormNormalizerConfig().name, "BatchNormNormalizerConfig")

    def test_tracker_namespace_exists(self) -> None:
        import humanoidverse.agents.tracker as tracker

        self.assertEqual(tracker.__all__, [])


if __name__ == "__main__":
    unittest.main()
