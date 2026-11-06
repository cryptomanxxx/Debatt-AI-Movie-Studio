"""CL-35 static checks; no GPU or Kaggle work."""
import ast
import re
import unittest

from scripts import kaggle_cl35_native_timestep_probe as cl35


class Cl35Tests(unittest.TestCase):
    def test_title_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl35.TITLE.lower()).strip("-"), cl35.SLUG)

    def test_native_timestep_probe_is_separate_from_forward(self):
        code = cl35._NATIVE_TIMESTEP_BLOCK
        self.assertEqual(code.count('root._prepare_timestep('), 1)
        self.assertIn('for ts_value in (1.0, 0.96875):', code)
        self.assertIn('from comfy.ldm.lightricks.av_model import LTXAVModel, CompressedTimestep', code)
        self.assertIn('"checkpoint_adaln_weights": True', code)
        self.assertIn('allowed_adaln_elements = 9 * 4096 * 4096', code)
        self.assertIn('dims != (9 * width, width)', code)
        self.assertIn('count > 80_000_000', code)
        self.assertIn('"preflight_checked_weights": adaln_preflight', code)
        self.assertLess(code.index('adaln_preflight = []'), code.index('root._prepare_timestep('))
        self.assertIn('"native_embedding_used_in_model_forward": False', code)
        self.assertIn('"real_ltx_scheduler": False', code)
        self.assertLess(code.index('final_latent = next_latent +'), code.index('root._prepare_timestep('))
        self.assertIn('child.kill()', cl35.KERNEL_SOURCE)
        self.assertIn('if peak_rss > 12_000_000_000:', cl35.KERNEL_SOURCE)
        ast.parse(code)

    def test_missing_report_rejected(self):
        with self.assertRaises(ValueError):
            cl35.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
