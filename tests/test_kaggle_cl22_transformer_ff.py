"""CL-22 static safety checks; do not run model or contact Kaggle."""
import re
import unittest

from scripts import kaggle_cl22_transformer_ff as cl22


class Cl22Tests(unittest.TestCase):
    def test_title_matches_kaggle_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl22.TITLE.lower()).strip("-"), cl22.SLUG)

    def test_real_bounded_feed_forward(self):
        s = cl22.KERNEL_SOURCE
        self.assertIn('blocks[0], "ff"', s)
        self.assertIn('y = ff(x)', s)
        self.assertIn('dims[0] * dims[1] > 80_000_000', s)
        self.assertIn('"attention_run": False', s)
        self.assertIn('"real_gguf_submodule": True', s)
        self.assertIn('"full_model_forward": False', s)
        self.assertIn('if peak_rss > 12_000_000_000:', s)
        self.assertIn('child.kill()', s)

    def test_incomplete_report_fails(self):
        with self.assertRaises(ValueError):
            cl22.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
