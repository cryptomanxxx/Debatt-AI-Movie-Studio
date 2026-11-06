"""CL-25 static tests; do not execute model or Kaggle."""
import re
import unittest

from scripts import kaggle_cl25_two_video_blocks as cl25


class Cl25Tests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl25.TITLE.lower()).strip("-"), cl25.SLUG)

    def test_bounded_two_block_execution(self):
        s = cl25.KERNEL_SOURCE
        self.assertIn('for index in (0, 1):', s)
        self.assertIn('vx, ax = blocks[index]((vx, ax)', s)
        self.assertIn('"block_indices": [0, 1]', s)
        self.assertIn('"video_branches_complete": True', s)
        self.assertIn('"run_ax": False', s)
        self.assertIn('"a2v_cross_attn": False', s)
        self.assertIn('"cross_attention_adaln": True', s)
        self.assertIn('v_pe=None', s)
        self.assertIn('"full_model_forward": False', s)
        self.assertIn('dims[0] * dims[1] > 80_000_000', s)
        self.assertIn('4 * 1024 * 1024 * 1024', s)
        self.assertIn('if peak_rss > 12_000_000_000:', s)
        self.assertIn('child.kill()', s)

    def test_missing_report_rejected(self):
        with self.assertRaises(ValueError):
            cl25.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
