"""CL-24 static tests: do not load the model or access Kaggle."""
import re
import unittest

from scripts import kaggle_cl24_video_block as cl24


class Cl24Tests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl24.TITLE.lower()).strip("-"), cl24.SLUG)

    def test_video_branch_is_bounded_and_explicit(self):
        s = cl24.KERNEL_SOURCE
        self.assertIn('block((vx, ax), v_context=context, v_timestep=timestep,', s)
        self.assertIn('"run_ax": False', s)
        self.assertIn('"a2v_cross_attn": False', s)
        self.assertIn('"v2a_cross_attn": False', s)
        self.assertIn('v_pe=None', s)
        self.assertIn('"video_branch_complete": True', s)
        self.assertIn('"synthetic_text_context": True', s)
        self.assertIn('"rotary_positions_applied": False', s)
        self.assertIn('"full_model_forward": False', s)
        self.assertIn('dims[0] * dims[1] > 80_000_000', s)
        self.assertIn('4 * 1024 * 1024 * 1024', s)
        self.assertIn('if peak_rss > 12_000_000_000:', s)
        self.assertIn('child.kill()', s)

    def test_incomplete_report_rejected(self):
        with self.assertRaises(ValueError):
            cl24.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
