"""CL-24B static tests; do not run Kaggle or the model."""
import re
import unittest

from scripts import kaggle_cl24b_prompt_adaln as cl24b


class Cl24bTests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl24b.TITLE.lower()).strip("-"), cl24b.SLUG)

    def test_real_prompt_adaln_and_bounded_video_branch(self):
        s = cl24b.KERNEL_SOURCE
        self.assertIn('prompt_scale_shift_table', s)
        self.assertIn('cross_attention_adaln', s)
        self.assertIn('ada_count == 9', s)
        self.assertIn('2 * context_width', s)
        self.assertIn('v_prompt_timestep=prompt_timestep', s)
        self.assertIn('"synthetic_prompt_timestep": True', s)
        self.assertIn('"cross_attention_adaln": True', s)
        self.assertIn('"run_ax": False', s)
        self.assertIn('v_pe=None', s)
        self.assertIn('4 * 1024 * 1024 * 1024', s)
        self.assertIn('if peak_rss > 12_000_000_000:', s)
        self.assertIn('child.kill()', s)

    def test_invalid_report_rejected(self):
        with self.assertRaises(ValueError):
            cl24b.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
