"""CL-27 static safety tests; never contact Kaggle."""
import re
import unittest

from scripts import kaggle_cl27_eight_video_blocks as cl27


class Cl27Tests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl27.TITLE.lower()).strip("-"), cl27.SLUG)

    def test_eight_real_video_blocks_are_bounded(self):
        s = cl27.KERNEL_SOURCE
        self.assertIn('for index in range(8):', s)
        self.assertIn('"block_indices": list(range(8))', s)
        self.assertIn('vx, ax = blocks[index]((vx, ax)', s)
        self.assertIn('spec["gpu_allocated_after_bytes"]', s)
        self.assertIn('spec["gpu_peak_allocated_bytes"]', s)
        self.assertIn('"run_ax": False', s)
        self.assertIn('v_pe=None', s)
        self.assertIn('"full_model_forward": False', s)
        self.assertIn('dims[0] * dims[1] > 80_000_000', s)
        self.assertIn('4 * 1024 * 1024 * 1024', s)
        self.assertIn('if peak_rss > 12_000_000_000:', s)
        self.assertIn('child.kill()', s)

    def test_bad_report_rejected(self):
        with self.assertRaises(ValueError):
            cl27.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
