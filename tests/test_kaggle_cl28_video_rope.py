"""CL-28 static safety checks; never trigger Kaggle."""
import re
import unittest

from scripts import kaggle_cl28_video_rope as cl28


class Cl28Tests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl28.TITLE.lower()).strip("-"), cl28.SLUG)

    def test_real_rotary_path_is_explicit_and_bounded(self):
        s = cl28.KERNEL_SOURCE
        self.assertIn("_prepare_positional_embeddings(coords, frame_rate=25", s)
        self.assertIn("v_pe=pe", s)
        self.assertIn('"rotary_positions_applied": True', s)
        self.assertIn('"coordinate_source": "synthetic_grid"', s)
        self.assertIn('"tokens": 4', s)
        self.assertIn('"frame_rate": 25', s)
        self.assertIn('"full_model_forward": False', s)
        self.assertIn('dims[0] * dims[1] > 80_000_000', s)
        self.assertIn('4 * 1024 * 1024 * 1024', s)
        self.assertIn('if peak_rss > 12_000_000_000:', s)
        self.assertIn('child.kill()', s)

    def test_bad_report_rejected(self):
        with self.assertRaises(ValueError):
            cl28.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
