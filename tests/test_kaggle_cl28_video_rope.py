"""CL-28 static safety checks; never trigger Kaggle."""
import re
import unittest

from scripts import kaggle_cl28_video_rope as cl28


class Cl28Tests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl28.TITLE.lower()).strip("-"), cl28.SLUG)

    def test_real_rotary_path_is_explicit_and_bounded(self):
        s = cl28.KERNEL_SOURCE
        self.assertIn("LTXBaseModel._prepare_positional_embeddings(", s)
        self.assertIn("root, coords, frame_rate=25", s)
        self.assertNotIn("pe = root._prepare_positional_embeddings(", s)
        self.assertIn("isinstance(root, LTXBaseModel)", s)
        self.assertIn("tuple(coords.shape) != (1, 3, 4, 2)", s)
        self.assertIn("not root.use_middle_indices_grid", s)
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

    def test_middle_grid_coordinates_executable_contract(self):
        # Runs without a GPU or torch; catches regressions in the actual
        # coordinate literal spliced into the Kaggle kernel.
        coords = cl28.SYNTHETIC_COORDINATE_PAIRS
        self.assertEqual(len(coords), 1)
        self.assertEqual(len(coords[0]), 3)
        for axis in coords[0]:
            self.assertEqual(len(axis), 4)
            for pair in axis:
                self.assertEqual(len(pair), 2)
                self.assertLess(pair[0], pair[1])
        centers = [tuple((coords[0][axis][token][0] + coords[0][axis][token][1]) / 2
                         for axis in range(3)) for token in range(4)]
        self.assertEqual(len(set(centers)), 4)
        self.assertIn(repr(coords), cl28.KERNEL_SOURCE)

    def test_bad_report_rejected(self):
        with self.assertRaises(ValueError):
            cl28.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
