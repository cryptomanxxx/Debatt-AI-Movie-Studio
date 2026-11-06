"""CL-29 static/in-process safety tests; do not run Kaggle."""
import ast
import re
import unittest

from scripts import kaggle_cl29_eight_blocks_rope as cl29
from scripts.kaggle_cl28_video_rope import SYNTHETIC_COORDINATE_PAIRS


class Cl29Tests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl29.TITLE.lower()).strip("-"), cl29.SLUG)

    def test_shared_rope_across_eight_blocks_with_guards(self):
        block = cl29._EIGHT_ROPE_BLOCK
        self.assertEqual(block.count('LTXBaseModel._prepare_positional_embeddings('), 1)
        self.assertIn('for index in range(8):', block)
        self.assertIn('v_pe=pe', block)
        self.assertNotIn('v_pe=None', block)
        self.assertIn('tuple(coords.shape) != (1, 3, 4, 2)', block)
        self.assertIn('not root.use_middle_indices_grid', block)
        self.assertIn('"tokens": 4', block)
        self.assertIn('"rotary_positions_applied": True', block)
        self.assertIn('"gpu_allocated_after_bytes"', block)
        self.assertIn('"gpu_peak_allocated_bytes"', block)
        self.assertIn('dims[0] * dims[1] > 80_000_000', block)
        self.assertIn('"run_ax": False', block)
        self.assertIn('"full_model_forward": False', block)
        self.assertIn(repr(SYNTHETIC_COORDINATE_PAIRS), block)
        ast.parse(block)
        self.assertIn('if peak_rss > 12_000_000_000:', cl29.KERNEL_SOURCE)
        self.assertIn('4 * 1024 * 1024 * 1024', cl29.KERNEL_SOURCE)
        self.assertIn('child.kill()', cl29.KERNEL_SOURCE)

    def test_invalid_report_rejected(self):
        with self.assertRaises(ValueError):
            cl29.validate_report({}, 'sha', 'run')


if __name__ == '__main__':
    unittest.main()
