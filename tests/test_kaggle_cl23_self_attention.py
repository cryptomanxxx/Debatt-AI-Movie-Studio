"""CL-23 static safety checks: never contact Kaggle or load LTX weights."""
import re
import unittest

from scripts import kaggle_cl23_self_attention as cl23


class Cl23Tests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl23.TITLE.lower()).strip("-"), cl23.SLUG)

    def test_two_token_attention_and_limits(self):
        s = cl23.KERNEL_SOURCE
        self.assertIn('blocks[0], "attn1"', s)
        self.assertIn('y = attention(x, pe=None, transformer_options={})', s)
        self.assertIn('in_width * 2', s)
        self.assertIn('"rotary_positions_applied": False', s)
        self.assertIn('"real_self_attention": True', s)
        self.assertIn('"full_transformer_block": False', s)
        self.assertIn('"full_model_forward": False', s)
        self.assertIn('dims[0] * dims[1] > 80_000_000', s)
        self.assertIn('3 * 1024 * 1024 * 1024', s)
        self.assertIn('if peak_rss > 12_000_000_000:', s)
        self.assertIn('child.kill()', s)

    def test_incomplete_report_rejected(self):
        with self.assertRaises(ValueError):
            cl23.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
