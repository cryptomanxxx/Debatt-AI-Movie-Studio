"""CL-36 offline static checks, without GPU or Kaggle execution."""
import ast
import re
import unittest

from scripts import kaggle_cl36_native_timestep_blocks as cl36


class Cl36Tests(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl36.TITLE.lower()).strip("-"), cl36.SLUG)

    def test_real_native_timestep_used_in_eight_blocks(self):
        code = cl36._NATIVE_BLOCKS_BLOCK
        self.assertIn('v_timestep=native_prepared[0]', code)
        self.assertIn('v_prompt_timestep=native_prepared[3]', code)
        self.assertIn('native_out_embed = native_embedded[0].data.expand(1, 4, width)', code)
        self.assertIn('root, native_vx, native_out_embed, None, **native_meta', code)
        self.assertIn('"native_timestep_into_blocks": True', code)
        self.assertIn('"real_ltx_scheduler": False', code)
        self.assertEqual(code.count('for index in range(8):'), 3)
        self.assertLess(code.index('adaln_preflight = []'), code.index('v_timestep=native_prepared[0]'))
        self.assertIn('child.kill()', cl36.KERNEL_SOURCE)
        self.assertIn('if peak_rss > 12_000_000_000:', cl36.KERNEL_SOURCE)
        ast.parse(code)

    def test_bad_report_rejected(self):
        with self.assertRaises(ValueError):
            cl36.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
