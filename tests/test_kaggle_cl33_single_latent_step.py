"""CL-33 static verification only; never triggers GPU or Kaggle work."""
import ast
import re
import unittest

from scripts import kaggle_cl33_single_latent_step as cl33


class Cl33Tests(unittest.TestCase):
    def test_slug_and_title(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl33.TITLE.lower()).strip("-"), cl33.SLUG)

    def test_one_step_after_real_output(self):
        code = cl33._SINGLE_STEP_BLOCK
        self.assertEqual(code.count('LTXVModel._process_input('), 1)
        self.assertEqual(code.count('LTXVModel._process_output('), 1)
        self.assertEqual(code.count('for index in range(8):'), 1)
        self.assertIn('"single_step": step_spec', code)
        self.assertIn('"real_ltx_scheduler": False', code)
        self.assertIn('"full_denoising_loop": False', code)
        self.assertIn('next_latent = previous + delta_sigma * velocity', code)
        self.assertLess(code.index('latent_output = LTXVModel._process_output('), code.index('next_latent = previous +'))
        self.assertIn('not torch.isfinite(next_latent).all().item()', code)
        self.assertIn('if peak_rss > 12_000_000_000:', cl33.KERNEL_SOURCE)
        self.assertIn('child.kill()', cl33.KERNEL_SOURCE)
        ast.parse(code)

    def test_bad_report_rejected(self):
        with self.assertRaises(ValueError):
            cl33.validate_report({}, 'sha', 'run')


if __name__ == '__main__':
    unittest.main()
