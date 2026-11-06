"""CL-34 offline checks; no model weights, Kaggle or GPU required."""
import ast
import re
import unittest

from scripts import kaggle_cl34_two_latent_steps as cl34


class Cl34Tests(unittest.TestCase):
    def test_title_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl34.TITLE.lower()).strip("-"), cl34.SLUG)

    def test_second_model_forward_receives_updated_input(self):
        code = cl34._TWO_STEP_BLOCK
        self.assertEqual(code.count('LTXVModel._process_output('), 2)
        self.assertEqual(code.count('for index in range(8):'), 2)
        self.assertIn('second_input = next_latent.to(dtype=latent.dtype)', code)
        self.assertIn('root, second_input, None, None)', code)
        self.assertIn('second_vx, second_ax = blocks[index](', code)
        self.assertIn('"second_input_from_first_update": True', code)
        self.assertIn('"real_ltx_scheduler": False', code)
        self.assertIn('"full_denoising_loop": False', code)
        self.assertLess(code.index('next_latent = previous +'), code.index('second_input = next_latent.to('))
        self.assertIn('child.kill()', cl34.KERNEL_SOURCE)
        self.assertIn('if peak_rss > 12_000_000_000:', cl34.KERNEL_SOURCE)
        ast.parse(code)

    def test_missing_report_rejected(self):
        with self.assertRaises(ValueError):
            cl34.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
