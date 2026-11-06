"""CL-32 static checks; no Kaggle or GPU use."""
import ast
import re
import unittest

from scripts import kaggle_cl32_latent_output as cl32


class Cl32Tests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl32.TITLE.lower()).strip("-"), cl32.SLUG)

    def test_output_after_eight_blocks_is_bounded(self):
        code = cl32._LATENT_OUTPUT_BLOCK
        self.assertEqual(code.count('LTXVModel._process_input('), 1)
        self.assertEqual(code.count('LTXVModel._process_output('), 1)
        self.assertIn('**input_meta', code)
        self.assertIn('for index in range(8):', code)
        self.assertIn('"latent_output": output_spec', code)
        self.assertIn('"native_output_projection": True', code)
        self.assertIn('"native_unpatchify": True', code)
        self.assertIn('"synthetic_embedded_timestep": True', code)
        self.assertIn('tuple(latent_output.shape) != tuple(latent.shape)', code)
        self.assertIn('out_weight.shape[0] * out_weight.shape[1] > 80_000_000', code)
        self.assertIn('"vae_encoded": False', code)
        self.assertIn('"real_video_input": False', code)
        self.assertIn('if peak_rss > 12_000_000_000:', cl32.KERNEL_SOURCE)
        self.assertIn('child.kill()', cl32.KERNEL_SOURCE)
        ast.parse(code)

    def test_bad_report_rejected(self):
        with self.assertRaises(ValueError):
            cl32.validate_report({}, 'sha', 'run')


if __name__ == '__main__':
    unittest.main()
