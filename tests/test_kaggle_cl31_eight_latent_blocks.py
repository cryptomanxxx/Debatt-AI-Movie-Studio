"""CL-31 static tests only; never execute GPU inference."""
import ast
import re
import unittest

from scripts import kaggle_cl31_eight_latent_blocks as cl31


class Cl31Tests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl31.TITLE.lower()).strip("-"), cl31.SLUG)

    def test_model_native_latents_enter_all_eight_blocks(self):
        code = cl31._EIGHT_LATENT_BLOCK
        self.assertIn('LTXVModel._process_input(root, latent, None, None)', code)
        self.assertIn('reshape(1, channels, 1, 2, 2)', code)
        self.assertIn('tuple(coords.shape) != (1, 3, 4, 2)', code)
        self.assertIn('LTXBaseModel._prepare_positional_embeddings(', code)
        self.assertIn('v_pe=pe', code)
        self.assertNotIn('v_pe=None', code)
        self.assertNotIn('coords = torch.tensor(', code)
        self.assertIn('for index in range(8):', code)
        self.assertEqual(code.count('LTXVModel._process_input('), 1)
        self.assertEqual(code.count('LTXBaseModel._prepare_positional_embeddings('), 1)
        self.assertIn('"coordinate_source": "latent_patchifier"', code)
        self.assertIn('"real_patchify_projection": True', code)
        self.assertIn('"vae_encoded": False', code)
        self.assertIn('"real_video_input": False', code)
        self.assertIn('"gpu_allocated_after_bytes"', code)
        self.assertIn('w.shape[0] * w.shape[1] > 80_000_000', code)
        self.assertIn('"run_ax": False', code)
        self.assertIn('if peak_rss > 12_000_000_000:', cl31.KERNEL_SOURCE)
        self.assertIn('child.kill()', cl31.KERNEL_SOURCE)
        ast.parse(code)

    def test_bad_report_rejected(self):
        with self.assertRaises(ValueError):
            cl31.validate_report({}, 'sha', 'run')


if __name__ == '__main__':
    unittest.main()
