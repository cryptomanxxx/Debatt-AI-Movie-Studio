"""CL-30 tests never run GPU or Kaggle."""
import ast
import re
import unittest

from scripts import kaggle_cl30_latent_patchify as cl30


class Cl30Tests(unittest.TestCase):
    def test_title_matches_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl30.TITLE.lower()).strip("-"), cl30.SLUG)

    def test_model_patchify_and_rope_contract(self):
        code = cl30._PATCHIFY_BLOCK
        self.assertIn('LTXVModel._process_input(root, latent, None, None)', code)
        self.assertIn('torch.linspace(-1, 1, channels * 4', code)
        self.assertIn('reshape(1, channels, 1, 2, 2)', code)
        self.assertIn('tuple(coords.shape) != (1, 3, 4, 2)', code)
        self.assertIn('LTXBaseModel._prepare_positional_embeddings(', code)
        self.assertIn('v_pe=pe', code)
        self.assertNotIn('coords = torch.tensor(', code)
        self.assertIn('"coordinate_source": "latent_patchifier"', code)
        self.assertIn('"real_patchify_projection": True', code)
        self.assertIn('"vae_encoded": False', code)
        self.assertIn('"real_video_input": False', code)
        self.assertIn('w.shape[0] * w.shape[1] > 80_000_000', code)
        self.assertIn('if peak_rss > 12_000_000_000:', cl30.KERNEL_SOURCE)
        self.assertIn('child.kill()', cl30.KERNEL_SOURCE)
        ast.parse(code)

    def test_bad_report_rejected(self):
        with self.assertRaises(ValueError):
            cl30.validate_report({}, 'sha', 'run')


if __name__ == '__main__':
    unittest.main()
