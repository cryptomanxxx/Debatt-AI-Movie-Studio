"""CL-21 local static regression tests; no downloads or GPU execution."""
import re
import unittest

from scripts import kaggle_cl21_gguf_linear as cl21


class Cl21Tests(unittest.TestCase):
    def test_canonical_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl21.TITLE.lower()).strip("-"), cl21.SLUG)

    def test_guarded_single_layer_forward(self):
        s = cl21.KERNEL_SOURCE
        self.assertIn('root = model.model.diffusion_model', s)
        self.assertIn('isinstance(layer, GGMLOps.Linear)', s)
        self.assertIn('shape[0] * shape[1] <= 8_000_000', s)
        self.assertNotIn('weight.numel() <= 8_000_000', s)
        self.assertIn('y = layer(x)', s)
        self.assertIn('torch.isfinite(y).all()', s)
        self.assertIn('"real_gguf_layer": True', s)
        self.assertIn('"full_model_forward": False', s)
        self.assertIn('if peak_rss > 12_000_000_000:', s)
        self.assertIn('child.kill()', s)

    def test_rejects_missing_result(self):
        with self.assertRaises(ValueError):
            cl21.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
