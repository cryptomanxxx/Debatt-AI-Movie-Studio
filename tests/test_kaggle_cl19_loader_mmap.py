"""Local static/report tests; never contact Kaggle or download model weights."""
import unittest

from scripts import kaggle_cl19_loader_mmap as cl19


class Cl19Tests(unittest.TestCase):
    def test_kernel_is_bounded_and_uses_real_loader(self):
        source = cl19.KERNEL_SOURCE
        self.assertIn('module.gguf_sd_loader(str(stage))', source)
        self.assertIn('"cl19_checks"', source)
        self.assertIn('cl19-sources', source)
        self.assertIn('signal.setitimer(signal.ITIMER_REAL, 120)', source)
        self.assertIn('"comfy_model_constructed": False', source)
        self.assertNotIn('comfy.sd.load_diffusion_model_state_dict(sd,', source)

    def test_rejects_missing_loader(self):
        with self.assertRaises(ValueError):
            cl19.validate_report({}, "a", "1")

    def test_unique_kaggle_identifiers(self):
        self.assertIn("cl19", cl19.SLUG)
        self.assertIn("cl19", str(cl19.RESULTS))
        self.assertIn("cl19", str(cl19.SUBMISSION))


if __name__ == "__main__":
    unittest.main()
