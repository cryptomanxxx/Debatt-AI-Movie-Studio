"""CL-20 static safety and report validation checks."""
import unittest

from scripts import kaggle_cl20_cpu_model as cl20


class Cl20Tests(unittest.TestCase):
    def test_cpu_only_construction_is_guarded(self):
        s = cl20.KERNEL_SOURCE
        self.assertIn('comfy.sd.load_diffusion_model_state_dict(', s)
        self.assertIn('"load_device": torch.device("cpu")', s)
        self.assertIn('"offload_device": torch.device("cpu")', s)
        self.assertIn('"custom_operations": GGMLOps()', s)
        self.assertIn('if peak_rss > 12_000_000_000:', s)
        self.assertIn('torch.cuda.max_memory_allocated(i)', s)
        self.assertIn('child.kill()', s)
        self.assertIn('"video_generated": False', s)
        self.assertIn('"comfy_model_constructed": True', s)
        self.assertIn('"cl20_checks"', s)

    def test_incomplete_report_is_rejected(self):
        with self.assertRaises(ValueError):
            cl20.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
