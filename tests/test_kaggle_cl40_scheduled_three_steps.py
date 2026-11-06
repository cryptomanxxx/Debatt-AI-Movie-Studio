"""Offline CL-40 integration and safety tests, no GPU / model downloads."""
import ast
import re
import unittest

from scripts import kaggle_cl40_scheduled_three_steps as cl40
from scripts.cl38_ltx_sigma_schedule import ltxv_sigma_schedule


class Cl40Tests(unittest.TestCase):
    def test_slug_and_schedule(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl40.TITLE.lower()).strip("-"), cl40.SLUG)
        self.assertEqual(cl40.SIGMAS, tuple(ltxv_sigma_schedule(tokens=4, steps=20)["sigmas"][:4]))
        self.assertTrue(all(-0.05 < b - a < 0 for a, b in zip(cl40.SIGMAS, cl40.SIGMAS[1:])))

    def test_three_native_evaluations_and_feedback(self):
        code = cl40._THREE_STEPS_BLOCK
        self.assertEqual(code.count("for index in range(8):"), 5)
        self.assertIn("third_native_input = final_native_latent.to(dtype=latent.dtype)", code)
        self.assertIn("root, third_native_input, None, None", code)
        self.assertIn("third_ts = torch.full((1, 4), native_sigmas[2]", code)
        self.assertIn("third_native_latent = final_native_latent + third_delta * third_velocity.float()", code)
        self.assertIn('"native_conditioned_evaluations": 3', code)
        self.assertIn('"official_scheduler_executed": False', code)
        self.assertIn('"video_generated": False', code)
        self.assertLess(code.index("final_native_latent ="), code.index("third_native_input ="))
        self.assertIn("if peak_rss > 12_000_000_000:", cl40.KERNEL_SOURCE)
        self.assertIn("child.kill()", cl40.KERNEL_SOURCE)
        ast.parse(code)
        ast.parse(cl40.KERNEL_SOURCE)

    def test_missing_report_is_rejected(self):
        with self.assertRaises(ValueError):
            cl40.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
