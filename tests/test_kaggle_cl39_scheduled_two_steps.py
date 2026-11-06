"""CL-39 offline safety and scheduler integration tests (no GPU)."""
import ast
import re
import unittest

from scripts import kaggle_cl39_scheduled_two_steps as cl39
from scripts.cl38_ltx_sigma_schedule import ltxv_sigma_schedule


class Cl39Tests(unittest.TestCase):
    def test_slug_and_scheduler_evidence(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl39.TITLE.lower()).strip("-"), cl39.SLUG)
        self.assertEqual(cl39.SIGMAS, tuple(ltxv_sigma_schedule(tokens=4, steps=20)["sigmas"][:3]))
        self.assertEqual(len(cl39.SIGMAS), 3)
        self.assertTrue(all(-0.05 < b - a < 0 for a, b in zip(cl39.SIGMAS, cl39.SIGMAS[1:])))

    def test_native_two_evaluations_preserved(self):
        code = cl39._SCHEDULED_STEPS_BLOCK
        self.assertEqual(code.count("for index in range(8):"), 4)
        self.assertIn("second_native_input = first_native_update.to(dtype=latent.dtype)", code)
        self.assertIn("root, second_native_input, None, None", code)
        self.assertIn("v_timestep=linked_prepared[0]", code)
        self.assertIn('"scheduler_formula_reproduction": True', code)
        self.assertIn('"official_scheduler_executed": False', code)
        self.assertIn('"video_generated": False', code)
        self.assertIn("if peak_rss > 12_000_000_000:", cl39.KERNEL_SOURCE)
        self.assertIn("child.kill()", cl39.KERNEL_SOURCE)
        ast.parse(code)
        ast.parse(cl39.KERNEL_SOURCE)

    def test_missing_report_rejected(self):
        with self.assertRaises(ValueError):
            cl39.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
