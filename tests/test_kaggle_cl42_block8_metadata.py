"""Offline CL-42 metadata-only guard tests (never download model)."""
import ast
import re
import unittest

from scripts import kaggle_cl42_block8_metadata as cl42


class Cl42Tests(unittest.TestCase):
    def test_slug_and_syntax(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl42.TITLE.lower()).strip("-"), cl42.SLUG)
        ast.parse(cl42.KERNEL_SOURCE)

    def test_index_scan_not_inference(self):
        code = cl42.KERNEL_SOURCE
        self.assertIn('first["block8_metadata"] = {', code)
        self.assertIn('pattern = re.compile(', code)
        self.assertIn('for tensor in reader.tensors:', code)
        self.assertIn('"under_projection_80m_guard": logical <= 80_000_000', code)
        self.assertIn('"comfy_model_constructed": False', code)
        self.assertIn('"gpu_inference": False', code)
        self.assertIn('"video_generated": False', code)
        self.assertIn('"reader_mode": "read_only_mmap"', code)
        self.assertNotIn('torch.cuda.synchronize(', code)

    def test_incomplete_report_rejected(self):
        with self.assertRaises(ValueError):
            cl42.validate_report({}, "sha", "run")


if __name__ == "__main__":
    unittest.main()
