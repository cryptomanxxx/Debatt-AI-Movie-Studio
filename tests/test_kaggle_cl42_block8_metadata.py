"""Offline CL-42 metadata-only guard tests (never download model)."""
import ast
import re
import unittest
from unittest.mock import patch

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
        self.assertIn('def scan_tensor_index(reader):', code)
        self.assertIn('block8, all_block_indices = scan_tensor_index(reader)', code)
        self.assertIn('"under_projection_80m_guard": logical <= 80_000_000', code)
        self.assertIn('"comfy_model_constructed": False', code)
        self.assertIn('"gpu_inference": False', code)
        self.assertIn('"video_generated": False', code)
        self.assertIn('"reader_mode": "read_only_mmap"', code)
        self.assertNotIn('torch.cuda.synchronize(', code)

    def test_incomplete_report_rejected(self):
        with self.assertRaises(ValueError):
            cl42.validate_report({}, "sha", "run")

    def test_missing_or_invalid_quantization_rejected(self):
        tensor = {"name": "transformer_blocks.8.attn1.to_q.weight",
                  "shape": [4, 4], "logical_elements": 16,
                  "under_projection_80m_guard": True,
                  "quantization_type": "2"}
        metadata = {"source": "read_only_gguf_tensor_index",
                    "comfy_model_constructed": False, "gpu_inference": False,
                    "full_model_forward": False, "video_generated": False,
                    "tensors": [tensor], "block8_tensor_count": 1,
                    "block8_exists_in_gguf_index": True,
                    "observed_video_block_indices": [8],
                    "all_block8_tensors_under_80m": True}
        report = {"cl42_checks": {"first_weight": {"block8_metadata": metadata}}}
        with patch.object(cl42, "validate_cl14"):
            cl42.validate_report(report, "sha", "run")
            for invalid in (None, [], {}, "", 42, "arbitrary object"):
                changed = dict(tensor, quantization_type=invalid)
                modified = {"cl42_checks": {"first_weight": {
                    "block8_metadata": {**metadata, "tensors": [changed]}}}}
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    cl42.validate_report(modified, "sha", "run")


if __name__ == "__main__":
    unittest.main()
