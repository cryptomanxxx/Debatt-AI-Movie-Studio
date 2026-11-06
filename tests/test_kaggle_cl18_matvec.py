import ast
import re
import unittest

from scripts import kaggle_cl18_matvec as cl18


class MatvecTest(unittest.TestCase):
    def test_kernel_arithmetic_is_bounded(self):
        source = cl18.KERNEL_SOURCE.replace("__RUN_ID__", "'50-1'").replace(
            "__GIT_SHA__", repr("a" * 40))
        ast.parse(source)
        self.assertEqual(cl18.SLUG, re.sub(r"[^a-z0-9]+", "-", cl18.TITLE.lower()).strip("-"))
        self.assertIn("cpu_result = torch.mv(cpu_matrix, cpu_input)", source)
        self.assertIn("gpu_result = torch.mv(gpu_matrix, gpu_input)", source)
        self.assertIn("torch.allclose(actual, cpu_result, rtol=1e-3, atol=1e-3)", source)
        self.assertIn("128 * 1024 * 1024", source)
        self.assertIn("512 * 1024 * 1024", source)
        self.assertIn("gpu_matrix = None", source)
        self.assertIn("stage.unlink()", source)
        self.assertIn('report["ltx_loaded"] = False', source)
        self.assertIn('report["video_generated"] = False', source)

    def test_validation_requires_cpu_gpu_agreement(self):
        sha, run = "a" * 40, "50-1"
        device = {"allocated_delta_bytes": 16_777_216,
                  "free_before_bytes": 15_000_000_000,
                  "free_after_bytes": 14_900_000_000,
                  "max_absolute_error": 0.0002, "cpu_gpu_allclose": True}
        proj = {"rows": 2048, "cols": 2048, "operation": "torch.mv",
                "input_type": "synthetic_linspace", "real_dequantized_weight_matrix": True,
                "full_model_forward": False, "video_generated": False,
                "devices": [{"device_index": 0, **device}, {"device_index": 1, **device}]}
        report = {"git_sha": sha, "run_id": run, "install_completed": True,
                  "no_deps": True, "no_cache": True,
                  "direct_requirement_install_steps": [{"ok": True}] * 12,
                  "model_downloaded": True, "ltx_loaded": False, "video_generated": False,
                  "torch_before": "2.11", "cl18_torch_after": "2.11",
                  "cuda_before": "13.0", "cl18_cuda_after": "13.0",
                  "cl18_disk_free_after_bytes": 20_000_000_000,
                  "cl18_checks": {"first_weight": {
                      "download_complete": True, "sha256_verified": True,
                      "stage_removed": True, "gguf_structure": {"parsed": True},
                      "gpu_sample": {"devices": [{}, {}]},
                      "dequant_block": {"devices": [{}, {}]},
                      "whole_tensor": {"whole_tensor_dequantized": True,
                                       "whole_model_loaded": False,
                                       "decoded_elements": 2048 * 2048},
                      "single_projection": proj}}}
        cl18.validate_report(report, sha, run)
        for key, bad_value in (("full_model_forward", True), ("input_type", "real_prompt"),
                               ("operation", "not_mv"), ("rows", 9000)):
            invalid = {**report, "cl18_checks": {"first_weight": {
                **report["cl18_checks"]["first_weight"],
                "single_projection": {**proj, key: bad_value}}}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                cl18.validate_report(invalid, sha, run)


if __name__ == "__main__":
    unittest.main()
