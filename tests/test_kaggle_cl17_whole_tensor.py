import ast
import re
import unittest

from scripts import kaggle_cl17_whole_tensor as cl17


class WholeTensorTest(unittest.TestCase):
    def test_generated_kernel_guardrails(self):
        source = cl17.KERNEL_SOURCE.replace("__RUN_ID__", "'40-1'").replace(
            "__GIT_SHA__", repr("a" * 40))
        ast.parse(source)
        self.assertEqual(cl17.SLUG, re.sub(r"[^a-z0-9]+", "-", cl17.TITLE.lower()).strip("-"))
        self.assertIn("4 * 1024 * 1024", source)
        self.assertIn("64 * 1024 * 1024", source)
        self.assertIn("128 * 1024 * 1024", source)
        self.assertIn("512 * 1024 * 1024", source)
        self.assertIn("whole_decoded = np.asarray(dequantize(whole_source.data, qtype))", source)
        self.assertIn("whole_source = None", source)
        self.assertIn("whole_candidates = None", source)
        self.assertIn("t = None", source)
        self.assertIn("stage.unlink()", source)
        self.assertIn('report["ltx_loaded"] = False', source)
        self.assertIn('report["video_generated"] = False', source)

    def test_validator_checks_exact_expansion_and_two_devices(self):
        sha, run = "a" * 40, "40-1"
        device = {"allocated_delta_bytes": 1024, "free_before_bytes": 10_000_000_000,
                  "free_after_bytes": 9_900_000_000}
        whole = {"quantization": "Q2_K", "compressed_bytes": 84,
                 "decoded_bytes": 1024, "decoded_elements": 256,
                 "decoded_dtype": "float32", "whole_tensor_dequantized": True,
                 "whole_model_loaded": False, "inference_performed": False,
                 "devices": [{"device_index": 0, **device}, {"device_index": 1, **device}]}
        report = {"git_sha": sha, "run_id": run, "install_completed": True,
                  "no_deps": True, "no_cache": True,
                  "direct_requirement_install_steps": [{"ok": True}] * 12,
                  "model_downloaded": True, "ltx_loaded": False, "video_generated": False,
                  "torch_before": "2.11", "cl17_torch_after": "2.11",
                  "cuda_before": "13.0", "cl17_cuda_after": "13.0",
                  "cl17_disk_free_after_bytes": 20_000_000_000,
                  "cl17_checks": {"first_weight": {
                      "download_complete": True, "sha256_verified": True, "stage_removed": True,
                      "gguf_structure": {"parsed": True},
                      "gpu_sample": {"devices": [{}, {}]},
                      "dequant_block": {"devices": [{}, {}]}, "whole_tensor": whole}}}
        cl17.validate_report(report, sha, run)
        for k, value in (("decoded_bytes", 2048), ("whole_model_loaded", True),
                         ("inference_performed", True), ("compressed_bytes", 85)):
            invalid = {**report, "cl17_checks": {"first_weight": {
                **report["cl17_checks"]["first_weight"],
                "whole_tensor": {**whole, k: value}}}}
            with self.subTest(k=k), self.assertRaises(ValueError):
                cl17.validate_report(invalid, sha, run)


if __name__ == "__main__":
    unittest.main()
