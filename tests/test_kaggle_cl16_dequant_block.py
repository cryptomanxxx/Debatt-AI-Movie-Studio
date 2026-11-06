import ast
import re
import unittest

from scripts import kaggle_cl16_dequant_block as cl16


class Q2BlockTest(unittest.TestCase):
    def test_generated_kernel_safety(self):
        source = cl16.KERNEL_SOURCE.replace("__RUN_ID__", "'30-1'").replace(
            "__GIT_SHA__", repr("a" * 40))
        ast.parse(source)
        self.assertEqual(cl16.SLUG, re.sub(r"[^a-z0-9]+", "-", cl16.TITLE.lower()).strip("-"))
        self.assertIn("GGMLQuantizationType.Q2_K", source)
        self.assertIn("block_elements, block_bytes = 256, 84", source)
        self.assertIn("dequantize(packed_block.reshape(1, block_bytes), qtype)", source)
        self.assertIn("np.isfinite(unpacked).all()", source)
        self.assertIn("q2_source = None", source)
        self.assertIn("q2_candidates = None", source)
        self.assertIn("signal.setitimer(signal.ITIMER_REAL, 720)", source)
        self.assertIn('report["ltx_loaded"] = False', source)
        self.assertIn('report["video_generated"] = False', source)

    def test_validator_rejects_missing_dequant_or_inference(self):
        sha, run = "a" * 40, "30-1"
        device = {"allocated_delta_bytes": 1024, "free_before_bytes": 10_000_000_000,
                  "free_after_bytes": 9_900_000_000}
        dequant = {"quantization": "Q2_K", "packed_bytes": 84,
                   "decoded_elements": 256, "decoded_bytes": 1024,
                   "decoded_dtype": "float32", "full_model_dequantized": False,
                   "neural_model_loaded": False, "inference_performed": False,
                   "devices": [{"device_index": 0, **device}, {"device_index": 1, **device}]}
        report = {"git_sha": sha, "run_id": run, "install_completed": True,
                  "no_deps": True, "no_cache": True,
                  "direct_requirement_install_steps": [{"ok": True}] * 12,
                  "model_downloaded": True, "ltx_loaded": False, "video_generated": False,
                  "torch_before": "2.11", "cl16_torch_after": "2.11",
                  "cuda_before": "13.0", "cl16_cuda_after": "13.0",
                  "cl16_disk_free_after_bytes": 20_000_000_000,
                  "cl16_checks": {"first_weight": {
                      "download_complete": True, "sha256_verified": True, "stage_removed": True,
                      "gguf_structure": {"parsed": True},
                      "gpu_sample": {"devices": [{}, {}]}, "dequant_block": dequant}}}
        cl16.validate_report(report, sha, run)
        for field, wrong in (("decoded_bytes", 2048), ("full_model_dequantized", True),
                             ("inference_performed", True)):
            bad = {**report, "cl16_checks": {"first_weight": {
                **report["cl16_checks"]["first_weight"],
                "dequant_block": {**dequant, field: wrong}}}}
            with self.subTest(field=field), self.assertRaises(ValueError):
                cl16.validate_report(bad, sha, run)


if __name__ == "__main__":
    unittest.main()
