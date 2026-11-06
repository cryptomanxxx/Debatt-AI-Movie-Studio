import ast
import re
import unittest

from scripts import kaggle_cl15_gpu_sample as cl15


class GPUTensorSampleTest(unittest.TestCase):
    def test_generated_kernel_guarded(self):
        source = cl15.KERNEL_SOURCE.replace("__RUN_ID__", "'20-1'").replace(
            "__GIT_SHA__", repr("a" * 40))
        ast.parse(source)
        self.assertEqual(cl15.SLUG, re.sub(r"[^a-z0-9]+", "-", cl15.TITLE.lower()).strip("-"))
        self.assertIn("torch.cuda.device_count() != 2", source)
        self.assertIn("1_000_000_000", source)
        self.assertIn("16_000_000", source)
        self.assertIn("4 * 1024 * 1024", source)
        self.assertIn("signal.setitimer(signal.ITIMER_REAL, 720)", source)
        self.assertIn('Path("/kaggle/temp")', source)
        self.assertIn("stage.unlink()", source)
        # All mmap-backed views are cleared in the GGUF reader's finally clause.
        self.assertIn("            sample = None\n            candidates = None", source)
        self.assertIn("            cpu_sample = None\n            sample_bytes = None", source)
        self.assertLess(source.index("            candidates = None"),
                        source.index("            del reader"))
        self.assertIn('report["ltx_loaded"] = False', source)
        self.assertIn('report["video_generated"] = False', source)

    def test_measurement_requires_two_guarded_gpus(self):
        sha, run = "a" * 40, "20-1"
        device = {"allocated_delta_bytes": 2048, "raw_sample_bytes": 2048,
                  "free_before_bytes": 12_000_000_000,
                  "free_during_bytes": 11_900_000_000}
        report = {
            "git_sha": sha, "run_id": run, "install_completed": True,
            "no_deps": True, "no_cache": True,
            "direct_requirement_install_steps": [{"ok": True}] * 12,
            "model_downloaded": True, "ltx_loaded": False, "video_generated": False,
            "torch_before": "2.11", "cl15_torch_after": "2.11",
            "cuda_before": "13.0", "cl15_cuda_after": "13.0",
            "cl15_disk_free_after_bytes": 20_000_000_000,
            "cl15_checks": {"first_weight": {
                "download_complete": True, "sha256_verified": True, "stage_removed": True,
                "gguf_structure": {"parsed": True, "tensor_count": 4349,
                                   "neural_weights_loaded": False},
                "gpu_sample": {"raw_packed_bytes_only": True, "sample_bytes": 2048,
                               "full_model_loaded": False, "inference_performed": False,
                               "devices": [{"device_index": 0, **device},
                                           {"device_index": 1, **device}]},
            }},
        }
        cl15.validate_report(report, sha, run)
        for invalid_devices in ([], [{"device_index": 0, **device}],
                                [{"device_index": 0, **device},
                                 {"device_index": 1, **{**device, "allocated_delta_bytes": 40_000_000}}]):
            bad = {**report, "cl15_checks": {"first_weight": {
                **report["cl15_checks"]["first_weight"], "gpu_sample": {
                    **report["cl15_checks"]["first_weight"]["gpu_sample"],
                    "devices": invalid_devices,
                }}}}
            with self.assertRaises(ValueError):
                cl15.validate_report(bad, sha, run)


if __name__ == "__main__":
    unittest.main()
