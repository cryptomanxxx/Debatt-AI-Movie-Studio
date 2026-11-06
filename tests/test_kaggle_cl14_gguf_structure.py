import ast
import re
import unittest

from scripts import kaggle_cl14_gguf_structure as cl14


class GGUFStructureTest(unittest.TestCase):
    def test_kernel_is_valid_and_guarded(self):
        source = cl14.KERNEL_SOURCE.replace("__RUN_ID__", "'10-1'").replace(
            "__GIT_SHA__", repr("a" * 40))
        ast.parse(source)
        self.assertEqual(cl14.SLUG, re.sub(r"[^a-z0-9]+", "-", cl14.TITLE.lower()).strip("-"))
        self.assertIn('gguf.GGUFReader(str(stage), "r")', source)
        self.assertIn("signal.setitimer(signal.ITIMER_REAL, 720)", source)
        self.assertIn('Path("/kaggle/temp")', source)
        self.assertIn("stage.unlink()", source)
        self.assertIn('report["ltx_loaded"] = False', source)
        self.assertIn('report["video_generated"] = False', source)

    def test_unparsed_or_loaded_report_rejected(self):
        sha, run = "a" * 40, "10-1"
        report = {
            "git_sha": sha, "run_id": run, "install_completed": True,
            "no_deps": True, "no_cache": True,
            "direct_requirement_install_steps": [{"ok": True}] * 12,
            "model_downloaded": True, "ltx_loaded": False, "video_generated": False,
            "torch_before": "2.11", "cl14_torch_after": "2.11",
            "cuda_before": "13.0", "cl14_cuda_after": "13.0",
            "cl14_disk_free_after_bytes": 20_000_000_000,
            "cl14_checks": {"first_weight": {
                "target_filename": "ltx-2.5-22b-distilled-Q2_K.gguf",
                "download_complete": True, "sha256_verified": True, "stage_removed": True,
                "gguf_structure": {"parsed": True, "tensor_count": 42,
                                   "reader_mode": "read_only_mmap",
                                   "neural_weights_loaded": False, "gpu_allocated": False},
            }},
        }
        cl14.validate_report(report, sha, run)
        for key in ("parsed", "gpu_allocated", "neural_weights_loaded"):
            bad = {**report, "cl14_checks": {"first_weight": {
                **report["cl14_checks"]["first_weight"], "gguf_structure": {
                    **report["cl14_checks"]["first_weight"]["gguf_structure"],
                    key: not report["cl14_checks"]["first_weight"]["gguf_structure"][key],
                }}}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                cl14.validate_report(bad, sha, run)


if __name__ == "__main__":
    unittest.main()
