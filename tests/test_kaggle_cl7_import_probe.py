import unittest

from scripts.kaggle_cl7_import_probe import KERNEL_SOURCE, validate_report


class CL7ImportTests(unittest.TestCase):
    def test_no_model_and_no_server(self):
        self.assertNotIn("snapshot_download", KERNEL_SOURCE)
        self.assertNotIn("hf_hub_download", KERNEL_SOURCE)
        self.assertNotIn("main.py", KERNEL_SOURCE)
        self.assertIn('"--no-deps"', KERNEL_SOURCE)
        self.assertIn('"pip", "check"', KERNEL_SOURCE)
        self.assertIn("MIN_FREE_DISK_BYTES = 19_300_000_000", KERNEL_SOURCE)

    def test_report_validation_accepts_missing_import_deps_as_diagnostic(self):
        good = {
            "git_sha": "a" * 40, "run_id": "123-1", "install_completed": True,
            "no_deps": True, "no_cache": True,
            "direct_requirement_install_steps": [{"ok": True}] * 12,
            "torch_before": "2.11", "cl7_torch_after": "2.11",
            "cuda_before": "13", "cl7_cuda_after": "13",
            "cl7_disk_free_after_bytes": 20_000_000_000,
            "comfyui_started": False, "model_downloaded": False,
            "ltx_loaded": False, "video_generated": False,
            "cl7_checks": {name: {"ok": False, "error": "missing dependency"}
                           for name in ("pip_check", "ComfyUI", "ComfyUI-GGUF", "comfy_nodes_import")},
        }
        validate_report(good, "a" * 40, "123-1")
        for change in ({"model_downloaded": True}, {"cl7_torch_after": "changed"},
                       {"cl7_disk_free_after_bytes": 0}, {"cl7_checks": {}}):
            with self.assertRaises(ValueError):
                validate_report({**good, **change}, "a" * 40, "123-1")
