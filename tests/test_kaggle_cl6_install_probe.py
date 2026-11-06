import unittest

from scripts.kaggle_cl6_install_probe import (
    KERNEL_SOURCE, SLUG, TITLE, published_kernel, validate_report,
)


class CL6InstallTests(unittest.TestCase):
    def test_notebook_slug_matches_title(self):
        self.assertEqual(TITLE.lower().replace(" ", "-"), SLUG)
        output = ("Kernel version 1 successfully pushed. Check "
                  "https://www.kaggle.com/code/neonhawk/debatt-ai-movie-cl6-dependency-probe")
        self.assertEqual(published_kernel(output, "neonhawk"),
                         "neonhawk/debatt-ai-movie-cl6-dependency-probe")
        with self.assertRaises(ValueError):
            published_kernel(output, "other-user")

    def test_limits_and_no_ltx_weights(self):
        self.assertIn('"--no-deps"', KERNEL_SOURCE)
        self.assertIn('"--no-cache-dir"', KERNEL_SOURCE)
        self.assertIn('"--only-binary=:all:"', KERNEL_SOURCE)
        self.assertIn("MIN_FREE_DISK_BYTES = 19_300_000_000", KERNEL_SOURCE)
        self.assertNotIn("snapshot_download", KERNEL_SOURCE)
        self.assertNotIn("hf_hub_download", KERNEL_SOURCE)

    def test_provenance_safety_and_success_validation(self):
        good = {"git_sha": "a" * 40, "run_id": "123-1", "install_completed": True,
                "no_deps": True, "no_cache": True, "torch_before": "2.11",
                "torch_after": "2.11", "cuda_before": "13.0", "cuda_after": "13.0",
                "disk_free_after_bytes": 19_800_000_000,
                "full_comfyui_installed": False, "comfyui_started": False,
                "model_downloaded": False, "ltx_loaded": False,
                "video_generated": False, "gpu_devices": [{"name": "Tesla T4"}],
                "direct_requirement_install_steps": [{"ok": True}] * 12}
        validate_report(good, "a" * 40, "123-1")
        for key, value in [("install_completed", False), ("model_downloaded", True),
                           ("no_deps", False), ("torch_after", "different"),
                           ("disk_free_after_bytes", 18_000_000_000)]:
            with self.assertRaises(ValueError):
                validate_report({**good, key: value}, "a" * 40, "123-1")

