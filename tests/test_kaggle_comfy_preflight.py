import unittest

from scripts.kaggle_comfy_preflight import (
    KERNEL_SOURCE, SLUG, TITLE, parse_published_kernel, validate_report,
)


class ComfyPreflightTests(unittest.TestCase):
    def test_canonical_slug_matches_title(self):
        self.assertEqual(SLUG, TITLE.lower().replace(" ", "-"))
        output = ("Kernel version 1 successfully pushed. Please check progress at "
                  "https://www.kaggle.com/code/neonhawk/debatt-ai-movie-comfyui-preflight")
        self.assertEqual(parse_published_kernel(output, "neonhawk"),
                         "neonhawk/debatt-ai-movie-comfyui-preflight")
        with self.assertRaises(ValueError):
            parse_published_kernel(output, "wrong-owner")

    def test_source_only_no_weights_or_heavy_pip_install(self):
        self.assertIn('"--depth", "1"', KERNEL_SOURCE)
        self.assertNotIn("pip install", KERNEL_SOURCE)
        self.assertNotIn("hf_hub_download", KERNEL_SOURCE)
        self.assertNotIn("snapshot_download", KERNEL_SOURCE)

    def test_reject_stale_or_unsafe_report(self):
        good = {"git_sha": "a" * 40, "run_id": "123-1", "source_only": True,
                "full_comfyui_installed": False, "comfyui_started": False,
                "model_downloaded": False, "ltx_loaded": False,
                "video_generated": False, "gpu_devices": [{"name": "Tesla T4"}],
                "source_repos": {"comfyui": {"cloned": True},
                                 "gguf_plugin": {"cloned": True}}}
        validate_report(good, "a" * 40, "123-1")
        for field, value in [("run_id", "old-1"), ("model_downloaded", True),
                             ("comfyui_started", True), ("source_only", False)]:
            with self.assertRaises(ValueError):
                validate_report({**good, field: value}, "a" * 40, "123-1")

