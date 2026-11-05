import unittest

from scripts.kaggle_gguf_probe import KERNEL_SOURCE, validate_report


class KaggleGGUFProbeTests(unittest.TestCase):
    def test_no_model_download_code(self):
        self.assertNotIn("hf_hub_download", KERNEL_SOURCE)
        self.assertNotIn("snapshot_download", KERNEL_SOURCE)
        self.assertIn("gguf>=0.13,<1", KERNEL_SOURCE)

    def test_provenance_and_safety(self):
        good = {"git_sha": "a" * 40, "run_id": "123-1", "gguf_install_succeeded": True,
                "gguf_version": "0.17", "gpu_devices": [{"name": "Tesla T4"}],
                "model_downloaded": False, "ltx_loaded": False, "video_generated": False}
        validate_report(good, "a" * 40, "123-1")
        for key, value in [("run_id", "122-1"), ("video_generated", True),
                           ("gguf_install_succeeded", False), ("model_downloaded", True)]:
            changed = {**good, key: value}
            with self.assertRaises(ValueError):
                validate_report(changed, "a" * 40, "123-1")
