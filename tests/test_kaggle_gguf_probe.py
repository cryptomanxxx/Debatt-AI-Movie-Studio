import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.kaggle_gguf_probe import (
    KERNEL_SOURCE, SLUG, parse_published_kernel, retrieve_diagnostics, validate_report,
)


class KaggleGGUFProbeTests(unittest.TestCase):
    def test_submission_slug_matches_existing_kaggle_notebook(self):
        self.assertEqual(SLUG, "debatt-ai-movie-gguf-environment-probe")

    def test_kaggle_canonical_url_overrides_requested_id(self):
        output = ("Your kernel title does not resolve to the specified id.\n"
                  "Kernel version 1 successfully pushed. Please check progress at "
                  "https://www.kaggle.com/code/neonhawk/debatt-ai-movie-gguf-environment-probe")
        self.assertEqual(
            parse_published_kernel(output, "neonhawk"),
            "neonhawk/debatt-ai-movie-gguf-environment-probe",
        )
        with self.assertRaises(ValueError):
            parse_published_kernel(output, "another-owner")
        with self.assertRaises(ValueError):
            parse_published_kernel(output.replace("gguf-environment-probe", "some-other-notebook"),
                                   "neonhawk")

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

    def test_failed_kaggle_run_still_collects_diagnostics(self):
        with TemporaryDirectory() as tmp:
            with patch("scripts.kaggle_gguf_probe.RESULTS", Path(tmp)):
                with patch("scripts.kaggle_gguf_probe.execute") as run:
                    run.return_value.returncode = 1
                    run.return_value.stdout = "Notebook failed; inspect gguf-probe.json"
                    self.assertFalse(retrieve_diagnostics("name/kernel"))
                    self.assertIn("Notebook failed", (Path(tmp) / "download.log").read_text())
                    run.assert_called_once()
