"""Safety contracts for the opt-in, first-checkpoint Kaggle test."""
import ast
import unittest

from scripts import kaggle_cl13_first_weight as cl13


class FirstWeightTest(unittest.TestCase):
    def test_kaggle_title_matches_canonical_slug(self):
        # Kaggle canonicalizes notebook titles, and cl7.main checks the resulting URL.
        import re
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl13.TITLE.lower()).strip("-"),
                         cl13.SLUG)

    def test_generated_kernel_parses_and_has_one_checkpoint(self):
        source = cl13.KERNEL_SOURCE.replace("__RUN_ID__", "'123-1'").replace(
            "__GIT_SHA__", repr("a" * 40))
        ast.parse(source)
        self.assertEqual(source.count('filename = "ltx-2.5-22b-distilled-Q2_K.gguf"'), 1)
        self.assertIn("13_000_000_000", source)
        self.assertIn("4_000_000_000", source)
        self.assertIn("720", source)
        self.assertIn('Path("/kaggle/temp")', source)
        self.assertIn("signal.setitimer(signal.ITIMER_REAL, 720)", source)
        self.assertIn("signal.setitimer(signal.ITIMER_REAL, 0)", source)
        self.assertNotIn('stage = working / "cl13-q2-download.part"', source)
        self.assertIn("digest.hexdigest() == checksum", source)
        self.assertIn("stage.unlink()", source)
        self.assertIn('report["ltx_loaded"] = False', source)
        self.assertIn('report["video_generated"] = False', source)

    def test_report_requires_verified_download_and_cleanup(self):
        sha, run_id = "a" * 40, "123-1"
        report = {
            "git_sha": sha, "run_id": run_id, "install_completed": True,
            "no_deps": True, "no_cache": True,
            "direct_requirement_install_steps": [{"ok": True}] * 12,
            "torch_before": "2.9", "cl13_torch_after": "2.9",
            "cuda_before": "12.8", "cl13_cuda_after": "12.8",
            "cl13_disk_free_after_bytes": 20_000_000_000,
            "ltx_loaded": False, "video_generated": False,
            "model_downloaded": True,
            "cl13_checks": {"first_weight": {
                "target_repository": "unsloth/LTX-2.5-GGUF",
                "target_filename": "ltx-2.5-22b-distilled-Q2_K.gguf",
                "attempted": True, "sha256_verified": True,
                "download_complete": True, "stage_removed": True,
                "disk_free_after_cleanup_bytes": 20_000_000_000,
            }},
        }
        cl13.validate_report(report, sha, run_id)
        for key in ("sha256_verified", "download_complete", "stage_removed"):
            bad = {**report, "cl13_checks": {"first_weight": {
                **report["cl13_checks"]["first_weight"], key: False}}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                cl13.validate_report(bad, sha, run_id)


if __name__ == "__main__":
    unittest.main()
