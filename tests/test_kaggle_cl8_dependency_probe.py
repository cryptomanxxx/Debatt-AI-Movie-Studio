import ast
import unittest

from scripts.kaggle_cl8_dependency_probe import KERNEL_SOURCE, validate_report


class CL8Tests(unittest.TestCase):
    def test_generated_script_and_boundaries(self):
        ast.parse(KERNEL_SOURCE.replace("__RUN_ID__", "'1-1'").replace("__GIT_SHA__", "'a'*40"))
        self.assertIn('"trampoline"', KERNEL_SOURCE)
        self.assertIn('"--only-binary=:all:"', KERNEL_SOURCE)
        self.assertIn('"--no-deps"', KERNEL_SOURCE)
        self.assertIn('"--no-cache-dir"', KERNEL_SOURCE)
        self.assertIn('"cl8_checks"', KERNEL_SOURCE)
        self.assertIn('"cl8-dependency-probe.json"', KERNEL_SOURCE)
        self.assertIn("shutil.rmtree(sources, ignore_errors=True)", KERNEL_SOURCE)
        self.assertNotIn("snapshot_download", KERNEL_SOURCE)
        self.assertNotIn("hf_hub_download", KERNEL_SOURCE)

    def test_report_fails_closed_on_safety_and_stale_fields(self):
        good = {"git_sha": "a" * 40, "run_id": "1-1", "install_completed": True,
                "no_deps": True, "no_cache": True,
                "direct_requirement_install_steps": [{"ok": True}] * 12,
                "torch_before": "2.11", "cl8_torch_after": "2.11",
                "cuda_before": "13", "cl8_cuda_after": "13",
                "cl8_disk_free_after_bytes": 20_000_000_000,
                "comfyui_started": False, "model_downloaded": False,
                "ltx_loaded": False, "video_generated": False,
                "cl8_checks": {"trampoline_install": {"ok": True},
                               "trampoline_disk_free_bytes": 20_000_000_000,
                               "pip_check": {"ok": False}, "ComfyUI": {"ok": True},
                               "ComfyUI-GGUF": {"ok": True},
                               "comfy_nodes_import": {"ok": False},
                               "template_wheel_metadata": {str(i): {"metadata_only": True}
                                                           for i in range(8)}}}
        validate_report(good, "a" * 40, "1-1")
        for change in ({"model_downloaded": True}, {"cl8_torch_after": "changed"},
                       {"cl8_checks": {}}, {"cl8_disk_free_after_bytes": 0}):
            with self.assertRaises(ValueError):
                validate_report({**good, **change}, "a" * 40, "1-1")
