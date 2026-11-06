import ast
import unittest

from scripts.kaggle_cl9_startup_probe import KERNEL_SOURCE, validate_report


class CL9StartupTests(unittest.TestCase):
    def test_generated_kernel_has_bounded_local_only_startup(self):
        ast.parse(KERNEL_SOURCE.replace("__RUN_ID__", "'1-1'").replace("__GIT_SHA__", "'a'*40"))
        self.assertIn('"--listen", "127.0.0.1"', KERNEL_SOURCE)
        self.assertIn('"--disable-auto-launch"', KERNEL_SOURCE)
        self.assertIn('/object_info', KERNEL_SOURCE)
        self.assertIn('"ComfyUI-GGUF"', KERNEL_SOURCE)
        self.assertIn('"startup_probe"', KERNEL_SOURCE)
        self.assertIn('server.terminate()', KERNEL_SOURCE)
        self.assertIn('server.kill()', KERNEL_SOURCE)
        self.assertIn('shutil.rmtree(sources, ignore_errors=True)', KERNEL_SOURCE)
        self.assertIn('MIN_FREE_DISK_BYTES', KERNEL_SOURCE)
        self.assertNotIn("snapshot_download", KERNEL_SOURCE)
        self.assertNotIn("hf_hub_download", KERNEL_SOURCE)

    def test_reports_are_diagnostics_not_falsely_successful_boots(self):
        good = {"git_sha": "a" * 40, "run_id": "1-1", "install_completed": True,
                "no_deps": True, "no_cache": True,
                "torch_before": "2.11", "cl9_torch_after": "2.11",
                "cuda_before": "13", "cl9_cuda_after": "13",
                "cl9_disk_free_after_bytes": 20_000_000_000,
                "model_downloaded": False, "ltx_loaded": False, "video_generated": False,
                "comfyui_started": False,
                "cl9_checks": {"trampoline_install": {"ok": True},
                               "ComfyUI": {"ok": True}, "ComfyUI-GGUF": {"ok": True},
                               "comfy_nodes_import": {"ok": True},
                               "startup_probe": {"attempted": True, "http_ok": False,
                                                 "gguf_registered": False}}}
        validate_report(good, "a" * 40, "1-1")
        good["comfyui_started"] = True
        good["cl9_checks"]["startup_probe"].update(http_ok=True, gguf_registered=True)
        validate_report(good, "a" * 40, "1-1")
        for change in ({"model_downloaded": True}, {"cl9_torch_after": "changed"},
                       {"comfyui_started": False}, {"cl9_disk_free_after_bytes": 0}):
            with self.assertRaises(ValueError):
                validate_report({**good, **change}, "a" * 40, "1-1")
