import ast
import unittest

from scripts.kaggle_cl12_config_probe import KERNEL_SOURCE, validate_report


class CL12Tests(unittest.TestCase):
    def test_generated_notebook_boundaries(self):
        ast.parse(KERNEL_SOURCE.replace("__RUN_ID__", "'1-1'").replace("__GIT_SHA__", "'a'*40"))
        for flag in ("--disable-dynamic-vram", "--lowvram", "--cpu-vae",
                     "--disable-smart-memory", "--disable-auto-launch"):
            self.assertIn(flag, KERNEL_SOURCE)
        self.assertIn('"127.0.0.1"', KERNEL_SOURCE)
        self.assertIn('"configuration_probes"', KERNEL_SOURCE)
        self.assertIn('server.terminate()', KERNEL_SOURCE)
        self.assertIn('server.kill()', KERNEL_SOURCE)
        self.assertIn('shutil.rmtree(sources, ignore_errors=True)', KERNEL_SOURCE)
        self.assertIn('MIN_FREE_DISK_BYTES', KERNEL_SOURCE)
        self.assertNotIn('snapshot_download', KERNEL_SOURCE)
        self.assertNotIn('hf_hub_download', KERNEL_SOURCE)

    def test_report_distinguishes_startup_from_model_execution(self):
        probes = {name: {"attempted": True, "http_ok": True,
                         "gguf_registered": True, "model_loaded": False}
                  for name in ("default", "lowvram_no_dynamic", "cpu_vae",
                               "aggressive_ram_offload")}
        good = {"git_sha": "a" * 40, "run_id": "1-1", "install_completed": True,
                "no_deps": True, "no_cache": True,
                "torch_before": "2.11", "cl12_torch_after": "2.11",
                "cuda_before": "13", "cl12_cuda_after": "13",
                "cl12_disk_free_after_bytes": 20_000_000_000,
                "comfyui_started": True,
                "model_downloaded": False, "ltx_loaded": False, "video_generated": False,
                "cl12_checks": {"configuration_probes": probes}}
        validate_report(good, "a" * 40, "1-1")
        for change in ({"model_downloaded": True}, {"cl12_torch_after": "changed"},
                       {"cl12_checks": {}}, {"cl12_disk_free_after_bytes": 0}):
            with self.assertRaises(ValueError):
                validate_report({**good, **change}, "a" * 40, "1-1")
