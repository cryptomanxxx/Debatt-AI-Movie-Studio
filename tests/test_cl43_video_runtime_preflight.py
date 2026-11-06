"""CL-43 preflight tests; never execute ComfyUI or download models."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.cl43_video_runtime_preflight import ASSETS, WORKFLOW, preflight


class Cl43Tests(unittest.TestCase):
    def test_missing_assets_prevent_false_success(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("scripts.cl43_video_runtime_preflight.shutil.which", return_value=None):
                result = preflight(folder)
        self.assertEqual(result["workflow"], WORKFLOW)
        self.assertFalse(result["all_required_assets_present"])
        self.assertFalse(result["ready_for_manual_workflow_import"])
        self.assertFalse(result["video_generated"])
        self.assertFalse(result["gpu_used"])
        self.assertFalse(result["automatic_downloads"])

    def test_official_lower_vram_assets_are_distinct_from_gguf(self):
        roles = [x[0] for x in ASSETS]
        self.assertEqual(roles, ["transformer", "text_encoder", "video_vae", "audio_vae"])
        self.assertTrue(all(name.endswith(".safetensors") for _, _, name in ASSETS))

    def test_ready_only_if_every_prerequisite_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "main.py").touch()
            (root / "custom_nodes" / "ComfyUI-LTXVideo").mkdir(parents=True)
            for _, subdir, filename in ASSETS:
                p = root / "models" / subdir / filename
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            with patch("scripts.cl43_video_runtime_preflight.shutil.which", return_value="/usr/bin/ffmpeg"), \
                 patch("scripts.cl43_video_runtime_preflight.shutil.disk_usage", return_value=SimpleNamespace(free=50_000_000_000)):
                result = preflight(root)
            self.assertTrue(result["ready_for_manual_workflow_import"])
            self.assertFalse(result["gpu_memory_fit_verified"])
            self.assertFalse(result["model_hashes_verified"])
            self.assertEqual(result["video_vae_template_override"], "ltx-2.5-video-vae-conv-bf16.safetensors")


if __name__ == "__main__":
    unittest.main()
