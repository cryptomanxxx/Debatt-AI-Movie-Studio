"""CL-45: export patching cannot silently produce a false GPU result."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.cl43_video_runtime_preflight import ASSETS
from scripts.cl45_prepare_real_video import prepare


class Cl45Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "input").mkdir()
        for name in ("start.png", "end.png"):
            (self.root / "input" / name).touch()
        self.workflow = self.root / "api.json"
        self.graph = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wrong.safetensors"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "wrong.safetensors"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wrong.safetensors"}},
            "5": {"class_type": "LoadImage", "inputs": {"image": "old1.png"}},
            "6": {"class_type": "LoadImage", "inputs": {"image": "old2.png"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
            "8": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "movie"}},
        }
        self.kwargs = dict(model_node="1", text_node="2", video_vae_node="3",
                           audio_vae_node="4", start_node="5", end_node="6", prompt_node="7")

    def run_prepare(self):
        self.workflow.write_text(json.dumps(self.graph), encoding="utf-8")
        with patch("scripts.cl45_prepare_real_video.preflight",
                   return_value={"ready_for_manual_workflow_import": True}):
            return prepare(self.root, self.workflow, "start.png", "end.png",
                           "a short sunrise over water", **self.kwargs)

    def test_replaces_all_models_and_default_video_vae(self):
        result = self.run_prepare()
        graph = result["graph"]
        self.assertEqual(graph["3"]["inputs"]["vae_name"], ASSETS[2][2])
        self.assertEqual(graph["1"]["inputs"]["unet_name"], ASSETS[0][2])
        self.assertEqual(graph["4"]["inputs"]["vae_name"], ASSETS[3][2])
        self.assertEqual(graph["5"]["inputs"]["image"], "start.png")
        self.assertEqual(graph["7"]["inputs"]["text"], "a short sunrise over water")
        self.assertFalse(result["report"]["video_generated"])
        self.assertFalse(result["report"]["comfyui_graph_validated_by_server"])

    def test_missing_image_refuses(self):
        (self.root / "input" / "start.png").unlink()
        with self.assertRaisesRegex(ValueError, "Missing ComfyUI/input"):
            self.run_prepare()

    def test_missing_model_selector_refuses(self):
        del self.graph["3"]["inputs"]["vae_name"]
        with self.assertRaisesRegex(ValueError, "video_vae"):
            self.run_prepare()

    def test_ui_graph_is_not_api_graph(self):
        self.graph = {"nodes": []}
        with self.assertRaisesRegex(ValueError, "API format"):
            self.run_prepare()

    def test_missing_save_node_refuses(self):
        del self.graph["8"]
        with self.assertRaisesRegex(ValueError, "video-saving"):
            self.run_prepare()

    def test_image_or_latent_saver_is_not_video_output(self):
        for saver in ("SaveImage", "SaveLatent"):
            self.graph["8"]["class_type"] = saver
            with self.subTest(saver=saver), self.assertRaisesRegex(ValueError, "video-saving"):
                self.run_prepare()

    def test_flattened_comfyui_subgraph_ids_work(self):
        self.graph = {"251:" + key: value for key, value in self.graph.items()}
        self.kwargs = {key: "251:" + value for key, value in self.kwargs.items()}
        result = self.run_prepare()
        self.assertEqual(result["graph"]["251:3"]["inputs"]["vae_name"], ASSETS[2][2])

    def test_video_combine_requires_mp4_and_prefix(self):
        self.graph["8"] = {"class_type": "VHS_VideoCombine", "inputs":
                            {"filename_prefix": "first-video", "format": "video/h264-mp4"}}
        self.assertIn("VHS_VideoCombine", self.run_prepare()["report"]["video_output_nodes"])
        self.graph["8"]["inputs"]["format"] = "image/gif"
        with self.assertRaisesRegex(ValueError, "video-saving"):
            self.run_prepare()

    def test_duplicate_node_ids_refuses(self):
        self.kwargs["end_node"] = "5"
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.run_prepare()

    def test_preflight_must_pass(self):
        self.workflow.write_text(json.dumps(self.graph), encoding="utf-8")
        with patch("scripts.cl45_prepare_real_video.preflight",
                   return_value={"ready_for_manual_workflow_import": False}):
            with self.assertRaisesRegex(ValueError, "not ready"):
                prepare(self.root, self.workflow, "start.png", "end.png", "test", **self.kwargs)


if __name__ == "__main__":
    unittest.main()
