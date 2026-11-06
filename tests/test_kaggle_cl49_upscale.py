"""Offline CL-49 tests: 1080p upscale config, faster VAE tiles, verification."""
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import kaggle_cl47_wan21_scene as cl47
from scripts import kaggle_cl48_wan22_scene as cl48

SHA = "c" * 40


def config(**overrides):
    values = dict(sha=SHA, run_id="9-1", prompt="Close-up of a politician speaking",
                  seed=1, resolution="1280x704", num_frames=49, steps=30,
                  guidance_scale=5.0, transformer_dtype="float16",
                  profile=cl48.WAN22_TI2V_5B, upscale="1080p")
    values.update(overrides)
    return cl47.build_config(**values)


class Cl49ConfigTests(unittest.TestCase):
    def test_landscape_and_portrait_targets(self):
        up = config()["upscale"]
        self.assertEqual((up["width"], up["height"]), (1920, 1080))
        self.assertEqual(up["video_name"], "wan22-scene-1080p.mp4")
        portrait = config(resolution="704x1280")["upscale"]
        self.assertEqual((portrait["width"], portrait["height"]), (1080, 1920))

    def test_upscaler_is_pinned(self):
        up = config()["upscale"]
        self.assertRegex(up["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(up["url"].startswith("https://github.com/xinntao/Real-ESRGAN/releases/download/"))
        self.assertRegex(up["package"], r"^spandrel==\d+\.\d+\.\d+$")

    def test_none_and_invalid(self):
        self.assertIsNone(config(upscale="none")["upscale"])
        self.assertIsNone(config(upscale=None)["upscale"])
        with self.assertRaises(ValueError):
            config(upscale="4k")

    def test_cl47_default_has_no_upscale(self):
        cfg = cl47.build_config(sha=SHA, run_id="1-1", prompt="a scene", seed=1,
                                resolution="832x480", num_frames=33, steps=25,
                                guidance_scale=5.0, transformer_dtype="float16")
        self.assertIsNone(cfg["upscale"])

    def test_larger_vae_tiles_keep_overlap_and_alignment(self):
        tiles = config()["vae_tile_kwargs"]
        self.assertEqual(tiles["tile_sample_min_height"] - tiles["tile_sample_stride_height"], 64)
        for value in tiles.values():
            self.assertEqual(value % 32, 0)  # 16x VAE x 2 patch for Wan 2.2


class Cl49KernelTests(unittest.TestCase):
    def kernel(self, cfg):
        ns = {"__name__": "kernel"}
        exec(compile(cl47.make_kernel(cfg), "run.py", "exec"), ns)
        return ns

    def test_kernel_verifies_weights_and_frees_pipeline_first(self):
        source = cl47.make_kernel(config())
        self.assertIn("if digest != up[\"sha256\"]:", source)
        self.assertLess(source.index("pipe.maybe_free_model_hooks()"), source.index("report[\"upscale_result\"] = run_upscale("))
        self.assertIn("load_from_file(weights)", source)

    def test_fit_and_crop_and_upscale_frames(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("torch not installed in this environment")
        ns = self.kernel(config())
        image = torch.rand(1, 3, 704, 1280)
        self.assertEqual(tuple(ns["fit_and_crop"](image, 1920, 1080, torch).shape), (1, 3, 1080, 1920))
        small = torch.rand(1, 3, 480, 832)
        self.assertEqual(tuple(ns["fit_and_crop"](small, 1920, 1080, torch).shape), (1, 3, 1080, 1920))
        cfg = config()
        cfg["upscale"] = {**cfg["upscale"], "width": 96, "height": 54}
        double = lambda t: torch.nn.functional.interpolate(t, scale_factor=2)
        frames = np.random.default_rng(0).random((3, 32, 64, 3), dtype=np.float32)
        out = ns["upscale_frames"](frames, cfg, torch, np, double, torch.device("cpu"), torch.float32)
        self.assertEqual((len(out), out[0].shape), (3, (54, 96, 3)))

    def test_upscaled_export_keeps_exact_1080_height(self):
        source = cl47.make_kernel(config())
        self.assertIn("export_exact(upscaled, video_path", source)
        self.assertIn("macro_block_size=1", source)
        try:
            import numpy as np
            from diffusers.utils import export_to_video
        except ImportError:
            self.skipTest("diffusers/imageio not installed in this environment")
        import shutil
        if not shutil.which("ffprobe"):
            self.skipTest("ffprobe not installed")
        ns = self.kernel(config())
        frames = [np.random.default_rng(i).random((1080, 1920, 3), dtype=np.float32) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "exact.mp4")
            ns["export_exact"](frames, path, 24, export_to_video)
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                  "-show_entries", "stream=width,height", "-of", "json", path],
                                 capture_output=True, text=True, check=True).stdout
            stream = json.loads(out)["streams"][0]
            self.assertEqual((stream["width"], stream["height"]), (1920, 1080))

    def test_text_encoder_is_freed_before_video_pipeline_loads(self):
        source = cl47.make_kernel(config())
        encode = source.index("embeds = encode_prompts(local, cfg, torch, device)")
        self.assertLess(encode, source.index("pipe = WanPipeline.from_pretrained("))
        self.assertIn("local, text_encoder=None, tokenizer=None, vae=vae,", source)
        self.assertIn("del encoder, text_encoder", source)
        self.assertIn("max_sequence_length=512", source)  # WanPipeline.__call__ default
        self.assertIn('"prompt_embeds": embeds[0], "negative_prompt_embeds": embeds[1]', source)

    def test_default_negative_prompt_targets_faces(self):
        negative = config()["negative_prompt"]
        for term in ("deformed eyes", "closed eyes", "morphing face", "distorted face"):
            self.assertIn(term, negative)
        self.assertIn("watermark", negative)  # original terms kept


class Cl49VerificationTests(unittest.TestCase):
    def test_upscaled_video_probed_at_target_size(self):
        cfg = config()
        up = cfg["upscale"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / up["video_name"]
            path.write_bytes(b"0" * 5000)
            payload = {"streams": [{"codec_name": "h264", "width": 1920, "height": 1080,
                                    "nb_read_frames": "49"}]}
            runner = lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(payload))
            info = cl47.probe_video(path, {**cfg, "width": up["width"], "height": up["height"]},
                                    runner=runner)
            self.assertEqual((info["width"], info["height"], info["frames"]), (1920, 1080, 49))
            with self.assertRaises(ValueError):
                cl47.probe_video(path, cfg, runner=runner)  # base 1280x704 must not match


if __name__ == "__main__":
    unittest.main()
