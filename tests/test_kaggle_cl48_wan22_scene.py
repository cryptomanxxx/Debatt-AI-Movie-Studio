"""Offline CL-48 tests: no GPU, Kaggle, model download or diffusers needed."""
import re
import unittest

from scripts import kaggle_cl47_wan21_scene as cl47
from scripts import kaggle_cl48_wan22_scene as cl48

SHA = "b" * 40
PROFILE = cl48.WAN22_TI2V_5B


def config(**overrides):
    values = dict(sha=SHA, run_id="7-1", prompt="Two politicians argue at a podium",
                  seed=1, resolution="832x480", num_frames=49, steps=30,
                  guidance_scale=5.0, transformer_dtype="float16", profile=PROFILE)
    values.update(overrides)
    return cl47.build_config(**values)


class Cl48ProfileTests(unittest.TestCase):
    def test_identity_differs_from_cl47(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", PROFILE.title.lower()).strip("-"), PROFILE.slug)
        for field in ("slug", "model_id", "stem", "results", "submission"):
            self.assertNotEqual(getattr(PROFILE, field), getattr(cl47.WAN21_1_3B, field))
        self.assertEqual(PROFILE.model_id, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

    def test_geometry_fits_wan22_vae_and_patches(self):
        self.assertTrue(all(h % 32 == 0 and w % 32 == 0 for h, w in PROFILE.resolutions.values()))
        self.assertTrue(all((n - 1) % 4 == 0 for n in PROFILE.frame_choices))

    def test_defaults_build(self):
        cfg = config()
        self.assertEqual((cfg["height"], cfg["width"], cfg["fps"]), (480, 832, 24))
        self.assertIsNone(cfg["flow_shift"])
        self.assertEqual(cfg["latent_tokens"], 13 * 15 * 26)
        self.assertEqual(cfg["report_name"], "wan22-scene.json")

    def test_token_budget(self):
        config(resolution="1280x704", num_frames=49)  # 11,440 tokens
        config(resolution="832x480", num_frames=121)  # 12,090 tokens
        with self.assertRaisesRegex(ValueError, "latent tokens"):
            config(resolution="1280x704", num_frames=73)

    def test_bounds(self):
        config(steps=50, transformer_dtype="bfloat16")
        for override in (dict(steps=51), dict(transformer_dtype="float32"),
                         dict(num_frames=33), dict(resolution="576x320")):
            with self.subTest(override=override), self.assertRaises(ValueError):
                config(**override)

    def test_cl47_profile_unchanged(self):
        cfg = cl47.build_config(sha=SHA, run_id="1-1", prompt="a scene", seed=1,
                                resolution="832x480", num_frames=81, steps=25,
                                guidance_scale=5.0, transformer_dtype="float16")
        self.assertEqual((cfg["fps"], cfg["flow_shift"], cfg["latent_tokens"]), (16, 3.0, 21 * 30 * 52))
        self.assertEqual(cfg["model_id"], "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    def test_kernel_keeps_repository_shift_when_none(self):
        source = cl47.make_kernel(config())
        self.assertIn('if cfg["flow_shift"] is not None:', source)
        self.assertIn('"model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"', source)

    def test_vae_tiling_only_above_proven_480p(self):
        self.assertFalse(config()["vae_tiling"])
        self.assertFalse(config(resolution="480x832")["vae_tiling"])
        self.assertTrue(config(resolution="1280x704")["vae_tiling"])
        self.assertTrue(config(resolution="704x1280", num_frames=25)["vae_tiling"])
        cl47_cfg = cl47.build_config(sha=SHA, run_id="1-1", prompt="a scene", seed=1,
                                     resolution="832x480", num_frames=33, steps=25,
                                     guidance_scale=5.0, transformer_dtype="float16")
        self.assertFalse(cl47_cfg["vae_tiling"])

    def test_kernel_reduces_fragmentation_before_torch_import(self):
        source = cl47.make_kernel(config(resolution="1280x704"))
        env = source.index('os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")')
        self.assertLess(env, source.index("import torch"))
        self.assertIn("pipe.vae.enable_tiling(", source)
        self.assertIn('progress("vae_decode"', source)


if __name__ == "__main__":
    unittest.main()
