"""Offline CL-47 tests: no GPU, Kaggle, model download or diffusers needed."""
import ast
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import kaggle_cl47_wan21_scene as cl47

SHA = "a" * 40


def config(**overrides):
    values = dict(sha=SHA, run_id="12-1", prompt="A calm debate stage, cinematic light",
                  seed=42, resolution="832x480", num_frames=33, steps=25,
                  guidance_scale=5.0, transformer_dtype="float16")
    values.update(overrides)
    return cl47.build_config(**values)


class Cl47ConfigTests(unittest.TestCase):
    def test_slug_matches_title(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl47.TITLE.lower()).strip("-"), cl47.SLUG)

    def test_defaults(self):
        cfg = config()
        self.assertEqual((cfg["height"], cfg["width"], cfg["fps"]), (480, 832, 16))
        self.assertEqual(cfg["flow_shift"], 3.0)
        self.assertEqual(cfg["model_id"], "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertTrue(all(re.fullmatch(r"[a-z0-9-]+==[0-9.]+", p) for p in cfg["packages"]))

    def test_prompt_whitespace_is_normalised(self):
        self.assertEqual(config(prompt="  a\n\tshot  ")["prompt"], "a shot")

    def test_rejects_bad_inputs(self):
        bad = [dict(sha="abc"), dict(run_id="x"), dict(prompt="no"), dict(prompt="x" * 801),
               dict(resolution="1920x1080"), dict(num_frames=30), dict(steps=5),
               dict(steps=41), dict(guidance_scale=0), dict(transformer_dtype="bfloat16"),
               dict(seed=-1), dict(revision="main; rm -rf /")]
        for override in bad:
            with self.subTest(override=override), self.assertRaises(ValueError):
                config(**override)

    def test_frame_choices_fit_wan_temporal_compression(self):
        self.assertTrue(all((n - 1) % 4 == 0 for n in cl47.FRAME_CHOICES))
        self.assertTrue(all(h % 16 == 0 and w % 16 == 0 for h, w in cl47.RESOLUTIONS.values()))


class Cl47KernelTests(unittest.TestCase):
    def test_kernel_embeds_config_safely(self):
        tricky = 'He said "hi" \'\'\' \\ and left'
        cfg = config(prompt=tricky)
        source = cl47.make_kernel(cfg)
        tree = ast.parse(source)
        assign = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and getattr(n.targets[0], "id", None) == "CONFIG")
        embedded = json.loads(ast.literal_eval(assign.value.args[0]))
        self.assertEqual(embedded, cfg)

    def test_kernel_never_fakes_success(self):
        source = cl47.KERNEL_SOURCE
        self.assertEqual(source.count('report["video_generated"] = True'), 1)
        self.assertLess(source.index("render(pipe"), source.index('report["video_generated"] = True'))
        self.assertIn("os.remove(VIDEO)", source)
        self.assertIn("raise\n", source)
        self.assertIn("enable_model_cpu_offload()", source)
        self.assertIn('"-c", constraints', source)
        self.assertIn("metadata.version(name)", source)

    def test_frame_stats_rejects_blank_and_nan(self):
        import numpy as np
        ns = {"__name__": "kernel"}
        exec(compile(cl47.make_kernel(config()), "run.py", "exec"), ns)
        good = np.random.default_rng(0).random((5, 8, 8, 3), dtype=np.float32)
        self.assertTrue(ns["frame_stats"](good, np)["finite"])
        with self.assertRaises(RuntimeError):
            ns["frame_stats"](np.zeros((5, 8, 8, 3), np.float32), np)
        bad = good.copy()
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            ns["frame_stats"](bad, np)


class Cl47ValidationTests(unittest.TestCase):
    def good_report(self, cfg):
        return {"git_sha": SHA, "run_id": cfg["run_id"], "video_generated": True, "error": None,
                "frames": {"finite": True, "shape": [cfg["num_frames"], 480, 832, 3]}}

    def test_report_validation(self):
        cfg = config()
        cl47.validate_report(self.good_report(cfg), cfg)
        for change in ({"git_sha": "b" * 40}, {"run_id": "1-1"}, {"error": "OOM"},
                       {"video_generated": False}, {"frames": {"finite": True, "shape": [17]}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                cl47.validate_report({**self.good_report(cfg), **change}, cfg)

    def test_probe_video_checks_frames_and_size(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.mp4"
            with self.assertRaises(ValueError):
                cl47.probe_video(path, cfg)
            path.write_bytes(b"0" * 5000)

            def fake(frames, width=832, height=480, code=0):
                payload = {"streams": [{"codec_name": "h264", "width": width,
                                        "height": height, "nb_read_frames": str(frames)}]}
                return lambda *a, **k: subprocess.CompletedProcess(a, code, json.dumps(payload))

            info = cl47.probe_video(path, cfg, runner=fake(33))
            self.assertEqual((info["frames"], info["seconds"]), (33, 2.06))
            for runner in (fake(32), fake(33, width=480), fake(33, code=1)):
                with self.assertRaises(ValueError):
                    cl47.probe_video(path, cfg, runner=runner)


if __name__ == "__main__":
    unittest.main()
