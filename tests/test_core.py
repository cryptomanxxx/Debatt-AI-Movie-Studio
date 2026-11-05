import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from movie_studio.core import plan, budget, save_plan
from movie_studio.cli import assemble


class StudioTests(unittest.TestCase):
    def test_plan_and_budget(self):
        scenes = plan("First shot\n\nSecond shot", 2)
        self.assertEqual([s.id for s in scenes], ["scene-0001", "scene-0002"])
        estimate = budget(scenes, 45)
        self.assertEqual(estimate["optimistic_minimum_days"], 2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            plan(" ")
        with self.assertRaises(ValueError):
            plan("hello", 0)
        with self.assertRaises(ValueError):
            budget(plan("hello"), -1)

    def test_serialization(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "plan.json"
            save_plan(plan("A shot"), target)
            self.assertEqual(json.loads(target.read_text())["scenes"][0]["prompt"], "A shot")

    def test_assemble_rejects_missing_clips(self):
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "clips.json"
            manifest.write_text('{"clips":["missing.mp4"]}')
            with self.assertRaises(ValueError):
                assemble(manifest, Path(tmp) / "out.mp4")


if __name__ == "__main__":
    unittest.main()
