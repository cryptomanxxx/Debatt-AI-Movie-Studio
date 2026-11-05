import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from movie_studio.core import plan
from movie_studio.queue import build_queue, validate_queue, write_queue


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"version": 1, "scenes": [
            {"id": s.id, "prompt": s.prompt, "duration_seconds": s.duration_seconds}
            for s in plan("A ship passes overhead\nThe astronaut looks up")
        ]}

    def test_queue_is_disabled_and_deterministic(self):
        a = build_queue(self.plan, provider="ltx-huggingface-zerogpu")
        b = build_queue(self.plan, provider="ltx-huggingface-zerogpu")
        self.assertFalse(a["generation_enabled"])
        self.assertEqual(a, b)
        self.assertEqual(validate_queue(a), 2)

    def test_rejects_duplicate_scenes(self):
        self.plan["scenes"].append(self.plan["scenes"][0])
        with self.assertRaises(ValueError):
            build_queue(self.plan)

    def test_rejects_enabled_generation(self):
        q = build_queue(self.plan)
        q["generation_enabled"] = True
        with self.assertRaises(ValueError):
            validate_queue(q)

    def test_no_overwrite(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "queue.json"
            write_queue(build_queue(self.plan), target)
            with self.assertRaises(FileExistsError):
                write_queue(build_queue(self.plan), target)
            self.assertEqual(len(json.loads(target.read_text())["jobs"]), 2)


if __name__ == "__main__":
    unittest.main()
