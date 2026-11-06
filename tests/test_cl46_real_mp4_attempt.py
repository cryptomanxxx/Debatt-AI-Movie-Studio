"""CL-46 tests use fake API responses, never claim a real video was made."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.cl46_real_mp4_attempt import CONFIRM, attempt, find_video, probe

BASE = "http://127.0.0.1:8188"
PID = "00000000-0000-0000-0000-000000000001"


class Cl46Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workflow = self.root / "api.json"
        self.workflow.write_text(json.dumps({"1": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "test"}}}))
        self.readiness = {"ready_for_manual_workflow_import": True}

    def client(self, base, endpoint, **kwargs):
        if endpoint == "/system_stats":
            return {"devices": [{"name": "NVIDIA GPU", "vram_total": 16_000_000_000}]}
        if endpoint == "/prompt":
            return {"prompt_id": PID}
        if endpoint == "/history/" + PID:
            return {PID: {"outputs": {"1": {"videos": [
                {"filename": "real.mp4", "subfolder": "", "type": "output"}]}}}}
        raise AssertionError(endpoint)

    def test_probe_requires_local_server_url(self):
        with self.assertRaisesRegex(ValueError, "local ComfyUI"):
            probe(self.root, "https://some-server.example", client=self.client)

    def test_probe_does_not_queue_gpu_job(self):
        with patch("scripts.cl46_real_mp4_attempt.preflight", return_value=self.readiness):
            result = probe(self.root, BASE, client=self.client)
        self.assertTrue(result["comfyui_online"])
        self.assertFalse(result["gpu_used"])

    def test_no_confirmation_never_contacts_server(self):
        with self.assertRaisesRegex(ValueError, "confirmation"):
            attempt(self.root, self.workflow, self.root / "movie.mp4",
                    base=BASE, confirm="WRONG", client=lambda *_a, **_k: self.fail("API called"))

    def test_missing_assets_never_queues(self):
        with patch("scripts.cl46_real_mp4_attempt.preflight",
                   return_value={"ready_for_manual_workflow_import": False}):
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                attempt(self.root, self.workflow, self.root / "movie.mp4",
                        base=BASE, confirm=CONFIRM, client=self.client)

    def test_mock_queue_and_download_path_is_not_real_media(self):
        downloads = []
        def fake_download(base, descriptor, destination):
            downloads.append((descriptor["filename"], destination))
            return 100
        with patch("scripts.cl46_real_mp4_attempt.preflight", return_value=self.readiness):
            result = attempt(self.root, self.workflow, self.root / "movie.mp4",
                             base=BASE, confirm=CONFIRM, client=self.client,
                             downloader=fake_download, clock=lambda: 0)
        self.assertEqual(result["prompt_id"], PID)
        self.assertEqual(downloads[0][0], "real.mp4")
        self.assertFalse((self.root / "movie.mp4").exists())

    def test_non_mp4_output_is_refused(self):
        self.assertIsNone(find_video({"outputs": {"1": {"images": [{"filename": "still.png"}]}}}))

    def test_queue_error_rejected(self):
        def client(base, endpoint, **kwargs):
            if endpoint == "/prompt":
                return {"error": "missing video VAE"}
            return self.client(base, endpoint, **kwargs)
        with patch("scripts.cl46_real_mp4_attempt.preflight", return_value=self.readiness):
            with self.assertRaisesRegex(RuntimeError, "missing video VAE"):
                attempt(self.root, self.workflow, self.root / "movie.mp4",
                        base=BASE, confirm=CONFIRM, client=client)


if __name__ == "__main__":
    unittest.main()
