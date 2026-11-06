"""CL-46 tests use fake API responses, never claim a real video was made."""
import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from urllib.error import HTTPError

from scripts.cl46_real_mp4_attempt import CONFIRM, _request, attempt, download_video, find_video, main, probe

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
            return {PID: {"status": {"status_str": "success", "completed": True},
                          "outputs": {"1": {"videos": [
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

    def test_status_error_overrides_existing_mp4(self):
        def client(base, endpoint, **kwargs):
            if endpoint.startswith("/history/"):
                result = self.client(base, endpoint, **kwargs)
                result[PID]["status"] = {"status_str": "error", "completed": False,
                                           "messages": [["execution_error", {"exception_message": "OOM"}]]}
                return result
            return self.client(base, endpoint, **kwargs)
        with patch("scripts.cl46_real_mp4_attempt.preflight", return_value=self.readiness):
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                attempt(self.root, self.workflow, self.root / "movie.mp4",
                        base=BASE, confirm=CONFIRM, client=client)

    def test_http_400_reports_comfyui_json_diagnostic(self):
        error = HTTPError(BASE + "/prompt", 400, "Bad Request", {},
                          io.BytesIO(b'{"node_errors":{"12":"unknown VAE"}}'))
        with patch("scripts.cl46_real_mp4_attempt.LOCAL_HTTP.open", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "unknown VAE"):
                _request(BASE, "/prompt", payload={"prompt": {}})

    def test_redirects_are_disabled(self):
        from scripts.cl46_real_mp4_attempt import NoRedirect
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "Found", {},
                                                        "https://example.com/stolen"))

    def test_rejects_fake_mp4_without_video_frames(self):
        class Response(io.BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *_):
                self.close()
        fake_mp4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
        ffprobe = Mock(returncode=1, stderr="missing moov", stdout="")
        with patch("scripts.cl46_real_mp4_attempt.LOCAL_HTTP.open",
                   return_value=Response(fake_mp4)), \
             patch("scripts.cl46_real_mp4_attempt.shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("scripts.cl46_real_mp4_attempt.subprocess.run", return_value=ffprobe):
            with self.assertRaisesRegex(ValueError, "FFprobe rejected"):
                download_video(BASE, {"filename": "broken.mp4", "subfolder": "", "type": "output"},
                               self.root / "movie.mp4")
        self.assertFalse((self.root / "movie.mp4").exists())

    def test_failing_attempt_overwrites_old_success_report(self):
        report = self.root / "attempt.json"
        report.write_text('{"status":"success","video_generated":true}')
        with patch("scripts.cl46_real_mp4_attempt.attempt", side_effect=TimeoutError("still running")):
            with self.assertRaisesRegex(TimeoutError, "still running"):
                main(["--comfy-root", str(self.root), "--workflow", str(self.workflow),
                      "--submit", "--confirm", CONFIRM, "--report", str(report)])
        state = json.loads(report.read_text())
        self.assertEqual(state["status"], "failed")
        self.assertFalse(state["video_generated"])


if __name__ == "__main__":
    unittest.main()
