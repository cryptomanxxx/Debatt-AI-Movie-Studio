"""Offline tests for the explicit CL-44 install gate."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.cl44_ltx_assets_installer import (CONFIRM, REPOSITORY,
                                                build_manifest, install, main)

REVISION = "a" * 40


class Cl44Tests(unittest.TestCase):
    def test_dry_run_never_installs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "plan.json"
            main(["--comfy-root", folder, "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(len(report["required_assets"]), 4)
            self.assertFalse(report["explicit_download_authorized"])
            self.assertFalse(report["gpu_used"])

    def test_requires_explicit_consent_and_pinned_revision(self):
        with self.assertRaises(ValueError):
            install("/does-not-matter", REVISION, "no")
        with self.assertRaises(ValueError):
            install("/does-not-matter", "main", CONFIRM)

    def _remote(self, root, *, bad_digest=False):
        root.mkdir(parents=True, exist_ok=True)
        (root / "main.py").touch()
        payload = b"x" * 1_000_000
        digest = hashlib.sha256(payload).hexdigest()
        siblings = []
        for item in build_manifest(root):
            siblings.append(SimpleNamespace(
                rfilename=item["repository_path"], size=len(payload),
                lfs={"sha256": ("0" * 64 if bad_digest else digest)}))
        return SimpleNamespace(sha=REVISION, siblings=siblings), payload

    def test_verified_existing_assets_need_no_download(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            info, payload = self._remote(root)
            for item in build_manifest(root):
                path = Path(item["destination"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            api = Mock()
            api.model_info.return_value = info
            download = Mock(side_effect=AssertionError("download forbidden"))
            with patch("scripts.cl44_ltx_assets_installer.shutil.disk_usage",
                       return_value=SimpleNamespace(free=10_000_000_001)):
                result = install(root, REVISION, CONFIRM, api=api, download=download)
            self.assertEqual(len(result["installed_and_sha256_verified_roles"]), 4)
            self.assertEqual(result["verified_total_bytes"], 4_000_000)
            self.assertFalse(result["video_generated"])
            api.model_info.assert_called_once_with(REPOSITORY, revision=REVISION, files_metadata=True)
            download.assert_not_called()

    def test_remote_metadata_must_be_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            info, _ = self._remote(root)
            info.siblings[0].lfs = None
            with self.assertRaisesRegex(ValueError, "LFS SHA-256"):
                install(root, REVISION, CONFIRM, api=Mock(model_info=Mock(return_value=info)), download=Mock())

    def test_insufficient_disk_stops_before_download(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            info, _ = self._remote(root)
            download = Mock()
            with patch("scripts.cl44_ltx_assets_installer.shutil.disk_usage",
                       return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(ValueError, "Not enough free space"):
                    install(root, REVISION, CONFIRM, api=Mock(model_info=Mock(return_value=info)), download=download)
            download.assert_not_called()

    def test_corrupted_download_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            info, _ = self._remote(root)

            def corrupt(**kwargs):
                path = root / "models" / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"z" * 1_000_000)

            with patch("scripts.cl44_ltx_assets_installer.shutil.disk_usage",
                       return_value=SimpleNamespace(free=99_000_000_000)):
                with self.assertRaisesRegex(ValueError, "SHA-256/size"):
                    install(root, REVISION, CONFIRM,
                            api=Mock(model_info=Mock(return_value=info)), download=corrupt)


if __name__ == "__main__":
    unittest.main()
