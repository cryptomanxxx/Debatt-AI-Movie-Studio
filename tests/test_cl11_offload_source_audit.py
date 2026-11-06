import tempfile
import unittest
from pathlib import Path

from scripts.cl11_offload_source_audit import inspect_sources


class CL11AuditTests(unittest.TestCase):
    def test_source_matches_are_evidence_not_runtime_proof(self):
        with tempfile.TemporaryDirectory() as root:
            paths = {"comfyui": Path(root) / "comfy", "gguf": Path(root) / "gguf"}
            (paths["comfyui"] / "comfy").mkdir(parents=True)
            (paths["gguf"]).mkdir(parents=True)
            (paths["comfyui"] / "comfy/cli_args.py").write_text(
                'parser.add_argument("--lowvram")\nparser.add_argument("--cpu")\n')
            (paths["gguf"] / "nodes.py").write_text('class UnetLoaderGGUF: pass\n')
            result = inspect_sources(paths)
            self.assertTrue(result["cli_lowvram"]["present"])
            self.assertTrue(result["cli_cpu"]["present"])
            self.assertTrue(result["gguf_loader"]["present"])
            self.assertFalse(result["gguf_cpu_or_offload"]["present"])
            self.assertEqual(result["gguf_loader"]["evidence_level"], "source text only")

    def test_missing_file_is_explicit_not_positive(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            checks = inspect_sources({"comfyui": p, "gguf": p})
            self.assertTrue(all(not check["present"] for check in checks.values()))
