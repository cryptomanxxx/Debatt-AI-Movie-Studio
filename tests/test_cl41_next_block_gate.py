"""CL-41 CPU-only evidence and no-GPU guard tests."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.cl41_next_block_gate import build_gate, main, validate_gate


class Cl41Tests(unittest.TestCase):
    def test_verified_range_and_next_candidate(self):
        gate = build_gate()
        self.assertEqual(gate["evidence"]["proven_video_block_indices"], list(range(8)))
        self.assertEqual(gate["next_candidate_block"], 8)
        self.assertEqual(gate["decision"], "METADATA_PREFLIGHT_REQUIRED")
        self.assertIs(gate["next_block_memory_fits"], None)
        self.assertGreater(gate["evidence"]["torch_headroom_bytes"], 0)
        self.assertGreater(gate["evidence"]["sampled_rss_headroom_bytes"], 0)

    def test_claims_are_safely_limited(self):
        gate = build_gate()
        for key in ("next_candidate_weights_verified", "total_video_block_count_verified",
                    "next_block_weight_sizes_known", "next_kaggle_run_authorized",
                    "gpu_run_triggered", "model_weights_downloaded", "full_model_forward",
                    "video_generated"):
            self.assertIs(gate[key], False)
        with self.assertRaises(ValueError):
            validate_gate({**gate, "next_candidate_weights_verified": True})

    def test_json_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            main(["--output", str(path)])
            validate_gate(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
