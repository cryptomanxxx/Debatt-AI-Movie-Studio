import unittest

from scripts.ltx25_cl10_memory_budget import analyze


class CL10BudgetTests(unittest.TestCase):
    def fixture(self, transformer=8_000_000_000):
        return {"repositories": {"transformer": {"revision": "rev"}},
                "files": {
                    "transformer_q2": {"bytes": transformer},
                    "transformer_ud_q2": {"bytes": 9_000_000_000},
                    "encoder_q2": {"bytes": 5_000_000_000},
                    "video_vae": {"bytes": 1_000_000_000},
                    "audio_vae": {"bytes": 500_000_000},
                }}

    def test_does_not_pool_two_gpu_memories(self):
        result = analyze(self.fixture(), free_gpu=(10_000_000_000, 10_000_000_000),
                         free_disk=20_000_000_000)
        q2 = result["packs"]["distilled_q2"]
        self.assertEqual(q2["total_file_bytes"], 14_500_000_000)
        self.assertEqual(q2["single_gpu_file_proxy_margins_bytes"],
                         [-4_500_000_000, -4_500_000_000])
        self.assertTrue(q2["hypothetical_two_gpu_placement"]["both_file_proxy_margins_positive"])
        self.assertFalse(q2["hypothetical_two_gpu_placement"]["backend_support_verified"])
        self.assertEqual(q2["runtime_vram_feasibility"], "unknown")
        self.assertFalse(result["model_downloaded"])
        self.assertFalse(result["gpu_used"])

    def test_low_disk_does_not_claim_fit(self):
        result = analyze(self.fixture(), free_disk=16_000_000_000)
        self.assertFalse(result["packs"]["distilled_q2"]["disk_weights_only_fit"])
        self.assertLess(result["packs"]["distilled_q2"]["disk_bytes_after_weights_and_reserve"], 0)

    def test_missing_or_invalid_metadata_fails_closed(self):
        for value in [0, -1, 1.5, None]:
            with self.assertRaises(ValueError):
                analyze(self.fixture(transformer=value))
        with self.assertRaises(ValueError):
            analyze(self.fixture(), free_gpu=(30_000_000_000,))
