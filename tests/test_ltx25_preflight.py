import unittest

from scripts.ltx25_preflight import DISK_FREE, FILES, PACKS, RESERVE, evaluate


class PreflightTests(unittest.TestCase):
    def test_no_fit_with_large_models(self):
        sizes = {path: 10_000_000_000 for path in FILES.values()}
        report = evaluate(sizes, "abc")
        self.assertFalse(report["gpu_used"])
        for name in PACKS:
            self.assertFalse(report["packs"][name]["fits_disk"])

    def test_fit_under_disk_budget(self):
        sizes = {path: 1_000_000_000 for path in FILES.values()}
        report = evaluate(sizes, "abc")
        self.assertGreater(DISK_FREE - RESERVE, 4_000_000_000)
        for name in PACKS:
            self.assertTrue(report["packs"][name]["fits_disk"])

    def test_missing_file_is_error(self):
        with self.assertRaises(ValueError):
            evaluate({}, "abc")
