"""CPU-only CL-38 scheduler formula checks (no ComfyUI or model imports)."""
import math
import tempfile
import unittest
from pathlib import Path

from scripts.cl38_ltx_sigma_schedule import (
    build_report,
    ltxv_sigma_schedule,
    main,
    validate_report,
)


class LTXSigmaScheduleTests(unittest.TestCase):
    def test_known_anchor_shifts(self):
        self.assertAlmostEqual(ltxv_sigma_schedule(tokens=1024)["shift"], 0.95)
        self.assertAlmostEqual(ltxv_sigma_schedule(tokens=4096)["shift"], 2.05)
        self.assertLess(ltxv_sigma_schedule(tokens=4)["shift"], 0.95)

    def test_stretched_schedule_is_bounded_and_ends_with_zero(self):
        for tokens in (4, 1024, 4096):
            result = ltxv_sigma_schedule(tokens=tokens)
            values = result["sigmas"]
            self.assertEqual(len(values), 5)
            self.assertEqual(values[0], 1.0)
            self.assertAlmostEqual(values[-2], 0.1)
            self.assertEqual(values[-1], 0.0)
            self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in values))
            self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_unstretched_preserves_raw_transform(self):
        result = ltxv_sigma_schedule(tokens=4096, stretch=False)
        raw_midpoint = math.exp(2.05) / (math.exp(2.05) + 1)
        self.assertAlmostEqual(result["sigmas"][2], raw_midpoint)
        self.assertGreater(result["sigmas"][-2], 0.1)

    def test_rejects_unbounded_or_invalid_parameters(self):
        for kwargs in ({"tokens": 0}, {"tokens": 4097}, {"tokens": True},
                       {"tokens": 4, "steps": 21}, {"tokens": 4, "steps": 1},
                       {"tokens": 4, "terminal": 1},
                       {"tokens": 4, "base_shift": float("nan")},
                       {"tokens": 4, "stretch": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ltxv_sigma_schedule(**kwargs)

    def test_report_is_explicit_about_limitations(self):
        report = build_report()
        validate_report(report)
        for field in ("official_scheduler_executed", "full_model_forward",
                      "video_generated", "model_weights_downloaded", "gpu_used"):
            self.assertIs(report[field], False)
        self.assertTrue(report["four_token_shift_outside_1024_to_4096_reference_range"])
        self.assertTrue(report["sigma_schedule_depends_on_token_count"])
        with self.assertRaises(ValueError):
            validate_report({**report, "video_generated": True})

    def test_cli_writes_valid_json_without_gpu(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            main(["--output", str(path)])
            validate_report(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
