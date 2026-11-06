"""CL-38: bounded CPU-only reproduction of ComfyUI LTXVScheduler mathematics.

This is not an invocation of ComfyUI's scheduler class or a denoising sampler.
Source: Comfy-Org/ComfyUI comfy_extras/nodes_lt.py, LTXVScheduler.execute.
"""
import argparse
import json
import math
from pathlib import Path

MAX_STEPS = 20
DEFAULT_BASE_SHIFT = 0.95
DEFAULT_MAX_SHIFT = 2.05
DEFAULT_TERMINAL = 0.1


def ltxv_sigma_schedule(*, tokens, steps=4, base_shift=DEFAULT_BASE_SHIFT,
                        max_shift=DEFAULT_MAX_SHIFT, stretch=True,
                        terminal=DEFAULT_TERMINAL):
    """Reproduce the official LTXVScheduler formula with Python float arithmetic.

    Unlike ComfyUI's torch.float32 vector, this reference uses Python floats;
    differences in last few digits are expected and are not a correctness claim.
    """
    if (not isinstance(tokens, int) or isinstance(tokens, bool)
            or not 1 <= tokens <= 4096 or not isinstance(steps, int)
            or isinstance(steps, bool) or not 2 <= steps <= MAX_STEPS
            or not all(isinstance(v, (float, int)) and math.isfinite(v)
                       for v in (base_shift, max_shift, terminal))
            or not 0 <= base_shift <= 100 or not 0 <= max_shift <= 100
            or not 0 <= terminal < 1 or not isinstance(stretch, bool)):
        raise ValueError("Unsafe or unsupported bounded sigma schedule parameters")

    mm = (max_shift - base_shift) / (4096 - 1024)
    shift = tokens * mm + base_shift - mm * 1024
    if not -20 <= shift <= 20:
        raise ValueError("Unsafe sigma shift for bounded exponentiation")
    exp_shift = math.exp(shift)
    values = []
    for index in range(steps + 1):
        raw = 1.0 - index / steps
        if raw == 0:
            value = 0.0
        else:
            value = exp_shift / (exp_shift + (1.0 / raw - 1.0))
        values.append(value)

    if stretch:
        # ComfyUI stretches nonzero values so that the last one is terminal.
        nonzero = values[:-1]
        scale_factor = (1.0 - nonzero[-1]) / (1.0 - terminal)
        if not (math.isfinite(scale_factor) and scale_factor > 0):
            raise ValueError("Invalid schedule stretch divisor")
        values[:-1] = [1.0 - ((1.0 - z) / scale_factor) for z in nonzero]

    if (len(values) != steps + 1 or values[0] != 1.0 or values[-1] != 0.0
            or any(not math.isfinite(v) or v < -1e-10 or v > 1 + 1e-10
                   for v in values)
            or any(not values[i] > values[i + 1] for i in range(steps))
            or (stretch and not math.isclose(values[-2], terminal, abs_tol=1e-12))):
        raise ValueError("Sigma schedule is not finite, bounded and strictly decreasing")
    return {"tokens": tokens, "steps": steps, "shift": shift,
            "sigmas": values, "stretch": stretch, "terminal": terminal}


def build_report():
    tiny = ltxv_sigma_schedule(tokens=4)
    standard = ltxv_sigma_schedule(tokens=4096)
    unstreched = ltxv_sigma_schedule(tokens=4096, stretch=False)
    return {"experiment": "cl38_ltx_sigma_schedule",
            "implementation": "cpu_python_reproduction_of_ComfyUI_LTXVScheduler_formula",
            "official_scheduler_executed": False,
            "full_model_forward": False, "video_generated": False,
            "model_weights_downloaded": False, "gpu_used": False,
            "defaults": {"base_shift": DEFAULT_BASE_SHIFT,
                         "max_shift": DEFAULT_MAX_SHIFT,
                         "terminal": DEFAULT_TERMINAL},
            "tiny_four_token": tiny, "reference_4096_token": standard,
            "unscaled_4096_token": unstreched,
            "four_token_shift_outside_1024_to_4096_reference_range": True,
            "sigma_schedule_depends_on_token_count": tiny["shift"] != standard["shift"],
            "valid": True}


def validate_report(report):
    expected = build_report()
    if report != expected:
        raise ValueError("CL-38 schedule or its evidence fields are inconsistent")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cl38-sigma-results/report.json"))
    args = parser.parse_args(argv)
    report = build_report()
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
