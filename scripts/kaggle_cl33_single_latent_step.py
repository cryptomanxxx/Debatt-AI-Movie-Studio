"""CL-33: one illustrative bounded Euler update after the CL-32 latent output."""
import ast
import math
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl32_latent_output import (
    KERNEL_SOURCE as CL32_SOURCE, _LATENT_OUTPUT_BLOCK as CL32_BLOCK,
    validate_report as validate_cl32,
)

SLUG = "debatt-ai-movie-cl33-single-latent-step"
TITLE = "Debatt AI Movie CL33 Single Latent Step"
REPORT_NAME = "cl33-single-latent-step.json"
RESULTS = Path("kaggle-cl33-results")
SUBMISSION = Path("kaggle-cl33-submission")

_EULER_STEP = (
    '    # Deliberately synthetic sigma pair: this is not the LTX scheduler.\n'
    '    sigma_from, sigma_to = 1.0, 0.96875\n'
    '    delta_sigma = sigma_to - sigma_from\n'
    '    if not (-0.05 <= delta_sigma < 0):\n'
    '        raise RuntimeError("Unsafe synthetic Euler step size")\n'
    '    previous = latent.float()\n'
    '    velocity = latent_output.float()\n'
    '    next_latent = previous + delta_sigma * velocity\n'
    '    if (tuple(next_latent.shape) != tuple(latent.shape)\n'
    '            or not torch.isfinite(next_latent).all().item()):\n'
    '        raise RuntimeError("Single latent step produced invalid output")\n'
    '    max_abs_change = float((next_latent - previous).abs().max().item())\n'
    '    max_abs_residual = float(((next_latent - previous) -\n'
    '                              delta_sigma * velocity).abs().max().item())\n'
    '    if not (0 < max_abs_change < 1_000_000 and max_abs_residual <= 1e-5):\n'
    '        raise RuntimeError("Single latent step arithmetic check failed")\n'
    '    step_spec = {"single_euler_update": True,\n'
    '                 "synthetic_sigma_schedule": True,\n'
    '                 "real_ltx_scheduler": False,\n'
    '                 "real_diffusion_timestep": False,\n'
    '                 "full_denoising_loop": False,\n'
    '                 "model_evaluations": 1,\n'
    '                 "input_shape": list(latent.shape),\n'
    '                 "output_shape": list(next_latent.shape),\n'
    '                 "sigma_from": sigma_from, "sigma_to": sigma_to,\n'
    '                 "max_abs_change": max_abs_change,\n'
    '                 "max_abs_residual": max_abs_residual, "finite": True,\n'
    '                 "gpu_allocated_after_bytes": int(torch.cuda.memory_allocated(0)),\n'
    '                 "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0))}\n'
)

_SINGLE_STEP_BLOCK = CL32_BLOCK.replace(
    '    result = {"block_indices": list(range(8)),',
    _EULER_STEP + '    result = {"block_indices": list(range(8)),'
).replace(
    '"blocks": specs, "latent_output": output_spec,',
    '"blocks": specs, "latent_output": output_spec, "single_step": step_spec,'
)
assert _SINGLE_STEP_BLOCK != CL32_BLOCK
assert '"single_step": step_spec' in _SINGLE_STEP_BLOCK
assert _SINGLE_STEP_BLOCK.count('LTXVModel._process_output(') == 1


def make_kernel():
    source = CL32_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL32_BLOCK) == 1
    child = child.replace(CL32_BLOCK, _SINGLE_STEP_BLOCK)
    source = source.replace(lines[0], marker + repr(child))
    for old, new in (
        ('"comfy_gguf_latent_output"', '"comfy_gguf_single_latent_step"'),
        ('"cl32_checks"', '"cl33_checks"'),
        ('"cl32_disk_free_after_bytes"', '"cl33_disk_free_after_bytes"'),
        ('"cl32_torch_after"', '"cl33_torch_after"'),
        ('"cl32_cuda_after"', '"cl33_cuda_after"'),
        ('cl32-sources', 'cl33-sources'),
        ('cl32-latent-output.json', REPORT_NAME),
        ('cl32_latent_output_complete', 'cl33_single_latent_step_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl33_checks", {})
    first = checks.get("first_weight", {})
    model = first.get("comfy_gguf_single_latent_step", {})
    old_report = dict(report)
    old_checks = dict(checks)
    old_first = dict(first)
    old_first["comfy_gguf_latent_output"] = model
    old_checks["first_weight"] = old_first
    old_report["cl32_checks"] = old_checks
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        old_report["cl32_" + suffix] = report.get("cl33_" + suffix)
    validate_cl32(old_report, sha, run_id)

    result = model.get("eight_video_blocks", {})
    step = result.get("single_step", {})
    if (step.get("single_euler_update") is not True
            or step.get("synthetic_sigma_schedule") is not True
            or step.get("real_ltx_scheduler") is not False
            or step.get("real_diffusion_timestep") is not False
            or step.get("full_denoising_loop") is not False
            or step.get("model_evaluations") != 1
            or step.get("input_shape") != result.get("latent_output", {}).get("latent_output_shape")
            or step.get("output_shape") != step.get("input_shape")
            or step.get("sigma_from") != 1.0
            or step.get("sigma_to") != 0.96875
            or step.get("finite") is not True
            or not isinstance(step.get("max_abs_change"), (float, int))
            or not math.isfinite(step["max_abs_change"])
            or not (0 < step["max_abs_change"] < 1_000_000)
            or not isinstance(step.get("max_abs_residual"), (float, int))
            or not math.isfinite(step["max_abs_residual"])
            or not (0 <= step["max_abs_residual"] <= 1e-5)
            or not (0 <= step.get("gpu_allocated_after_bytes", -1) <= 4 * 1024**3)
            or not (0 < step.get("gpu_peak_allocated_bytes", 0) <= 4 * 1024**3)):
        raise ValueError("CL-33 single latent update incomplete/unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
