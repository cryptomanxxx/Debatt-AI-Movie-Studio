"""CL-39: two native bounded evaluations with CPU-reproduced LTX sigmas."""
import ast
import math
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.cl38_ltx_sigma_schedule import ltxv_sigma_schedule
from scripts.kaggle_cl37_native_two_steps import (
    KERNEL_SOURCE as CL37_SOURCE,
    _NATIVE_TWO_STEPS_BLOCK as CL37_BLOCK,
    validate_report as validate_cl37,
)

SLUG = "debatt-ai-movie-cl39-scheduled-two-steps"
TITLE = "Debatt AI Movie CL39 Scheduled Two Steps"
REPORT_NAME = "cl39-scheduled-two-steps.json"
RESULTS = Path("kaggle-cl39-results")
SUBMISSION = Path("kaggle-cl39-submission")

# The 4-token setting is outside ComfyUI's 1024–4096 calibration anchors.
# Only the first two updates of a 20-step schedule are evaluated.
SCHEDULE = ltxv_sigma_schedule(tokens=4, steps=20)
SIGMAS = tuple(SCHEDULE["sigmas"][:3])
assert SIGMAS[0] == 1.0 and all(-0.05 < b - a < 0 for a, b in zip(SIGMAS, SIGMAS[1:]))

_SCHEDULED_STEPS_BLOCK = CL37_BLOCK.replace(
    '    # These sigma values are synthetic: NOT a real LTX scheduler.\n'
    '    native_sigmas = (1.0, 0.96875, 0.9375)\n',
    '    # CPU reproduction of ComfyUI LTXVScheduler formula, NOT direct execution.\n'
    '    # Four tokens extrapolate below scheduler calibration anchors.\n'
    f'    native_sigmas = {SIGMAS!r}\n'
).replace(
    '                   "synthetic_sigmas": list(native_sigmas),\n'
    '                   "real_ltx_scheduler": False,',
    '                   "synthetic_sigmas": list(native_sigmas),\n'
    '                   "real_ltx_scheduler": False,\n'
    '                   "scheduler_formula_reproduction": True,\n'
    '                   "official_scheduler_executed": False,\n'
    '                   "scheduler_steps": 20,\n'
    '                   "scheduler_token_count": 4,\n'
    '                   "scheduler_extrapolated": True,'
)
assert _SCHEDULED_STEPS_BLOCK != CL37_BLOCK
assert f'    native_sigmas = {SIGMAS!r}' in _SCHEDULED_STEPS_BLOCK
assert '"scheduler_formula_reproduction": True' in _SCHEDULED_STEPS_BLOCK


def make_kernel():
    source = CL37_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL37_BLOCK) == 1
    source = source.replace(lines[0], marker + repr(child.replace(CL37_BLOCK, _SCHEDULED_STEPS_BLOCK)))
    for old, new in (
        ('"comfy_gguf_native_two_steps"', '"comfy_gguf_scheduled_two_steps"'),
        ('"cl37_checks"', '"cl39_checks"'),
        ('"cl37_disk_free_after_bytes"', '"cl39_disk_free_after_bytes"'),
        ('"cl37_torch_after"', '"cl39_torch_after"'),
        ('"cl37_cuda_after"', '"cl39_cuda_after"'),
        ('cl37-sources', 'cl39-sources'),
        ('cl37-native-two-steps.json', REPORT_NAME),
        ('cl37_native_two_steps_complete', 'cl39_scheduled_two_steps_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl39_checks", {})
    first = checks.get("first_weight", {})
    model = first.get("comfy_gguf_scheduled_two_steps", {})
    linked = model.get("eight_video_blocks", {}).get("linked_native", {})
    values = linked.get("synthetic_sigmas")
    if (not isinstance(values, list) or len(values) != 3
            or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                   or not math.isclose(x, reference, rel_tol=0, abs_tol=1e-7)
                   for x, reference in zip(values, SIGMAS))
            or linked.get("scheduler_formula_reproduction") is not True
            or linked.get("official_scheduler_executed") is not False
            or linked.get("scheduler_steps") != 20
            or linked.get("scheduler_token_count") != 4
            or linked.get("scheduler_extrapolated") is not True):
        raise ValueError("CL-39 scheduled sigma evidence missing/inconsistent")
    # Reuse all CL-37 safety checks while adapting the one expected sigma field.
    previous = dict(report)
    prior_checks = dict(checks)
    prior_first = dict(first)
    prior_model = dict(model)
    prior_eight = dict(model.get("eight_video_blocks", {}))
    prior_linked = dict(linked)
    prior_linked["synthetic_sigmas"] = [1.0, 0.96875, 0.9375]
    prior_eight["linked_native"] = prior_linked
    prior_model["eight_video_blocks"] = prior_eight
    prior_first["comfy_gguf_native_two_steps"] = prior_model
    prior_checks["first_weight"] = prior_first
    previous["cl37_checks"] = prior_checks
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        previous["cl37_" + suffix] = report.get("cl39_" + suffix)
    validate_cl37(previous, sha, run_id)


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
