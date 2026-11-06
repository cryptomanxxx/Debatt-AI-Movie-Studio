"""CL-40: three linked native video-model evaluations with LTX-formula sigmas."""
import ast
import math
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.cl38_ltx_sigma_schedule import ltxv_sigma_schedule
from scripts.kaggle_cl39_scheduled_two_steps import (
    KERNEL_SOURCE as CL39_SOURCE,
    _SCHEDULED_STEPS_BLOCK as CL39_BLOCK,
    SIGMAS as CL39_SIGMAS,
    validate_report as validate_cl39,
)

SLUG = "debatt-ai-movie-cl40-scheduled-three-steps"
TITLE = "Debatt AI Movie CL40 Scheduled Three Steps"
REPORT_NAME = "cl40-scheduled-three-steps.json"
RESULTS = Path("kaggle-cl40-results")
SUBMISSION = Path("kaggle-cl40-submission")

SIGMAS = tuple(ltxv_sigma_schedule(tokens=4, steps=20)["sigmas"][:4])
assert SIGMAS[:3] == CL39_SIGMAS
assert all(-0.05 < b - a < 0 for a, b in zip(SIGMAS, SIGMAS[1:]))

_THIRD_PASS = (
    '    # Third bounded native video pass, feeding back the second Euler update.\n'
    '    third_delta = native_sigmas[3] - native_sigmas[2]\n'
    '    if not -0.05 <= third_delta < 0:\n'
    '        raise RuntimeError("Unsafe third scheduled Euler delta")\n'
    '    third_native_input = final_native_latent.to(dtype=latent.dtype)\n'
    '    if not torch.isfinite(third_native_input).all().item():\n'
    '        raise RuntimeError("Nonfinite third native input")\n'
    '    third_vx, third_coords, third_meta = LTXVModel._process_input(\n'
    '        root, third_native_input, None, None)\n'
    '    third_pe = LTXBaseModel._prepare_positional_embeddings(\n'
    '        root, third_coords, frame_rate=25, x_dtype=torch.float16)\n'
    '    third_ts = torch.full((1, 4), native_sigmas[2],\n'
    '                          device="cuda:0", dtype=torch.float32)\n'
    '    third_prepared, third_embedded, _ = root._prepare_timestep(\n'
    '        third_ts, batch_size=1, hidden_dtype=torch.float16,\n'
    '        orig_shape=tuple(third_native_input.shape), has_spatial_mask=False)\n'
    '    if (tuple(third_vx.shape) != (1, 4, width)\n'
    '            or not isinstance(third_prepared[0], CompressedTimestep)\n'
    '            or not isinstance(third_embedded[0], CompressedTimestep)\n'
    '            or tuple(third_prepared[0].data.shape) != (1, 1, 9 * width)\n'
    '            or tuple(third_prepared[3].shape) != (1, 1, 2 * width)):\n'
    '        raise RuntimeError("Third native conditioning shape invalid")\n'
    '    third_ax = torch.empty((1, 0, 1), device="cuda:0", dtype=torch.float16)\n'
    '    third_indices = []\n'
    '    for index in range(8):\n'
    '        context_width = specs[index]["context_width"]\n'
    '        third_context = torch.linspace(-0.5, 0.5, 4 * context_width,\n'
    '                                      device="cuda:0", dtype=torch.float16).reshape(1, 4, context_width)\n'
    '        third_vx, third_ax = blocks[index](\n'
    '            (third_vx, third_ax), v_context=third_context,\n'
    '            v_timestep=third_prepared[0],\n'
    '            v_prompt_timestep=third_prepared[3],\n'
    '            v_pe=third_pe, transformer_options=options)\n'
    '        torch.cuda.synchronize(0)\n'
    '        if (tuple(third_vx.shape) != (1, 4, width)\n'
    '                or third_ax.numel() != 0\n'
    '                or not torch.isfinite(third_vx).all().item()):\n'
    '            raise RuntimeError("Third native block failed at " + str(index))\n'
    '        third_indices.append(index)\n'
    '    third_embed = third_embedded[0].data.expand(1, 4, width)\n'
    '    third_velocity = LTXVModel._process_output(\n'
    '        root, third_vx, third_embed, None, **third_meta)\n'
    '    torch.cuda.synchronize(0)\n'
    '    if (tuple(third_velocity.shape) != tuple(latent.shape)\n'
    '            or not torch.isfinite(third_velocity).all().item()):\n'
    '        raise RuntimeError("Third native velocity invalid")\n'
    '    third_native_latent = final_native_latent + third_delta * third_velocity.float()\n'
    '    if (tuple(third_native_latent.shape) != tuple(latent.shape)\n'
    '            or not torch.isfinite(third_native_latent).all().item()):\n'
    '        raise RuntimeError("Third native Euler result invalid")\n'
    '    third_change = float((third_native_latent - final_native_latent).abs().max().item())\n'
    '    third_error = float(((third_native_latent - final_native_latent) -\n'
    '                         third_delta * third_velocity.float()).abs().max().item())\n'
    '    if not (0 < third_change < 1_000_000 and 0 <= third_error <= 1e-5):\n'
    '        raise RuntimeError("Third scheduled update arithmetic invalid")\n'
    '    third_spec = {"native_conditioned_evaluations": 3,\n'
    '                  "native_conditioned_updates": 3,\n'
    '                  "third_input_from_second_update": True,\n'
    '                  "native_adaln_third_pass": True,\n'
    '                  "third_block_indices": third_indices,\n'
    '                  "scheduled_sigmas": list(native_sigmas),\n'
    '                  "scheduler_formula_reproduction": True,\n'
    '                  "official_scheduler_executed": False,\n'
    '                  "scheduler_extrapolated": True,\n'
    '                  "full_model_forward": False,\n'
    '                  "video_generated": False,\n'
    '                  "final_shape": list(third_native_latent.shape),\n'
    '                  "max_abs_third_change": third_change,\n'
    '                  "max_abs_third_residual": third_error,\n'
    '                  "finite": True,\n'
    '                  "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0))}\n'
)

assert '    linked_spec = {' in CL39_BLOCK
_THREE_STEPS_BLOCK = CL39_BLOCK.replace(
    f'    native_sigmas = {CL39_SIGMAS!r}\n',
    f'    native_sigmas = {SIGMAS!r}\n'
).replace('    linked_spec = {', _THIRD_PASS + '    linked_spec = {').replace(
    '"linked_native": linked_spec,',
    '"linked_native": linked_spec, "third_native": third_spec,'
)
assert _THREE_STEPS_BLOCK.count(_THIRD_PASS) == 1


def make_kernel():
    source = CL39_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL39_BLOCK) == 1
    source = source.replace(lines[0], marker + repr(child.replace(CL39_BLOCK, _THREE_STEPS_BLOCK)))
    for old, new in (
        ('"comfy_gguf_scheduled_two_steps"', '"comfy_gguf_scheduled_three_steps"'),
        ('"cl39_checks"', '"cl40_checks"'),
        ('"cl39_disk_free_after_bytes"', '"cl40_disk_free_after_bytes"'),
        ('"cl39_torch_after"', '"cl40_torch_after"'),
        ('"cl39_cuda_after"', '"cl40_cuda_after"'),
        ('cl39-sources', 'cl40-sources'),
        ('cl39-scheduled-two-steps.json', REPORT_NAME),
        ('cl39_scheduled_two_steps_complete', 'cl40_scheduled_three_steps_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl40_checks", {})
    first = checks.get("first_weight", {})
    model = first.get("comfy_gguf_scheduled_three_steps", {})
    eight = model.get("eight_video_blocks", {})
    linked = eight.get("linked_native", {})
    third = eight.get("third_native", {})
    if (third.get("native_conditioned_evaluations") != 3
            or third.get("native_conditioned_updates") != 3
            or third.get("third_input_from_second_update") is not True
            or third.get("native_adaln_third_pass") is not True
            or third.get("third_block_indices") != list(range(8))
            or third.get("scheduler_formula_reproduction") is not True
            or third.get("official_scheduler_executed") is not False
            or third.get("scheduler_extrapolated") is not True
            or third.get("full_model_forward") is not False
            or third.get("video_generated") is not False
            or third.get("finite") is not True
            or third.get("final_shape") != [1, 128, 1, 2, 2]
            or not isinstance(third.get("scheduled_sigmas"), list)
            or len(third["scheduled_sigmas"]) != 4
            or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                   or not math.isclose(x, expected, rel_tol=0, abs_tol=1e-7)
                   for x, expected in zip(third["scheduled_sigmas"], SIGMAS))
            or any(not isinstance(third.get(k), (int, float)) or not math.isfinite(third[k])
                   or not (0 < third[k] < 1_000_000)
                   for k in ("max_abs_third_change",))
            or not isinstance(third.get("max_abs_third_residual"), (int, float))
            or not math.isfinite(third["max_abs_third_residual"])
            or not 0 <= third["max_abs_third_residual"] <= 1e-5
            or not isinstance(third.get("gpu_peak_allocated_bytes"), int)
            or not 0 < third["gpu_peak_allocated_bytes"] <= 4 * 1024**3):
        raise ValueError("CL-40 three native scheduled steps missing/unsafe")

    # CL-39 validator expects the first three sigmas only; all other checks reused.
    previous = dict(report)
    prior_checks = dict(checks)
    prior_first = dict(first)
    prior_model = dict(model)
    prior_eight = dict(eight)
    prior_linked = dict(linked)
    prior_linked["synthetic_sigmas"] = list(CL39_SIGMAS)
    prior_eight["linked_native"] = prior_linked
    prior_model["eight_video_blocks"] = prior_eight
    prior_first["comfy_gguf_scheduled_two_steps"] = prior_model
    prior_checks["first_weight"] = prior_first
    previous["cl39_checks"] = prior_checks
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        previous["cl39_" + suffix] = report.get("cl40_" + suffix)
    validate_cl39(previous, sha, run_id)


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
