"""CL-34: two bounded synthetic Euler steps, each with fresh GGUF model evaluation."""
import ast
import math
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl33_single_latent_step import (
    KERNEL_SOURCE as CL33_SOURCE, _SINGLE_STEP_BLOCK as CL33_BLOCK,
    validate_report as validate_cl33,
)

SLUG = "debatt-ai-movie-cl34-two-latent-steps"
TITLE = "Debatt AI Movie CL34 Two Latent Steps"
REPORT_NAME = "cl34-two-latent-steps.json"
RESULTS = Path("kaggle-cl34-results")
SUBMISSION = Path("kaggle-cl34-submission")

_SECOND_STEP = (
    '    # Feed the first updated latent back through the same eight real-weight\n'
    '    # video blocks. Both model timesteps remain synthetic zeros.\n'
    '    second_input = next_latent.to(dtype=latent.dtype)\n'
    '    if not torch.isfinite(second_input).all().item():\n'
    '        raise RuntimeError("Second model input has nonfinite values")\n'
    '    second_vx, second_coords, second_meta = LTXVModel._process_input(\n'
    '        root, second_input, None, None)\n'
    '    if (tuple(second_vx.shape) != (1, 4, width)\n'
    '            or tuple(second_coords.shape) != tuple(coords.shape)\n'
    '            or not torch.isfinite(second_vx).all().item()):\n'
    '        raise RuntimeError("Second model patchify output invalid")\n'
    '    second_pe = LTXBaseModel._prepare_positional_embeddings(\n'
    '        root, second_coords, frame_rate=25, x_dtype=torch.float16)\n'
    '    if (not isinstance(second_pe, tuple) or len(second_pe) != 2\n'
    '            or second_pe[0].shape != pe[0].shape\n'
    '            or not torch.isfinite(second_pe[0]).all().item()):\n'
    '        raise RuntimeError("Second model RoPE positions invalid")\n'
    '    second_ax = ax\n'
    '    second_blocks = []\n'
    '    for index in range(8):\n'
    '        context_width = specs[index]["context_width"]\n'
    '        context = torch.linspace(-0.5, 0.5, 4 * context_width,\n'
    '                                 device="cuda:0", dtype=torch.float16).reshape(1, 4, context_width)\n'
    '        timestep = torch.zeros((1, 4, 9 * width), device="cuda:0", dtype=torch.float16)\n'
    '        prompt_timestep = torch.zeros((1, 4, 2 * context_width),\n'
    '                                      device="cuda:0", dtype=torch.float16)\n'
    '        second_vx, second_ax = blocks[index](\n'
    '            (second_vx, second_ax), v_context=context, v_timestep=timestep,\n'
    '            v_prompt_timestep=prompt_timestep, v_pe=second_pe,\n'
    '            transformer_options=options)\n'
    '        torch.cuda.synchronize(0)\n'
    '        if (tuple(second_vx.shape) != (1, 4, width) or second_ax.numel() != 0\n'
    '                or not torch.isfinite(second_vx).all().item()):\n'
    '            raise RuntimeError("Second model block failed at " + str(index))\n'
    '        second_blocks.append(index)\n'
    '    second_velocity = LTXVModel._process_output(\n'
    '        root, second_vx, embedded_timestep, None, **second_meta)\n'
    '    torch.cuda.synchronize(0)\n'
    '    if (tuple(second_velocity.shape) != tuple(latent.shape)\n'
    '            or not torch.isfinite(second_velocity).all().item()):\n'
    '        raise RuntimeError("Second native latent output invalid")\n'
    '    second_sigma_to = 0.9375\n'
    '    second_delta = second_sigma_to - sigma_to\n'
    '    if not (-0.05 <= second_delta < 0):\n'
    '        raise RuntimeError("Unsafe second synthetic sigma delta")\n'
    '    final_latent = next_latent + second_delta * second_velocity.float()\n'
    '    if (tuple(final_latent.shape) != tuple(latent.shape)\n'
    '            or not torch.isfinite(final_latent).all().item()):\n'
    '        raise RuntimeError("Second latent update invalid")\n'
    '    second_change = float((final_latent - next_latent).abs().max().item())\n'
    '    second_residual = float(((final_latent - next_latent) -\n'
    '                             second_delta * second_velocity.float()).abs().max().item())\n'
    '    if not (0 < second_change < 1_000_000 and second_residual <= 1e-5):\n'
    '        raise RuntimeError("Second latent arithmetic check failed")\n'
    '    two_step_spec = {"model_evaluations": 2, "updates": 2,\n'
    '                     "second_input_from_first_update": True,\n'
    '                     "second_input_cast_to_float16": True,\n'
    '                     "second_native_patchify": True,\n'
    '                     "second_native_unpatchify": True,\n'
    '                     "second_block_indices": second_blocks,\n'
    '                     "synthetic_sigma_schedule": True,\n'
    '                     "real_ltx_scheduler": False,\n'
    '                     "real_diffusion_timestep": False,\n'
    '                     "full_denoising_loop": False,\n'
    '                     "sigma_values": [sigma_from, sigma_to, second_sigma_to],\n'
    '                     "input_shape": list(latent.shape),\n'
    '                     "final_shape": list(final_latent.shape),\n'
    '                     "max_abs_second_change": second_change,\n'
    '                     "max_abs_second_residual": second_residual,\n'
    '                     "final_finite": True,\n'
    '                     "gpu_allocated_after_bytes": int(torch.cuda.memory_allocated(0)),\n'
    '                     "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0))}\n'
)

_TWO_STEP_BLOCK = CL33_BLOCK.replace(
    '    result = {"block_indices": list(range(8)),',
    _SECOND_STEP + '    result = {"block_indices": list(range(8)),'
).replace(
    '"latent_output": output_spec, "single_step": step_spec,',
    '"latent_output": output_spec, "single_step": step_spec,\n'
    '              "two_step": two_step_spec,'
)
assert _TWO_STEP_BLOCK != CL33_BLOCK
assert _TWO_STEP_BLOCK.count('LTXVModel._process_output(') == 2
assert '"two_step": two_step_spec' in _TWO_STEP_BLOCK


def make_kernel():
    source = CL33_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL33_BLOCK) == 1
    source = source.replace(lines[0], marker + repr(child.replace(CL33_BLOCK, _TWO_STEP_BLOCK)))
    for old, new in (
        ('"comfy_gguf_single_latent_step"', '"comfy_gguf_two_latent_steps"'),
        ('"cl33_checks"', '"cl34_checks"'),
        ('"cl33_disk_free_after_bytes"', '"cl34_disk_free_after_bytes"'),
        ('"cl33_torch_after"', '"cl34_torch_after"'),
        ('"cl33_cuda_after"', '"cl34_cuda_after"'),
        ('cl33-sources', 'cl34-sources'),
        ('cl33-single-latent-step.json', REPORT_NAME),
        ('cl33_single_latent_step_complete', 'cl34_two_latent_steps_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl34_checks", {})
    first = checks.get("first_weight", {})
    model = first.get("comfy_gguf_two_latent_steps", {})
    previous = dict(report)
    prev_checks = dict(checks)
    prev_first = dict(first)
    prev_first["comfy_gguf_single_latent_step"] = model
    prev_checks["first_weight"] = prev_first
    previous["cl33_checks"] = prev_checks
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        previous["cl33_" + suffix] = report.get("cl34_" + suffix)
    validate_cl33(previous, sha, run_id)
    result = model.get("eight_video_blocks", {})
    step = result.get("two_step", {})
    if (step.get("model_evaluations") != 2 or step.get("updates") != 2
            or step.get("second_input_from_first_update") is not True
            or step.get("second_input_cast_to_float16") is not True
            or step.get("second_native_patchify") is not True
            or step.get("second_native_unpatchify") is not True
            or step.get("second_block_indices") != list(range(8))
            or step.get("synthetic_sigma_schedule") is not True
            or step.get("real_ltx_scheduler") is not False
            or step.get("real_diffusion_timestep") is not False
            or step.get("full_denoising_loop") is not False
            or step.get("sigma_values") != [1.0, 0.96875, 0.9375]
            or step.get("input_shape") != result.get("single_step", {}).get("output_shape")
            or step.get("final_shape") != step.get("input_shape")
            or step.get("final_finite") is not True
            or not isinstance(step.get("max_abs_second_change"), (int, float))
            or not math.isfinite(step["max_abs_second_change"])
            or not (0 < step["max_abs_second_change"] < 1_000_000)
            or not isinstance(step.get("max_abs_second_residual"), (int, float))
            or not math.isfinite(step["max_abs_second_residual"])
            or not (0 <= step["max_abs_second_residual"] <= 1e-5)
            or not (0 <= step.get("gpu_allocated_after_bytes", -1) <= 4 * 1024**3)
            or not (0 < step.get("gpu_peak_allocated_bytes", 0) <= 4 * 1024**3)):
        raise ValueError("CL-34 two-step inference path incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
