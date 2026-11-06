"""CL-37: two linked synthetic Euler updates with native LTXAV AdaLN on both passes."""
import ast
import math
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl36_native_timestep_blocks import (
    KERNEL_SOURCE as CL36_SOURCE, _NATIVE_BLOCKS_BLOCK as CL36_BLOCK,
    validate_report as validate_cl36,
)

SLUG = "debatt-ai-movie-cl37-native-two-steps"
TITLE = "Debatt AI Movie CL37 Native Two Steps"
REPORT_NAME = "cl37-native-two-steps.json"
RESULTS = Path("kaggle-cl37-results")
SUBMISSION = Path("kaggle-cl37-submission")

_LINKED_NATIVE_STEPS = (
    '    # Two linked updates with native AdaLN on each model evaluation.\n'
    '    # These sigma values are synthetic: NOT a real LTX scheduler.\n'
    '    native_sigmas = (1.0, 0.96875, 0.9375)\n'
    '    first_delta = native_sigmas[1] - native_sigmas[0]\n'
    '    second_delta = native_sigmas[2] - native_sigmas[1]\n'
    '    if not (-0.05 <= first_delta < 0 and -0.05 <= second_delta < 0):\n'
    '        raise RuntimeError("Unsafe native-conditioning synthetic update")\n'
    '    first_native_update = latent.float() + first_delta * native_output.float()\n'
    '    if (tuple(first_native_update.shape) != tuple(latent.shape)\n'
    '            or not torch.isfinite(first_native_update).all().item()):\n'
    '        raise RuntimeError("Native-conditioned first Euler update invalid")\n'
    '    second_native_input = first_native_update.to(dtype=latent.dtype)\n'
    '    if not torch.isfinite(second_native_input).all().item():\n'
    '        raise RuntimeError("Native-conditioned feedback cast invalid")\n'
    '    linked_vx, linked_coords, linked_meta = LTXVModel._process_input(\n'
    '        root, second_native_input, None, None)\n'
    '    linked_pe = LTXBaseModel._prepare_positional_embeddings(\n'
    '        root, linked_coords, frame_rate=25, x_dtype=torch.float16)\n'
    '    linked_ts = torch.full((1, 4), native_sigmas[1],\n'
    '                           device="cuda:0", dtype=torch.float32)\n'
    '    linked_prepared, linked_embedded, _ = root._prepare_timestep(\n'
    '        linked_ts, batch_size=1, hidden_dtype=torch.float16,\n'
    '        orig_shape=tuple(second_native_input.shape), has_spatial_mask=False)\n'
    '    if (tuple(linked_vx.shape) != (1, 4, width)\n'
    '            or not isinstance(linked_prepared[0], CompressedTimestep)\n'
    '            or not isinstance(linked_embedded[0], CompressedTimestep)\n'
    '            or tuple(linked_prepared[0].data.shape) != (1, 1, 9 * width)\n'
    '            or tuple(linked_prepared[3].shape) != (1, 1, 2 * width)\n'
    '            or not torch.isfinite(linked_vx).all().item()):\n'
    '        raise RuntimeError("Native second-pass conditioning shape invalid")\n'
    '    linked_ax = torch.empty((1, 0, 1), device="cuda:0", dtype=torch.float16)\n'
    '    linked_indices = []\n'
    '    for index in range(8):\n'
    '        context_width = specs[index]["context_width"]\n'
    '        linked_context = torch.linspace(-0.5, 0.5, 4 * context_width,\n'
    '                                       device="cuda:0", dtype=torch.float16).reshape(1, 4, context_width)\n'
    '        linked_vx, linked_ax = blocks[index](\n'
    '            (linked_vx, linked_ax), v_context=linked_context,\n'
    '            v_timestep=linked_prepared[0],\n'
    '            v_prompt_timestep=linked_prepared[3],\n'
    '            v_pe=linked_pe, transformer_options=options)\n'
    '        torch.cuda.synchronize(0)\n'
    '        if (tuple(linked_vx.shape) != (1, 4, width)\n'
    '                or linked_ax.numel() != 0\n'
    '                or not torch.isfinite(linked_vx).all().item()):\n'
    '            raise RuntimeError("Native second-pass block failed at " + str(index))\n'
    '        linked_indices.append(index)\n'
    '    linked_out_embed = linked_embedded[0].data.expand(1, 4, width)\n'
    '    second_native_velocity = LTXVModel._process_output(\n'
    '        root, linked_vx, linked_out_embed, None, **linked_meta)\n'
    '    torch.cuda.synchronize(0)\n'
    '    if (tuple(second_native_velocity.shape) != tuple(latent.shape)\n'
    '            or not torch.isfinite(second_native_velocity).all().item()):\n'
    '        raise RuntimeError("Native second-pass output invalid")\n'
    '    final_native_latent = first_native_update + second_delta * second_native_velocity.float()\n'
    '    if (tuple(final_native_latent.shape) != tuple(latent.shape)\n'
    '            or not torch.isfinite(final_native_latent).all().item()):\n'
    '        raise RuntimeError("Native-conditioned second Euler update invalid")\n'
    '    first_change = float((first_native_update - latent.float()).abs().max().item())\n'
    '    second_change = float((final_native_latent - first_native_update).abs().max().item())\n'
    '    first_error = float(((first_native_update - latent.float()) -\n'
    '                         first_delta * native_output.float()).abs().max().item())\n'
    '    second_error = float(((final_native_latent - first_native_update) -\n'
    '                          second_delta * second_native_velocity.float()).abs().max().item())\n'
    '    if not (0 < first_change < 1_000_000 and 0 < second_change < 1_000_000\n'
    '            and first_error <= 1e-5 and second_error <= 1e-5):\n'
    '        raise RuntimeError("Native-conditioned update arithmetic failed")\n'
    '    linked_spec = {"native_conditioned_evaluations": 2,\n'
    '                   "native_conditioned_updates": 2,\n'
    '                   "second_input_from_first_update": True,\n'
    '                   "second_input_cast_to_float16": True,\n'
    '                   "native_adaln_both_passes": True,\n'
    '                   "native_output_both_passes": True,\n'
    '                   "second_block_indices": linked_indices,\n'
    '                   "synthetic_sigmas": list(native_sigmas),\n'
    '                   "real_ltx_scheduler": False,\n'
    '                   "full_model_forward": False,\n'
    '                   "video_generated": False,\n'
    '                   "input_shape": list(latent.shape),\n'
    '                   "final_shape": list(final_native_latent.shape),\n'
    '                   "max_abs_first_change": first_change,\n'
    '                   "max_abs_second_change": second_change,\n'
    '                   "max_abs_first_residual": first_error,\n'
    '                   "max_abs_second_residual": second_error,\n'
    '                   "finite": True,\n'
    '                   "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0))}\n'
)

_NATIVE_TWO_STEPS_BLOCK = CL36_BLOCK.replace(
    '    result = {"block_indices": list(range(8)),',
    _LINKED_NATIVE_STEPS + '    result = {"block_indices": list(range(8)),'
).replace(
    '"native_blocks": native_blocks_spec,',
    '"native_blocks": native_blocks_spec, "linked_native": linked_spec,'
)
assert _NATIVE_TWO_STEPS_BLOCK != CL36_BLOCK
assert '"linked_native": linked_spec' in _NATIVE_TWO_STEPS_BLOCK


def make_kernel():
    source = CL36_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL36_BLOCK) == 1
    source = source.replace(lines[0], marker + repr(child.replace(CL36_BLOCK, _NATIVE_TWO_STEPS_BLOCK)))
    for old, new in (
        ('"comfy_gguf_native_blocks"', '"comfy_gguf_native_two_steps"'),
        ('"cl36_checks"', '"cl37_checks"'),
        ('"cl36_disk_free_after_bytes"', '"cl37_disk_free_after_bytes"'),
        ('"cl36_torch_after"', '"cl37_torch_after"'),
        ('"cl36_cuda_after"', '"cl37_cuda_after"'),
        ('cl36-sources', 'cl37-sources'),
        ('cl36-native-timestep-blocks.json', REPORT_NAME),
        ('cl36_native_blocks_complete', 'cl37_native_two_steps_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl37_checks", {})
    first = checks.get("first_weight", {})
    model = first.get("comfy_gguf_native_two_steps", {})
    previous = dict(report)
    prior_checks = dict(checks)
    prior_first = dict(first)
    prior_first["comfy_gguf_native_blocks"] = model
    prior_checks["first_weight"] = prior_first
    previous["cl36_checks"] = prior_checks
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        previous["cl36_" + suffix] = report.get("cl37_" + suffix)
    validate_cl36(previous, sha, run_id)
    linked = model.get("eight_video_blocks", {}).get("linked_native", {})
    if (linked.get("native_conditioned_evaluations") != 2
            or linked.get("native_conditioned_updates") != 2
            or linked.get("second_input_from_first_update") is not True
            or linked.get("second_input_cast_to_float16") is not True
            or linked.get("native_adaln_both_passes") is not True
            or linked.get("native_output_both_passes") is not True
            or linked.get("second_block_indices") != list(range(8))
            or linked.get("synthetic_sigmas") != [1.0, 0.96875, 0.9375]
            or linked.get("real_ltx_scheduler") is not False
            or linked.get("full_model_forward") is not False
            or linked.get("video_generated") is not False
            or linked.get("input_shape") != [1, 128, 1, 2, 2]
            or linked.get("final_shape") != linked.get("input_shape")
            or linked.get("finite") is not True
            or any(not isinstance(linked.get(k), (int, float)) or not math.isfinite(linked[k])
                   or not (0 < linked[k] < 1_000_000)
                   for k in ("max_abs_first_change", "max_abs_second_change"))
            or any(not isinstance(linked.get(k), (int, float)) or not math.isfinite(linked[k])
                   or not (0 <= linked[k] <= 1e-5)
                   for k in ("max_abs_first_residual", "max_abs_second_residual"))
            or not (0 < linked.get("gpu_peak_allocated_bytes", 0) <= 4 * 1024**3)):
        raise ValueError("CL-37 linked native AdaLN updates incomplete/unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
