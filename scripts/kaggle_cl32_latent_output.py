"""CL-32: native LTX video output projection and unpatchify after eight blocks."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl31_eight_latent_blocks import (
    KERNEL_SOURCE as CL31_SOURCE, _EIGHT_LATENT_BLOCK as CL31_BLOCK,
)

SLUG = "debatt-ai-movie-cl32-latent-output"
TITLE = "Debatt AI Movie CL32 Latent Output"
REPORT_NAME = "cl32-latent-output.json"
RESULTS = Path("kaggle-cl32-results")
SUBMISSION = Path("kaggle-cl32-submission")

_OUTPUT_STEP = (
    '    # Synthetic embedded timestep, NOT a real diffusion timestep embedding.\n'
    '    from comfy.ldm.lightricks.model import LTXVModel\n'
    '    out_projection = root.proj_out\n'
    '    out_weight = getattr(out_projection, "weight", None)\n'
    '    if (not isinstance(out_projection, GGMLOps.Linear) or out_weight is None\n'
    '            or not hasattr(out_weight, "tensor_type") or len(out_weight.shape) != 2\n'
    '            or tuple(out_weight.shape) != (channels, width)\n'
    '            or out_weight.shape[0] * out_weight.shape[1] > 80_000_000):\n'
    '        raise RuntimeError("Unsafe or non-GGUF video output projection")\n'
    '    if tuple(root.scale_shift_table.shape) != (2, width):\n'
    '        raise RuntimeError("Unexpected output AdaLN table")\n'
    '    embedded_timestep = torch.zeros((1, 4, width),\n'
    '                                     device="cuda:0", dtype=torch.float16)\n'
    '    latent_output = LTXVModel._process_output(\n'
    '        root, vx, embedded_timestep, None, **input_meta)\n'
    '    torch.cuda.synchronize(0)\n'
    '    if (tuple(latent_output.shape) != tuple(latent.shape)\n'
    '            or not torch.isfinite(latent_output).all().item()):\n'
    '        raise RuntimeError("Native LTX latent output/unpatchify failed")\n'
    '    output_spec = {"native_output_projection": True,\n'
    '                   "native_unpatchify": True, "synthetic_embedded_timestep": True,\n'
    '                   "latent_input_shape": list(latent.shape),\n'
    '                   "latent_output_shape": list(latent_output.shape),\n'
    '                   "output_finite": True,\n'
    '                   "gpu_allocated_after_bytes": int(torch.cuda.memory_allocated(0)),\n'
    '                   "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0))}\n'
)

_LATENT_OUTPUT_BLOCK = CL31_BLOCK.replace(
    '    vx, coords, _ = LTXVModel._process_input(root, latent, None, None)',
    '    vx, coords, input_meta = LTXVModel._process_input(root, latent, None, None)'
).replace(
    '    result = {"block_indices": list(range(8)),',
    _OUTPUT_STEP + '    result = {"block_indices": list(range(8)),'
).replace(
    '"blocks": specs, "tokens": 4, "input_shape": [1, 4, width],',
    '"blocks": specs, "latent_output": output_spec,\n'
    '              "tokens": 4, "input_shape": [1, 4, width],'
)
assert _LATENT_OUTPUT_BLOCK != CL31_BLOCK
assert 'LTXVModel._process_output(' in _LATENT_OUTPUT_BLOCK
assert 'latent_output = LTXVModel._process_output(' in _LATENT_OUTPUT_BLOCK
assert '"latent_output": output_spec' in _LATENT_OUTPUT_BLOCK
assert _LATENT_OUTPUT_BLOCK.count('for index in range(8):') == 1


def make_kernel():
    source = CL31_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL31_BLOCK) == 1
    child = child.replace(CL31_BLOCK, _LATENT_OUTPUT_BLOCK)
    source = source.replace(lines[0], marker + repr(child))
    for old, new in (
        ('"comfy_gguf_eight_latent_blocks"', '"comfy_gguf_latent_output"'),
        ('"cl31_checks"', '"cl32_checks"'),
        ('"cl31_disk_free_after_bytes"', '"cl32_disk_free_after_bytes"'),
        ('"cl31_torch_after"', '"cl32_torch_after"'),
        ('"cl31_cuda_after"', '"cl32_cuda_after"'),
        ('cl31-sources', 'cl32-sources'),
        ('cl31-eight-latent-blocks.json', REPORT_NAME),
        ('cl31_eight_latent_blocks_complete', 'cl32_latent_output_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl32_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_latent_output", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl32_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl32_" + suffix)
    validate_cl14(baseline, sha, run_id)
    result = model.get("eight_video_blocks", {})
    blocks = result.get("blocks", [])
    out = result.get("latent_output", {})
    if (model.get("ok") is not True or model.get("comfy_model_constructed") is not True
            or model.get("full_model_forward") is not False or model.get("video_generated") is not False
            or result.get("block_indices") != list(range(8))
            or result.get("video_branches_complete") is not True
            or not isinstance(blocks, list) or len(blocks) != 8
            or any(b.get("block_index") != i or b.get("output_finite") is not True
                   or not all(2 <= b.get("layer_counts", {}).get(n, 0) <= 10
                              for n in ("attn1", "attn2", "ff"))
                   or not (0 <= b.get("gpu_allocated_after_bytes", -1) <= 4 * 1024**3)
                   or not (0 < b.get("gpu_peak_allocated_bytes", 0) <= 4 * 1024**3)
                   for i, b in enumerate(blocks))
            or result.get("tokens") != 4 or result.get("finite") is not True
            or result.get("input_shape") != result.get("output_shape")
            or result.get("input_shape") != [1, 4, 4096]
            or result.get("synthetic_input") is not True
            or result.get("synthetic_text_context") is not True
            or result.get("synthetic_timestep") is not True
            or result.get("synthetic_prompt_timestep") is not True
            or result.get("rotary_positions_applied") is not True
            or result.get("coordinate_source") != "latent_patchifier"
            or result.get("real_patchify_projection") is not True
            or result.get("synthetic_latent") is not True
            or result.get("vae_encoded") is not False
            or result.get("real_video_input") is not False
            or result.get("frame_rate") != 25
            or result.get("audio_branch_run") is not False
            or result.get("full_model_forward") is not False
            or out.get("native_output_projection") is not True
            or out.get("native_unpatchify") is not True
            or out.get("synthetic_embedded_timestep") is not True
            or out.get("latent_input_shape") != out.get("latent_output_shape")
            or not isinstance(out.get("latent_input_shape"), list)
            or len(out["latent_input_shape"]) != 5
            or out["latent_input_shape"][0] != 1
            or out["latent_input_shape"][2:] != [1, 2, 2]
            or out.get("output_finite") is not True
            or not (0 <= out.get("gpu_allocated_after_bytes", -1) <= 4 * 1024**3)
            or not (0 < out.get("gpu_peak_allocated_bytes", 0) <= 4 * 1024**3)
            or len(model.get("gpu_peak_allocated_bytes", [])) != 2
            or any(v > 4 * 1024**3 for v in model["gpu_peak_allocated_bytes"])
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-32 latent output or upstream video blocks incomplete/unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
