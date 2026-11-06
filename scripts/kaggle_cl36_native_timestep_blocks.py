"""CL-36: one eight-block native-ADaLN-modulated video inference probe."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl35_native_timestep_probe import (
    KERNEL_SOURCE as CL35_SOURCE, _NATIVE_TIMESTEP_BLOCK as CL35_BLOCK,
    validate_report as validate_cl35,
)

SLUG = "debatt-ai-movie-cl36-native-timestep-blocks"
TITLE = "Debatt AI Movie CL36 Native Timestep Blocks"
REPORT_NAME = "cl36-native-timestep-blocks.json"
RESULTS = Path("kaggle-cl36-results")
SUBMISSION = Path("kaggle-cl36-submission")

_NATIVE_BLOCK_PASS = (
    '    # Fresh video-only inference with native AV timestep embeddings.\n'
    '    # Sigma-like input is synthetic, not the real diffusion scheduler.\n'
    '    native_input = latent.clone()\n'
    '    native_vx, native_coords, native_meta = LTXVModel._process_input(\n'
    '        root, native_input, None, None)\n'
    '    native_pe = LTXBaseModel._prepare_positional_embeddings(\n'
    '        root, native_coords, frame_rate=25, x_dtype=torch.float16)\n'
    '    ts_input = torch.full((1, 4), 1.0, device="cuda:0", dtype=torch.float32)\n'
    '    native_prepared, native_embedded, _ = root._prepare_timestep(\n'
    '        ts_input, batch_size=1, hidden_dtype=torch.float16,\n'
    '        orig_shape=tuple(native_input.shape), has_spatial_mask=False)\n'
    '    if (len(native_prepared) != 5 or len(native_embedded) != 2\n'
    '            or not isinstance(native_prepared[0], CompressedTimestep)\n'
    '            or not isinstance(native_embedded[0], CompressedTimestep)\n'
    '            or tuple(native_vx.shape) != (1, 4, width)\n'
    '            or native_prepared[3] is None\n'
    '            or tuple(native_prepared[0].data.shape) != (1, 1, 9 * width)\n'
    '            or tuple(native_prepared[3].shape) != (1, 1, 2 * width)):\n'
    '        raise RuntimeError("Native block conditioning interface invalid")\n'
    '    native_ax = torch.empty((1, 0, 1), device="cuda:0", dtype=torch.float16)\n'
    '    native_indices = []\n'
    '    for index in range(8):\n'
    '        context_width = specs[index]["context_width"]\n'
    '        native_context = torch.linspace(-0.5, 0.5, 4 * context_width,\n'
    '                                       device="cuda:0", dtype=torch.float16).reshape(1, 4, context_width)\n'
    '        native_vx, native_ax = blocks[index](\n'
    '            (native_vx, native_ax), v_context=native_context,\n'
    '            v_timestep=native_prepared[0],\n'
    '            v_prompt_timestep=native_prepared[3], v_pe=native_pe,\n'
    '            transformer_options=options)\n'
    '        torch.cuda.synchronize(0)\n'
    '        if (tuple(native_vx.shape) != (1, 4, width) or native_ax.numel() != 0\n'
    '                or not torch.isfinite(native_vx).all().item()):\n'
    '            raise RuntimeError("Native timestep block failed at " + str(index))\n'
    '        native_indices.append(index)\n'
    '    # Native output embedding is compressed per frame: four tokens share it.\n'
    '    native_out_embed = native_embedded[0].data.expand(1, 4, width)\n'
    '    if not torch.isfinite(native_out_embed).all().item():\n'
    '        raise RuntimeError("Nonfinite native output embedding")\n'
    '    native_output = LTXVModel._process_output(\n'
    '        root, native_vx, native_out_embed, None, **native_meta)\n'
    '    torch.cuda.synchronize(0)\n'
    '    if (tuple(native_output.shape) != tuple(latent.shape)\n'
    '            or not torch.isfinite(native_output).all().item()):\n'
    '        raise RuntimeError("Native timestep output projection failed")\n'
    '    native_blocks_spec = {"native_timestep_into_blocks": True,\n'
    '                          "native_timestep_into_output": True,\n'
    '                          "native_prompt_timestep": True,\n'
    '                          "video_blocks": native_indices,\n'
    '                          "per_frame_compressed": True,\n'
    '                          "synthetic_sigma": 1.0,\n'
    '                          "real_ltx_scheduler": False,\n'
    '                          "real_text_conditioning": False,\n'
    '                          "full_model_forward": False,\n'
    '                          "video_generated": False,\n'
    '                          "input_shape": list(native_input.shape),\n'
    '                          "output_shape": list(native_output.shape),\n'
    '                          "finite": True,\n'
    '                          "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0))}\n'
)

_NATIVE_BLOCKS_BLOCK = CL35_BLOCK.replace(
    '    result = {"block_indices": list(range(8)),',
    _NATIVE_BLOCK_PASS + '    result = {"block_indices": list(range(8)),'
).replace(
    '"native_timestep": native_spec,',
    '"native_timestep": native_spec, "native_blocks": native_blocks_spec,'
)
assert _NATIVE_BLOCKS_BLOCK != CL35_BLOCK
assert '"native_blocks": native_blocks_spec' in _NATIVE_BLOCKS_BLOCK


def make_kernel():
    source = CL35_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL35_BLOCK) == 1
    source = source.replace(lines[0], marker + repr(child.replace(CL35_BLOCK, _NATIVE_BLOCKS_BLOCK)))
    for old, new in (
        ('"comfy_gguf_native_timestep"', '"comfy_gguf_native_blocks"'),
        ('"cl35_checks"', '"cl36_checks"'),
        ('"cl35_disk_free_after_bytes"', '"cl36_disk_free_after_bytes"'),
        ('"cl35_torch_after"', '"cl36_torch_after"'),
        ('"cl35_cuda_after"', '"cl36_cuda_after"'),
        ('cl35-sources', 'cl36-sources'),
        ('cl35-native-timestep-probe.json', REPORT_NAME),
        ('cl35_native_timestep_complete', 'cl36_native_blocks_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl36_checks", {})
    first = checks.get("first_weight", {})
    model = first.get("comfy_gguf_native_blocks", {})
    previous = dict(report)
    prior_checks = dict(checks)
    prior_first = dict(first)
    prior_first["comfy_gguf_native_timestep"] = model
    prior_checks["first_weight"] = prior_first
    previous["cl35_checks"] = prior_checks
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        previous["cl35_" + suffix] = report.get("cl36_" + suffix)
    validate_cl35(previous, sha, run_id)
    native = model.get("eight_video_blocks", {}).get("native_blocks", {})
    if (native.get("native_timestep_into_blocks") is not True
            or native.get("native_timestep_into_output") is not True
            or native.get("native_prompt_timestep") is not True
            or native.get("video_blocks") != list(range(8))
            or native.get("per_frame_compressed") is not True
            or native.get("synthetic_sigma") != 1.0
            or native.get("real_ltx_scheduler") is not False
            or native.get("real_text_conditioning") is not False
            or native.get("full_model_forward") is not False
            or native.get("video_generated") is not False
            or native.get("input_shape") != [1, 128, 1, 2, 2]
            or native.get("output_shape") != native.get("input_shape")
            or native.get("finite") is not True
            or not (0 < native.get("gpu_peak_allocated_bytes", 0) <= 4 * 1024**3)):
        raise ValueError("CL-36 native timestep transformer probe incomplete/unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
