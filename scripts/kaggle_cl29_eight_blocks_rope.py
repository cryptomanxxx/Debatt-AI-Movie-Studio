"""CL-29: eight real-weight LTX video blocks with shared real rotary positions."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl27_eight_video_blocks import KERNEL_SOURCE as CL27_SOURCE, _EIGHT_BLOCKS as CL27_BLOCK
from scripts.kaggle_cl28_video_rope import SYNTHETIC_COORDINATE_PAIRS

SLUG = "debatt-ai-movie-cl29-eight-blocks-rope"
TITLE = "Debatt AI Movie CL29 Eight Blocks RoPE"
REPORT_NAME = "cl29-eight-blocks-rope.json"
RESULTS = Path("kaggle-cl29-results")
SUBMISSION = Path("kaggle-cl29-submission")

_COORDINATE_SETUP = (
    '    # Target LTX-2.5 AV checkpoint uses middle-grid start/end positions.\n'
    '    from comfy.ldm.lightricks.model import LTXBaseModel\n'
    '    if not isinstance(root, LTXBaseModel) or not root.use_middle_indices_grid:\n'
    '        raise RuntimeError("CL29 requires LTX video middle-grid RoPE")\n'
    '    coords = torch.tensor(' + repr(SYNTHETIC_COORDINATE_PAIRS) + ',\n'
    '                          device="cuda:0", dtype=torch.float32)\n'
    '    if tuple(coords.shape) != (1, 3, 4, 2):\n'
    '        raise RuntimeError("Invalid four-token middle-grid coordinates")\n'
    '    # Bypass LTXAVModel override, which expects video/audio coordinate pairs.\n'
    '    pe = LTXBaseModel._prepare_positional_embeddings(\n'
    '        root, coords, frame_rate=25, x_dtype=torch.float16)\n'
    '    if not (isinstance(pe, tuple) and len(pe) == 2\n'
    '            and torch.is_tensor(pe[0])\n'
    '            and pe[0].shape[0:3] == (1, 4, blocks[0].attn1.heads)\n'
    '            and torch.isfinite(pe[0]).all().item()):\n'
    '        raise RuntimeError("Invalid video RoPE embeddings")\n'
)

_EIGHT_ROPE_BLOCK = CL27_BLOCK.replace(
    '# Synthetic two-token video and text inputs, synthetic AdaLN conditioning;\n'
    '# no RoPE, audio, video output or full model/diffusion forward.',
    '# Synthetic four-token video/text inputs, synthetic AdaLN conditioning;\n'
    '# real video RoPE; no audio, video output or full model/diffusion forward.'
).replace(
    '    vx = torch.linspace(-1, 1, 2 * width, device="cuda:0", dtype=torch.float16).reshape(1, 2, width)',
    '    vx = torch.linspace(-1, 1, 4 * width, device="cuda:0", dtype=torch.float16).reshape(1, 4, width)'
).replace(
    '    options = {"run_vx": True,',
    _COORDINATE_SETUP + '    options = {"run_vx": True,'
).replace(
    '2 * context_width, device="cuda:0", dtype=torch.float16).reshape(1, 2, context_width)',
    '4 * context_width, device="cuda:0", dtype=torch.float16).reshape(1, 4, context_width)'
).replace(
    'torch.zeros((1, 2, 9 * width)',
    'torch.zeros((1, 4, 9 * width)'
).replace(
    'torch.zeros((1, 2, 2 * context_width)',
    'torch.zeros((1, 4, 2 * context_width)'
).replace(
    'v_prompt_timestep=prompt_timestep, v_pe=None,',
    'v_prompt_timestep=prompt_timestep, v_pe=pe,'
).replace(
    'if tuple(vx.shape) != (1, 2, width)',
    'if tuple(vx.shape) != (1, 4, width)'
).replace(
    '"tokens": 2, "input_shape": [1, 2, width]',
    '"tokens": 4, "input_shape": [1, 4, width]'
).replace(
    '"cross_attention_adaln": True, "rotary_positions_applied": False,',
    '"cross_attention_adaln": True, "rotary_positions_applied": True,\n'
    '              "coordinate_source": "synthetic_middle_grid", "frame_rate": 25,'
)
assert _EIGHT_ROPE_BLOCK != CL27_BLOCK
assert 'for index in range(8):' in _EIGHT_ROPE_BLOCK
assert 'v_pe=pe' in _EIGHT_ROPE_BLOCK
assert 'v_pe=None' not in _EIGHT_ROPE_BLOCK
assert _EIGHT_ROPE_BLOCK.count('LTXBaseModel._prepare_positional_embeddings(') == 1


def make_kernel():
    source = CL27_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL27_BLOCK) == 1
    child = child.replace(CL27_BLOCK, _EIGHT_ROPE_BLOCK)
    source = source.replace(lines[0], marker + repr(child))
    for old, new in (
        ('"comfy_gguf_eight_blocks"', '"comfy_gguf_eight_rope_blocks"'),
        ('"cl27_checks"', '"cl29_checks"'),
        ('"cl27_disk_free_after_bytes"', '"cl29_disk_free_after_bytes"'),
        ('"cl27_torch_after"', '"cl29_torch_after"'),
        ('"cl27_cuda_after"', '"cl29_cuda_after"'),
        ('cl27-sources', 'cl29-sources'),
        ('cl27-eight-video-blocks.json', REPORT_NAME),
        ('cl27_eight_video_blocks_complete', 'cl29_eight_rope_blocks_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl29_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_eight_rope_blocks", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl29_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl29_" + suffix)
    validate_cl14(baseline, sha, run_id)
    result = model.get("eight_video_blocks", {})
    blocks = result.get("blocks", [])
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
            or not isinstance(result.get("input_shape"), list)
            or len(result["input_shape"]) != 3 or result["input_shape"][:2] != [1, 4]
            or result.get("synthetic_input") is not True
            or result.get("synthetic_text_context") is not True
            or result.get("synthetic_timestep") is not True
            or result.get("synthetic_prompt_timestep") is not True
            or result.get("cross_attention_adaln") is not True
            or result.get("rotary_positions_applied") is not True
            or result.get("coordinate_source") != "synthetic_middle_grid"
            or result.get("frame_rate") != 25
            or result.get("audio_branch_run") is not False
            or result.get("full_model_forward") is not False
            or len(model.get("gpu_peak_allocated_bytes", [])) != 2
            or any(v > 4 * 1024**3 for v in model["gpu_peak_allocated_bytes"])
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-29 eight RoPE video blocks incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
