"""CL-28: real LTX video rotary positions, four tokens, one bounded block."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl24b_prompt_adaln import KERNEL_SOURCE as CL24B_SOURCE, _PROMPT_BLOCK as CL24B_BLOCK

SLUG = "debatt-ai-movie-cl28-video-rope"
TITLE = "Debatt AI Movie CL28 Video RoPE"
REPORT_NAME = "cl28-video-rope.json"
RESULTS = Path("kaggle-cl28-results")
SUBMISSION = Path("kaggle-cl28-submission")

# [batch][t/y/x][token][start/end]. The target AV checkpoint sets
# use_middle_indices_grid=True, which requires a four-dimensional grid.
SYNTHETIC_COORDINATE_PAIRS = [[
    [[0, 1], [0, 1], [0, 1], [0, 1]],
    [[0, 1], [0, 1], [1, 2], [1, 2]],
    [[0, 1], [1, 2], [0, 1], [1, 2]],
]]

_ROPE_BLOCK = CL24B_BLOCK.replace(
    '# No audio path, actual RoPE, diffusion timestep embedding or full model.',
    '# No audio path, real text encoder, diffusion timestep embedding or full model.'
).replace(
    '    vx = torch.linspace(-1, 1, 2 * v_width, device="cuda:0", dtype=torch.float16).reshape(1, 2, v_width)',
    '    vx = torch.linspace(-1, 1, 4 * v_width, device="cuda:0", dtype=torch.float16).reshape(1, 4, v_width)'
).replace(
    '    context = torch.linspace(-0.5, 0.5, 2 * context_width, device="cuda:0", dtype=torch.float16).reshape(1, 2, context_width)',
    '    context = torch.linspace(-0.5, 0.5, 4 * context_width, device="cuda:0", dtype=torch.float16).reshape(1, 4, context_width)'
).replace(
    '    timestep = torch.zeros((1, 2, ada_count * v_width), device="cuda:0", dtype=torch.float16)',
    '    timestep = torch.zeros((1, 4, ada_count * v_width), device="cuda:0", dtype=torch.float16)'
).replace(
    '    prompt_timestep = torch.zeros((1, 2, 2 * context_width), device="cuda:0", dtype=torch.float16)',
    '    prompt_timestep = torch.zeros((1, 4, 2 * context_width), device="cuda:0", dtype=torch.float16)\n'
    '    # Four distinct synthetic (frame, y, x) coordinates; 25 fps for video time.\n'
    '    coords = torch.tensor(' + repr(SYNTHETIC_COORDINATE_PAIRS) + ',\n'
    '                          device="cuda:0", dtype=torch.float32)\n'
    '    if tuple(coords.shape) != (1, 3, 4, 2) or not root.use_middle_indices_grid:\n'
    '        raise RuntimeError("Unexpected LTX middle-grid RoPE coordinate contract")\n'
    '    # Explicit base video implementation: LTXAVModel overrides this method for AV pairs.\n'
    '    from comfy.ldm.lightricks.model import LTXBaseModel\n'
    '    if not isinstance(root, LTXBaseModel):\n'
    '        raise RuntimeError("Unexpected LTX root type for video RoPE")\n'
    '    pe = LTXBaseModel._prepare_positional_embeddings(\n'
    '        root, coords, frame_rate=25, x_dtype=torch.float16)\n'
    '    if not (isinstance(pe, tuple) and len(pe) == 2\n'
    '            and torch.is_tensor(pe[0]) and pe[0].shape[0:3] == (1, 4, block.attn1.heads)\n'
    '            and torch.isfinite(pe[0]).all().item()):\n'
    '        raise RuntimeError("Model RoPE construction invalid")'
).replace(
    '                         v_pe=None, v_prompt_timestep=prompt_timestep, transformer_options=options)',
    '                         v_pe=pe, v_prompt_timestep=prompt_timestep, transformer_options=options)'
).replace(
    'if (tuple(out_v.shape) != (1, 2, v_width)',
    'if (tuple(out_v.shape) != (1, 4, v_width)'
).replace(
    '"input_shape": [1, 2, v_width]',
    '"input_shape": [1, 4, v_width]'
).replace(
    '"tokens": 2, "finite": True, "synthetic_input": True,',
    '"tokens": 4, "finite": True, "synthetic_input": True,'
).replace(
    '"rotary_positions_applied": False, "audio_branch_run": False,',
    '"rotary_positions_applied": True, "coordinate_source": "synthetic_grid",\n'
    '              "frame_rate": 25, "audio_branch_run": False,'
)
assert _ROPE_BLOCK != CL24B_BLOCK
assert 'v_pe=pe' in _ROPE_BLOCK
assert '_prepare_positional_embeddings' in _ROPE_BLOCK
assert 'LTXBaseModel._prepare_positional_embeddings(' in _ROPE_BLOCK


def make_kernel():
    source = CL24B_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL24B_BLOCK) == 1
    child = child.replace(CL24B_BLOCK, _ROPE_BLOCK)
    source = source.replace(lines[0], marker + repr(child))
    for old, new in (
        ('"comfy_gguf_prompt_adaln"', '"comfy_gguf_video_rope"'),
        ('"cl24b_checks"', '"cl28_checks"'),
        ('"cl24b_disk_free_after_bytes"', '"cl28_disk_free_after_bytes"'),
        ('"cl24b_torch_after"', '"cl28_torch_after"'),
        ('"cl24b_cuda_after"', '"cl28_cuda_after"'),
        ('cl24b-sources', 'cl28-sources'),
        ('cl24b-prompt-adaln.json', REPORT_NAME),
        ('cl24b_prompt_adaln_complete', 'cl28_video_rope_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl28_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_video_rope", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl28_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl28_" + suffix)
    validate_cl14(baseline, sha, run_id)
    block = model.get("video_block", {})
    counts = block.get("layer_counts", {})
    if (model.get("ok") is not True or model.get("comfy_model_constructed") is not True
            or model.get("full_model_forward") is not False or model.get("video_generated") is not False
            or block.get("block_index") != 0 or block.get("video_branch_complete") is not True
            or block.get("submodules") != ["attn1", "attn2", "ff"]
            or not all(2 <= counts.get(n, 0) <= 10 for n in ("attn1", "attn2", "ff"))
            or block.get("tokens") != 4 or block.get("finite") is not True
            or block.get("input_shape") != block.get("output_shape")
            or block.get("input_shape", [None, None])[1] != 4
            or block.get("synthetic_input") is not True
            or block.get("synthetic_text_context") is not True
            or block.get("synthetic_timestep") is not True
            or block.get("synthetic_prompt_timestep") is not True
            or block.get("cross_attention_adaln") is not True
            or block.get("rotary_positions_applied") is not True
            or block.get("coordinate_source") != "synthetic_grid"
            or block.get("frame_rate") != 25
            or block.get("audio_branch_run") is not False
            or block.get("full_model_forward") is not False
            or len(model.get("gpu_peak_allocated_bytes", [])) != 2
            or any(v > 4 * 1024**3 for v in model["gpu_peak_allocated_bytes"])
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-28 video RoPE block incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
