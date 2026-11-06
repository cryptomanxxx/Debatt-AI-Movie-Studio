"""CL-24B: retry bounded first video block with its real prompt AdaLN branch."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl24_video_block import KERNEL_SOURCE as CL24_SOURCE, _VIDEO_BLOCK as CL24_BLOCK

SLUG = "debatt-ai-movie-cl24b-prompt-adaln"
TITLE = "Debatt AI Movie CL24B Prompt AdaLN"
REPORT_NAME = "cl24b-prompt-adaln.json"
RESULTS = Path("kaggle-cl24b-results")
SUBMISSION = Path("kaggle-cl24b-submission")

# Exact ComfyUI apply_cross_attention_adaln requirements: video timestep
# has nine per-token feature groups, while v_prompt_timestep has two groups
# matching v_context's channel width. Both are synthetic here.
_PROMPT_BLOCK = CL24_BLOCK.replace(
    'if getattr(block, "cross_attention_adaln", False):\n'
    '    raise RuntimeError("Cross-attention AdaLN needs a separate bounded prompt-conditioning probe")',
    'if not getattr(block, "cross_attention_adaln", False):\n'
    '    raise RuntimeError("Expected real cross-attention AdaLN configuration")\n'
    'prompt_table = getattr(block, "prompt_scale_shift_table", None)\n'
    'if prompt_table is None or tuple(prompt_table.shape) != (2, int(block.attn2.to_k.weight.shape[1])):\n'
    '    raise RuntimeError("Unexpected prompt AdaLN parameter shape")'
).replace(
    'and 6 <= ada_count <= 9 and int(table.shape[1]) == v_width',
    'and ada_count == 9 and int(table.shape[1]) == v_width'
).replace(
    '    options = {"run_vx": True,',
    '    prompt_timestep = torch.zeros((1, 2, 2 * context_width), device="cuda:0", dtype=torch.float16)\n'
    '    options = {"run_vx": True,'
).replace(
    '                         v_pe=None, transformer_options=options)',
    '                         v_pe=None, v_prompt_timestep=prompt_timestep, transformer_options=options)'
).replace(
    '"synthetic_text_context": True, "synthetic_timestep": True,',
    '"synthetic_text_context": True, "synthetic_timestep": True,\n'
    '              "synthetic_prompt_timestep": True, "cross_attention_adaln": True,'
)
assert _PROMPT_BLOCK != CL24_BLOCK
assert "v_prompt_timestep=prompt_timestep" in _PROMPT_BLOCK


def make_kernel():
    marker = '        child_code = '
    source = CL24_SOURCE
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL24_BLOCK) == 1
    child = child.replace(CL24_BLOCK, _PROMPT_BLOCK)
    source = source.replace(lines[0], marker + repr(child))
    for old, new in (
        ('"comfy_gguf_video_block"', '"comfy_gguf_prompt_adaln"'),
        ('"cl24_checks"', '"cl24b_checks"'),
        ('"cl24_disk_free_after_bytes"', '"cl24b_disk_free_after_bytes"'),
        ('"cl24_torch_after"', '"cl24b_torch_after"'),
        ('"cl24_cuda_after"', '"cl24b_cuda_after"'),
        ('cl24-sources', 'cl24b-sources'),
        ('cl24-video-block.json', REPORT_NAME),
        ('cl24_video_block_complete', 'cl24b_prompt_adaln_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl24_video_block import validate_report as validate_cl24
    first = report.get("cl24b_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_prompt_adaln", {})
    baseline = dict(report)
    baseline["cl24_checks"] = dict(report.get("cl24b_checks", {}))
    baseline["cl24_checks"]["first_weight"] = dict(first)
    baseline["cl24_checks"]["first_weight"]["comfy_gguf_video_block"] = model
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl24_" + suffix] = report.get("cl24b_" + suffix)
    validate_cl24(baseline, sha, run_id)
    block = model.get("video_block", {})
    if (block.get("synthetic_prompt_timestep") is not True
            or block.get("cross_attention_adaln") is not True):
        raise ValueError("CL-24B did not complete real prompt AdaLN branch")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
