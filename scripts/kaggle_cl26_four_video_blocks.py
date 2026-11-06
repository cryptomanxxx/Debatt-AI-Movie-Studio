"""CL-26: four consecutive real LTX AV video transformer branches, bounded."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl25_two_video_blocks import KERNEL_SOURCE as CL25_SOURCE, _TWO_BLOCKS as CL25_BLOCK

SLUG = "debatt-ai-movie-cl26-four-video-blocks"
TITLE = "Debatt AI Movie CL26 Four Video Blocks"
REPORT_NAME = "cl26-four-video-blocks.json"
RESULTS = Path("kaggle-cl26-results")
SUBMISSION = Path("kaggle-cl26-submission")

_FOUR_BLOCKS = CL25_BLOCK.replace(
    "# Actual LTX video branches of transformer blocks 0 and 1, in sequence.",
    "# Actual LTX video branches of transformer blocks 0 through 3, in sequence."
).replace(
    'if blocks is None or len(blocks) < 2:',
    'if blocks is None or len(blocks) < 4:'
).replace(
    'raise RuntimeError("Model does not have two transformer blocks")',
    'raise RuntimeError("Model does not have four transformer blocks")'
).replace(
    'for index in (0, 1):',
    'for index in range(4):'
).replace(
    'if specs[0]["video_width"] != specs[1]["video_width"]:',
    'if any(spec["video_width"] != specs[0]["video_width"] for spec in specs[1:]):'
).replace(
    '        spec["output_finite"] = True',
    '        spec["output_finite"] = True\n'
    '        spec["gpu_allocated_after_bytes"] = int(torch.cuda.memory_allocated(0))\n'
    '        spec["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))'
).replace(
    '"block_indices": [0, 1]',
    '"block_indices": [0, 1, 2, 3]'
)
assert _FOUR_BLOCKS != CL25_BLOCK
assert "for index in range(4):" in _FOUR_BLOCKS


def make_kernel():
    source = CL25_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL25_BLOCK) == 1
    child = child.replace(CL25_BLOCK, _FOUR_BLOCKS)
    assert child.count('"two_video_blocks": result,') == 1
    child = child.replace('"two_video_blocks": result,', '"four_video_blocks": result,')
    source = source.replace(lines[0], marker + repr(child))
    for old, new in (
        ('"comfy_gguf_two_blocks"', '"comfy_gguf_four_blocks"'),
        ('"cl25_checks"', '"cl26_checks"'),
        ('"cl25_disk_free_after_bytes"', '"cl26_disk_free_after_bytes"'),
        ('"cl25_torch_after"', '"cl26_torch_after"'),
        ('"cl25_cuda_after"', '"cl26_cuda_after"'),
        ('cl25-sources', 'cl26-sources'),
        ('cl25-two-video-blocks.json', REPORT_NAME),
        ('cl25_two_video_blocks_complete', 'cl26_four_video_blocks_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl26_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_four_blocks", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl26_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl26_" + suffix)
    validate_cl14(baseline, sha, run_id)
    result = model.get("four_video_blocks", {})
    blocks = result.get("blocks", [])
    if (model.get("ok") is not True or model.get("comfy_model_constructed") is not True
            or model.get("full_model_forward") is not False or model.get("video_generated") is not False
            or result.get("block_indices") != [0, 1, 2, 3]
            or result.get("video_branches_complete") is not True
            or not isinstance(blocks, list) or len(blocks) != 4
            or any(b.get("block_index") != i or b.get("output_finite") is not True
                   or not all(2 <= b.get("layer_counts", {}).get(n, 0) <= 10
                              for n in ("attn1", "attn2", "ff"))
                   or not (0 <= b.get("gpu_allocated_after_bytes", -1) <= 4 * 1024**3)
                   or not (0 < b.get("gpu_peak_allocated_bytes", 0) <= 4 * 1024**3)
                   for i, b in enumerate(blocks))
            or result.get("tokens") != 2 or result.get("finite") is not True
            or result.get("input_shape") != result.get("output_shape")
            or result.get("synthetic_input") is not True
            or result.get("synthetic_text_context") is not True
            or result.get("synthetic_timestep") is not True
            or result.get("synthetic_prompt_timestep") is not True
            or result.get("cross_attention_adaln") is not True
            or result.get("rotary_positions_applied") is not False
            or result.get("audio_branch_run") is not False
            or result.get("full_model_forward") is not False
            or len(model.get("gpu_peak_allocated_bytes", [])) != 2
            or any(v > 4 * 1024**3 for v in model["gpu_peak_allocated_bytes"])
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-26 four video blocks incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
