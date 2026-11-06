"""CL-25: two consecutive real LTX AV transformer video branches, bounded."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl24b_prompt_adaln import KERNEL_SOURCE as CL24B_SOURCE, _PROMPT_BLOCK as CL24B_BLOCK

SLUG = "debatt-ai-movie-cl25-two-video-blocks"
TITLE = "Debatt AI Movie CL25 Two Video Blocks"
REPORT_NAME = "cl25-two-video-blocks.json"
RESULTS = Path("kaggle-cl25-results")
SUBMISSION = Path("kaggle-cl25-submission")

_TWO_BLOCKS = '''# Actual LTX video branches of transformer blocks 0 and 1, in sequence.
# Synthetic two-token video and text inputs, synthetic AdaLN conditioning;
# no RoPE, audio, video output or full model/diffusion forward.
from cl20_gguf_package.ops import GGMLOps
root = model.model.diffusion_model
blocks = getattr(root, "transformer_blocks", None)
if blocks is None or len(blocks) < 2:
    raise RuntimeError("Model does not have two transformer blocks")
specs = []
for index in (0, 1):
    block = blocks[index]
    if not getattr(block, "cross_attention_adaln", False):
        raise RuntimeError("Expected prompt AdaLN in both blocks")
    required = ("attn1", "attn2", "ff", "scale_shift_table", "prompt_scale_shift_table")
    if any(not hasattr(block, name) for name in required):
        raise RuntimeError("Missing required video block component")
    v_width = int(block.attn1.to_q.weight.shape[1])
    context_width = int(block.attn2.to_k.weight.shape[1])
    if (not (2 <= v_width <= 8192 and 2 <= context_width <= 8192)
            or tuple(block.scale_shift_table.shape) != (9, v_width)
            or tuple(block.prompt_scale_shift_table.shape) != (2, context_width)):
        raise RuntimeError("Unexpected video or AdaLN dimensions")
    counts = {}
    for name in ("attn1", "attn2", "ff"):
        linears = [(n, layer) for n, layer in getattr(block, name).named_modules()
                   if isinstance(layer, GGMLOps.Linear)]
        if not (2 <= len(linears) <= 10):
            raise RuntimeError("Unexpected GGUF projection count")
        for _, layer in linears:
            w = getattr(layer, "weight", None)
            if w is None or not hasattr(w, "tensor_type") or len(w.shape) != 2:
                raise RuntimeError("Video projection is not a GGUF matrix")
            dims = tuple(int(d) for d in w.shape)
            if min(dims) < 2 or max(dims) > 32768 or dims[0] * dims[1] > 80_000_000:
                raise RuntimeError("Projection exceeds safety cap")
        counts[name] = len(linears)
    specs.append({"block_index": index, "video_width": v_width,
                  "context_width": context_width, "layer_counts": counts})
if specs[0]["video_width"] != specs[1]["video_width"]:
    raise RuntimeError("Incompatible consecutive transformer dimensions")
with torch.inference_mode():
    width = specs[0]["video_width"]
    vx = torch.linspace(-1, 1, 2 * width, device="cuda:0", dtype=torch.float16).reshape(1, 2, width)
    ax = torch.empty((1, 0, 1), device="cuda:0", dtype=torch.float16)
    options = {"run_vx": True, "run_ax": False, "a2v_cross_attn": False,
               "v2a_cross_attn": False}
    for spec in specs:
        index = spec["block_index"]
        context_width = spec["context_width"]
        context = torch.linspace(-0.5, 0.5, 2 * context_width, device="cuda:0", dtype=torch.float16).reshape(1, 2, context_width)
        timestep = torch.zeros((1, 2, 9 * width), device="cuda:0", dtype=torch.float16)
        prompt_timestep = torch.zeros((1, 2, 2 * context_width), device="cuda:0", dtype=torch.float16)
        vx, ax = blocks[index]((vx, ax), v_context=context, v_timestep=timestep,
                               v_prompt_timestep=prompt_timestep, v_pe=None,
                               transformer_options=options)
        torch.cuda.synchronize(0)
        if tuple(vx.shape) != (1, 2, width) or ax.numel() != 0 or not torch.isfinite(vx).all().item():
            raise RuntimeError("Video block failed at index " + str(index))
        spec["output_finite"] = True
    result = {"block_indices": [0, 1], "video_branches_complete": True,
              "blocks": specs, "tokens": 2, "input_shape": [1, 2, width],
              "output_shape": list(vx.shape), "finite": True,
              "synthetic_input": True, "synthetic_text_context": True,
              "synthetic_timestep": True, "synthetic_prompt_timestep": True,
              "cross_attention_adaln": True, "rotary_positions_applied": False,
              "audio_branch_run": False, "full_model_forward": False}
'''


def make_kernel():
    source = CL24B_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL24B_BLOCK) == 1
    child = child.replace(CL24B_BLOCK, _TWO_BLOCKS)
    assert child.count('"video_block": result,') == 1
    child = child.replace('"video_block": result,', '"two_video_blocks": result,')
    source = source.replace(lines[0], marker + repr(child))
    for old, new in (
        ('"comfy_gguf_prompt_adaln"', '"comfy_gguf_two_blocks"'),
        ('"cl24b_checks"', '"cl25_checks"'),
        ('"cl24b_disk_free_after_bytes"', '"cl25_disk_free_after_bytes"'),
        ('"cl24b_torch_after"', '"cl25_torch_after"'),
        ('"cl24b_cuda_after"', '"cl25_cuda_after"'),
        ('cl24b-sources', 'cl25-sources'),
        ('cl24b-prompt-adaln.json', REPORT_NAME),
        ('cl24b_prompt_adaln_complete', 'cl25_two_video_blocks_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl25_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_two_blocks", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl25_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl25_" + suffix)
    validate_cl14(baseline, sha, run_id)
    result = model.get("two_video_blocks", {})
    blocks = result.get("blocks", [])
    if (model.get("ok") is not True or model.get("comfy_model_constructed") is not True
            or model.get("full_model_forward") is not False or model.get("video_generated") is not False
            or result.get("block_indices") != [0, 1]
            or result.get("video_branches_complete") is not True
            or not isinstance(blocks, list) or len(blocks) != 2
            or any(b.get("block_index") != i or b.get("output_finite") is not True
                   or not all(2 <= b.get("layer_counts", {}).get(n, 0) <= 10
                              for n in ("attn1", "attn2", "ff")) for i, b in enumerate(blocks))
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
            or any(v > 4 * 1024 * 1024 * 1024 for v in model["gpu_peak_allocated_bytes"])
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-25 consecutive video blocks incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
