"""CL-24: bounded video branch of one real LTX AV transformer block."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl23_self_attention import KERNEL_SOURCE as CL23_SOURCE, _ATTENTION as CL23_ATTENTION

SLUG = "debatt-ai-movie-cl24-video-block"
TITLE = "Debatt AI Movie CL24 Video Block"
REPORT_NAME = "cl24-video-block.json"
RESULTS = Path("kaggle-cl24-results")
SUBMISSION = Path("kaggle-cl24-submission")

_VIDEO_BLOCK = '''# Complete video path of first audio-video block, with synthetic conditioning.
# No audio path, actual RoPE, diffusion timestep embedding or full model.
from cl20_gguf_package.ops import GGMLOps
root = model.model.diffusion_model
blocks = getattr(root, "transformer_blocks", None)
if blocks is None or len(blocks) < 1:
    raise RuntimeError("No transformer blocks")
block = blocks[0]
required = ("attn1", "attn2", "ff", "scale_shift_table")
if any(not hasattr(block, name) for name in required):
    raise RuntimeError("Missing video submodule or AdaLN table")
if getattr(block, "cross_attention_adaln", False):
    raise RuntimeError("Cross-attention AdaLN needs a separate bounded prompt-conditioning probe")
v_width = int(block.attn1.to_q.weight.shape[1])
context_width = int(block.attn2.to_k.weight.shape[1])
table = block.scale_shift_table
ada_count = int(table.shape[0])
if not (2 <= v_width <= 8192 and 2 <= context_width <= 8192
        and 6 <= ada_count <= 9 and int(table.shape[1]) == v_width):
    raise RuntimeError("Unexpected video conditioning dimensions")
layer_counts = {}
for name in ("attn1", "attn2", "ff"):
    layers = [(n, layer) for n, layer in getattr(block, name).named_modules()
              if isinstance(layer, GGMLOps.Linear)]
    if not (2 <= len(layers) <= 10):
        raise RuntimeError("Unexpected video projection count: " + name)
    for _, layer in layers:
        w = getattr(layer, "weight", None)
        if w is None or not hasattr(w, "tensor_type") or len(w.shape) != 2:
            raise RuntimeError("Non-GGUF video matrix")
        dims = tuple(int(n) for n in w.shape)
        if (min(dims) < 2 or max(dims) > 32768 or dims[0] * dims[1] > 80_000_000):
            raise RuntimeError("Video matrix exceeds safety cap")
    layer_counts[name] = len(layers)
with torch.inference_mode():
    vx = torch.linspace(-1, 1, 2 * v_width, device="cuda:0", dtype=torch.float16).reshape(1, 2, v_width)
    ax = torch.empty((1, 0, 1), device="cuda:0", dtype=torch.float16)
    context = torch.linspace(-0.5, 0.5, 2 * context_width, device="cuda:0", dtype=torch.float16).reshape(1, 2, context_width)
    timestep = torch.zeros((1, 2, ada_count * v_width), device="cuda:0", dtype=torch.float16)
    options = {"run_vx": True, "run_ax": False, "a2v_cross_attn": False,
               "v2a_cross_attn": False}
    out_v, out_a = block((vx, ax), v_context=context, v_timestep=timestep,
                         v_pe=None, transformer_options=options)
    torch.cuda.synchronize(0)
    if (tuple(out_v.shape) != (1, 2, v_width)
            or out_a.numel() != 0 or not torch.isfinite(out_v).all().item()):
        raise RuntimeError("Video transformer block output invalid")
    result = {"block_index": 0, "video_branch_complete": True,
              "submodules": ["attn1", "attn2", "ff"], "layer_counts": layer_counts,
              "input_shape": [1, 2, v_width], "output_shape": list(out_v.shape),
              "tokens": 2, "finite": True, "synthetic_input": True,
              "synthetic_text_context": True, "synthetic_timestep": True,
              "rotary_positions_applied": False, "audio_branch_run": False,
              "full_model_forward": False}
'''


def make_kernel():
    source = CL23_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL23_ATTENTION) == 1
    child = child.replace(CL23_ATTENTION, _VIDEO_BLOCK)
    assert child.count('"self_attention": result,') == 1
    child = child.replace('"self_attention": result,', '"video_block": result,')
    source = source.replace(lines[0], marker + repr(child))
    assert source.count('3 * 1024 * 1024 * 1024') == 1
    source = source.replace('3 * 1024 * 1024 * 1024', '4 * 1024 * 1024 * 1024')
    for old, new in (
        ('"comfy_gguf_attention"', '"comfy_gguf_video_block"'),
        ('"cl23_checks"', '"cl24_checks"'),
        ('"cl23_disk_free_after_bytes"', '"cl24_disk_free_after_bytes"'),
        ('"cl23_torch_after"', '"cl24_torch_after"'),
        ('"cl23_cuda_after"', '"cl24_cuda_after"'),
        ('cl23-sources', 'cl24-sources'),
        ('cl23-self-attention.json', REPORT_NAME),
        ('cl23_self_attention_complete', 'cl24_video_block_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl24_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_video_block", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl24_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl24_" + suffix)
    validate_cl14(baseline, sha, run_id)
    block = model.get("video_block", {})
    counts = block.get("layer_counts", {})
    if (model.get("ok") is not True or model.get("comfy_model_constructed") is not True
            or model.get("full_model_forward") is not False
            or model.get("video_generated") is not False
            or block.get("block_index") != 0
            or block.get("video_branch_complete") is not True
            or block.get("submodules") != ["attn1", "attn2", "ff"]
            or not all(2 <= counts.get(n, 0) <= 10 for n in ("attn1", "attn2", "ff"))
            or block.get("tokens") != 2 or block.get("finite") is not True
            or block.get("synthetic_input") is not True
            or block.get("synthetic_text_context") is not True
            or block.get("synthetic_timestep") is not True
            or block.get("rotary_positions_applied") is not False
            or block.get("audio_branch_run") is not False
            or block.get("full_model_forward") is not False
            or block.get("input_shape") != block.get("output_shape")
            or len(model.get("gpu_peak_allocated_bytes", [])) != 2
            or any(v > 4 * 1024 * 1024 * 1024 for v in model["gpu_peak_allocated_bytes"])
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-24 real video transformer branch incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
