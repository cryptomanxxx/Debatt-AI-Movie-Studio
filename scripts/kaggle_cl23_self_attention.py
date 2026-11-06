"""CL-23: bounded genuine LTX video self-attention with two synthetic tokens."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl22_transformer_ff import KERNEL_SOURCE as CL22_SOURCE, _FEED_FORWARD as CL22_FF

SLUG = "debatt-ai-movie-cl23-self-attention"
TITLE = "Debatt AI Movie CL23 Self Attention"
REPORT_NAME = "cl23-self-attention.json"
RESULTS = Path("kaggle-cl23-results")
SUBMISSION = Path("kaggle-cl23-submission")

_ATTENTION = '''# Actual first LTX video self-attention using two synthetic tokens.
# PE=None deliberately bypasses rotary positional embedding; no full block,
# conditioning, diffusion forward or video generation occurs.
from cl20_gguf_package.ops import GGMLOps
root = model.model.diffusion_model
blocks = getattr(root, "transformer_blocks", None)
if blocks is None or len(blocks) < 1:
    raise RuntimeError("No transformer blocks in LTX model")
attention = getattr(blocks[0], "attn1", None)
if attention is None:
    raise RuntimeError("Missing LTX video self-attention submodule")
linears = [(name, layer) for name, layer in attention.named_modules()
           if isinstance(layer, GGMLOps.Linear)]
if not (4 <= len(linears) <= 10):
    raise RuntimeError("Unexpected self-attention linear count")
layers = []
for name, layer in linears:
    w = getattr(layer, "weight", None)
    if w is None or not hasattr(w, "tensor_type") or len(w.shape) != 2:
        raise RuntimeError("Attention weight is not a GGUF matrix")
    dims = tuple(int(n) for n in w.shape)
    if (min(dims) < 2 or max(dims) > 32768
            or dims[0] * dims[1] > 80_000_000):
        raise RuntimeError("Attention weight exceeds logical size limit")
    layers.append({"name": name[:100], "shape": list(dims),
                   "tensor_type": str(w.tensor_type)[:50]})
in_width = int(getattr(attention, "query_dim", linears[0][1].weight.shape[1]))
if not (2 <= in_width <= 8192):
    raise RuntimeError("Invalid video attention input dimension")
with torch.inference_mode():
    x = torch.linspace(-1, 1, in_width * 2, device="cuda:0", dtype=torch.float16).reshape(1, 2, in_width)
    y = attention(x, pe=None, transformer_options={})
    torch.cuda.synchronize(0)
    if tuple(y.shape) != tuple(x.shape) or not torch.isfinite(y).all().item():
        raise RuntimeError("Self-attention output invalid")
    result = {"block_index": 0, "submodule": "attn1",
              "layer_count": len(linears), "layers": layers,
              "input_shape": list(x.shape), "output_shape": list(y.shape),
              "tokens": 2, "finite": True, "real_self_attention": True,
              "synthetic_input": True, "rotary_positions_applied": False,
              "full_transformer_block": False}
'''


def make_kernel():
    source = CL22_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL22_FF) == 1
    child = child.replace(CL22_FF, _ATTENTION)
    assert child.count('"transformer_ff": result,') == 1
    child = child.replace('"transformer_ff": result,', '"self_attention": result,')
    source = source.replace(lines[0], marker + repr(child))
    assert source.count('2 * 1024 * 1024 * 1024') == 1
    source = source.replace('2 * 1024 * 1024 * 1024', '3 * 1024 * 1024 * 1024')
    for old, new in (
        ('"comfy_gguf_transformer_ff"', '"comfy_gguf_attention"'),
        ('"cl22_checks"', '"cl23_checks"'),
        ('"cl22_disk_free_after_bytes"', '"cl23_disk_free_after_bytes"'),
        ('"cl22_torch_after"', '"cl23_torch_after"'),
        ('"cl22_cuda_after"', '"cl23_cuda_after"'),
        ('cl22-sources', 'cl23-sources'),
        ('cl22-transformer-ff.json', REPORT_NAME),
        ('cl22_transformer_ff_complete', 'cl23_self_attention_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl23_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_attention", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl23_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl23_" + suffix)
    validate_cl14(baseline, sha, run_id)
    attention = model.get("self_attention", {})
    if (model.get("ok") is not True
            or model.get("comfy_model_constructed") is not True
            or model.get("full_model_forward") is not False
            or model.get("video_generated") is not False
            or attention.get("block_index") != 0
            or attention.get("submodule") != "attn1"
            or attention.get("real_self_attention") is not True
            or attention.get("synthetic_input") is not True
            or attention.get("rotary_positions_applied") is not False
            or attention.get("full_transformer_block") is not False
            or attention.get("finite") is not True
            or attention.get("tokens") != 2
            or not (4 <= attention.get("layer_count", 0) <= 10)
            or len(attention.get("layers", [])) != attention.get("layer_count")
            or attention.get("input_shape") != attention.get("output_shape")
            or len(model.get("gpu_peak_allocated_bytes", [])) != 2
            or any(v > 3 * 1024 * 1024 * 1024 for v in model["gpu_peak_allocated_bytes"])
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-23 real self-attention incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
