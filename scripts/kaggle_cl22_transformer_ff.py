"""CL-22: exercise the real feed-forward sub-block of one LTX transformer."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl21_gguf_linear import KERNEL_SOURCE as CL21_SOURCE, _LINEAR as CL21_LINEAR

SLUG = "debatt-ai-movie-cl22-transformer-feed-forward"
TITLE = "Debatt AI Movie CL22 Transformer Feed Forward"
REPORT_NAME = "cl22-transformer-ff.json"
RESULTS = Path("kaggle-cl22-results")
SUBMISSION = Path("kaggle-cl22-submission")

_FEED_FORWARD = '''# Real LTX-2.5 first transformer block feed-forward submodule.
# The full block also requires attention/conditioning, and is NOT invoked.
import gguf
from cl20_gguf_package.ops import GGMLOps
root = model.model.diffusion_model
blocks = getattr(root, "transformer_blocks", None)
if blocks is None or len(blocks) < 1:
    raise RuntimeError("LTX model has no transformer blocks")
ff = getattr(blocks[0], "ff", None)
if ff is None:
    raise RuntimeError("LTX first transformer block has no feed-forward module")
layers = [(name, layer) for name, layer in ff.named_modules()
          if isinstance(layer, GGMLOps.Linear)]
if not (2 <= len(layers) <= 8):
    raise RuntimeError("Unexpected feed-forward linear layer count")
sizes = []
for name, layer in layers:
    w = getattr(layer, "weight", None)
    if w is None or not hasattr(w, "tensor_type") or len(w.shape) != 2:
        raise RuntimeError("Feed-forward layer has no real GGUF matrix")
    dims = tuple(int(n) for n in w.shape)
    if (min(dims) < 2 or max(dims) > 32768
            or dims[0] * dims[1] > 80_000_000):
        raise RuntimeError("Feed-forward layer exceeds matrix safety cap")
    sizes.append({"name": name[:100], "shape": list(dims),
                  "tensor_type": str(w.tensor_type)[:50]})
input_width = sizes[0]["shape"][1]
if not (2 <= input_width <= 8192):
    raise RuntimeError("Invalid first transformer feed-forward input width")
with torch.inference_mode():
    x = torch.linspace(-1, 1, input_width, device="cuda:0", dtype=torch.float16).reshape(1, 1, -1)
    y = ff(x)
    torch.cuda.synchronize(0)
    if tuple(y.shape) != tuple(x.shape) or not torch.isfinite(y).all().item():
        raise RuntimeError("Transformer feed-forward output invalid")
    result = {"block_index": 0, "submodule": "ff",
              "layer_count": len(layers), "layers": sizes,
              "input_shape": list(x.shape), "output_shape": list(y.shape),
              "finite": True, "real_gguf_submodule": True,
              "synthetic_input": True, "attention_run": False}
'''


def make_kernel():
    source = CL21_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL21_LINEAR) == 1
    child = child.replace(CL21_LINEAR, _FEED_FORWARD)
    assert child.count('"linear_forward": result,') == 1
    child = child.replace('"linear_forward": result,', '"transformer_ff": result,')
    source = source.replace(lines[0], marker + repr(child))
    assert source.count('1024 * 1024 * 1024') == 1
    source = source.replace('1024 * 1024 * 1024', '2 * 1024 * 1024 * 1024')
    for old, new in (
        ('"comfy_gguf_layer"', '"comfy_gguf_transformer_ff"'),
        ('"cl21_checks"', '"cl22_checks"'),
        ('"cl21_disk_free_after_bytes"', '"cl22_disk_free_after_bytes"'),
        ('"cl21_torch_after"', '"cl22_torch_after"'),
        ('"cl21_cuda_after"', '"cl22_cuda_after"'),
        ('cl21-sources', 'cl22-sources'),
        ('cl21-gguf-linear.json', REPORT_NAME),
        ('cl21_gguf_linear_forward_complete', 'cl22_transformer_ff_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl22_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_transformer_ff", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl22_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl22_" + suffix)
    validate_cl14(baseline, sha, run_id)
    ff = model.get("transformer_ff", {})
    if (model.get("ok") is not True
            or model.get("comfy_model_constructed") is not True
            or model.get("full_model_forward") is not False
            or model.get("video_generated") is not False
            or ff.get("block_index") != 0
            or ff.get("submodule") != "ff"
            or ff.get("real_gguf_submodule") is not True
            or ff.get("synthetic_input") is not True
            or ff.get("attention_run") is not False
            or ff.get("finite") is not True
            or not (2 <= ff.get("layer_count", 0) <= 8)
            or len(ff.get("layers", [])) != ff.get("layer_count")
            or ff.get("input_shape") != ff.get("output_shape")
            or len(model.get("gpu_peak_allocated_bytes", [])) != 2
            or any(v > 2 * 1024 * 1024 * 1024 for v in model["gpu_peak_allocated_bytes"])
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-22 transformer feed-forward incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
