"""CL-21: run one bounded real GGMLOps.Linear forward, no full LTX inference."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl20_cpu_model import KERNEL_SOURCE as CL20_SOURCE

SLUG = "debatt-ai-movie-cl21-gguf-linear-forward"
TITLE = "Debatt AI Movie CL21 GGUF Linear Forward"
REPORT_NAME = "cl21-gguf-linear.json"
RESULTS = Path("kaggle-cl21-results")
SUBMISSION = Path("kaggle-cl21-submission")

_LINEAR = '''# A real custom-GGUF layer forward with a synthetic input vector.
# Do not call diffusion_model.forward or any sampler. Do not move the full
# model to CUDA: GGMLOps handles this one quantized layer's cast internally.
import gguf
from cl20_gguf_package.ops import GGMLOps
root = model.model.diffusion_model
candidates = []
for name, layer in root.named_modules():
    if not isinstance(layer, GGMLOps.Linear):
        continue
    weight = getattr(layer, "weight", None)
    if weight is None or not hasattr(weight, "tensor_type"):
        continue
    shape = tuple(int(n) for n in weight.shape)
    if (len(shape) == 2 and 1 < shape[0] <= 8192
            and 1 < shape[1] <= 8192 and weight.numel() <= 8_000_000
            and weight.tensor_type not in (gguf.GGMLQuantizationType.F32,
                                           gguf.GGMLQuantizationType.F16)):
        candidates.append((name, layer, shape))
if not candidates:
    raise RuntimeError("No bounded quantized GGMLOps.Linear layer found")
name, layer, shape = min(candidates, key=lambda entry: entry[2][0] * entry[2][1])
with torch.inference_mode():
    x = torch.linspace(-1, 1, shape[1], device="cuda:0", dtype=torch.float16).reshape(1, -1)
    y = layer(x)
    torch.cuda.synchronize(0)
    if tuple(y.shape) != (1, shape[0]) or not torch.isfinite(y).all().item():
        raise RuntimeError("GGUF custom linear returned invalid output")
    result = {"name": name[:160], "shape": list(shape),
              "output_shape": list(y.shape), "finite": True,
              "input": "synthetic_linspace", "real_gguf_layer": True}
'''


def make_kernel():
    source = CL20_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    target = 'model_patcher_type = type(model).__name__\n'
    assert child.count(target) == 1
    child = child.replace(target, target + _LINEAR + '\n')
    assert child.count('"model_patcher_type": model_patcher_type,') == 1
    child = child.replace('"model_patcher_type": model_patcher_type,',
                          '"model_patcher_type": model_patcher_type,\n'
                          '    "linear_forward": result,')
    source = source.replace(lines[0], marker + repr(child))
    # The single layer may transiently dequantize weights; still refuse >1GB
    # peak PyTorch allocation per GPU, or >12GB sampled host RAM.
    assert source.count('128 * 1024 * 1024') == 1
    source = source.replace('128 * 1024 * 1024', '1024 * 1024 * 1024')
    for old, new in (
        ('"comfy_gguf_model"', '"comfy_gguf_layer"'),
        ('"cl20_checks"', '"cl21_checks"'),
        ('"cl20_disk_free_after_bytes"', '"cl21_disk_free_after_bytes"'),
        ('"cl20_torch_after"', '"cl21_torch_after"'),
        ('"cl20_cuda_after"', '"cl21_cuda_after"'),
        ('cl20-sources', 'cl21-sources'),
        ('cl20-cpu-model.json', REPORT_NAME),
        ('cl20_cpu_model_construct_complete', 'cl21_gguf_linear_forward_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl21_checks", {}).get("first_weight", {})
    layer = first.get("comfy_gguf_layer", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl21_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl21_" + suffix)
    validate_cl14(baseline, sha, run_id)
    linear = layer.get("linear_forward", {})
    if (layer.get("ok") is not True
            or layer.get("comfy_model_constructed") is not True
            or layer.get("full_model_forward") is not False
            or layer.get("video_generated") is not False
            or linear.get("finite") is not True
            or linear.get("real_gguf_layer") is not True
            or linear.get("input") != "synthetic_linspace"
            or len(linear.get("shape", [])) != 2
            or len(layer.get("gpu_peak_allocated_bytes", [])) != 2
            or any(n > 1024 * 1024 * 1024 for n in layer["gpu_peak_allocated_bytes"])
            or not (0 < layer.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-21 custom GGUF linear forward incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
