"""CL-20: bounded attempt to construct the GGUF LTX model on CPU only."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl19_loader_mmap import KERNEL_SOURCE as CL19_SOURCE, _CHILD_CODE as CL19_CHILD

SLUG = "debatt-ai-movie-cl20-cpu-model-construct"
TITLE = "Debatt AI Movie CL20 CPU Model Construction"
REPORT_NAME = "cl20-cpu-model.json"
RESULTS = Path("kaggle-cl20-results")
SUBMISSION = Path("kaggle-cl20-submission")

_CONSTRUCT = '''# Construct the real diffusion model on CPU with GGUF custom linear ops.
# No GPU model loading, sampling, inference, or generation is permitted.
import inspect
import comfy.sd
from cl20_gguf_package.ops import GGMLOps
options = {"custom_operations": GGMLOps(),
           "load_device": torch.device("cpu"),
           "offload_device": torch.device("cpu")}
kwargs = {}
if "metadata" in inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters:
    kwargs["metadata"] = extra.get("metadata", {})
model = comfy.sd.load_diffusion_model_state_dict(
    state_dict, model_options=options, **kwargs)
if model is None:
    raise RuntimeError("ComfyUI could not recognize the LTX diffusion model")
model_patcher_type = type(model).__name__
'''


def make_kernel():
    child = CL19_CHILD
    assert child.count('pkg_name = "cl19_gguf_package"') == 1
    child = child.replace('pkg_name = "cl19_gguf_package"',
                          'pkg_name = "cl20_gguf_package"')
    assert child.count('print(json.dumps({') == 1
    child = child.replace('print(json.dumps({', _CONSTRUCT + '\nprint(json.dumps({')
    assert child.count('"state_dict_parsed_only": True, "comfy_model_constructed": False,') == 1
    child = child.replace('"state_dict_parsed_only": True, "comfy_model_constructed": False,',
                          '"state_dict_parsed_only": False, "comfy_model_constructed": True,\n'
                          '    "model_patcher_type": model_patcher_type,')
    source = CL19_SOURCE
    assert source.count(repr(CL19_CHILD)) == 1
    source = source.replace(repr(CL19_CHILD), repr(child))
    # A full 22B GGUF model construction may consume much more host memory
    # than a memory-mapped index. A separate watchdog kills observed overruns.
    assert source.count('3_000_000_000') == 2
    source = source.replace('3_000_000_000', '12_000_000_000')
    assert source.count('16 * 1024 * 1024') == 1
    source = source.replace('16 * 1024 * 1024', '128 * 1024 * 1024')
    assert source.count('> 120') == 1
    source = source.replace('> 120', '> 180')
    for old, new in (
        ('"comfy_gguf_loader"', '"comfy_gguf_model"'),
        ('"cl19_checks"', '"cl20_checks"'),
        ('"cl19_disk_free_after_bytes"', '"cl20_disk_free_after_bytes"'),
        ('"cl19_torch_after"', '"cl20_torch_after"'),
        ('"cl19_cuda_after"', '"cl20_cuda_after"'),
        ('cl19-sources', 'cl20-sources'),
        ('cl19-gguf-loader-mmap.json', REPORT_NAME),
        ('cl19_gguf_loader_mmap_complete', 'cl20_cpu_model_construct_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl20_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_model", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl20_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl20_" + suffix)
    validate_cl14(baseline, sha, run_id)
    if (model.get("ok") is not True
            or model.get("entrypoint") != "gguf_sd_loader"
            or model.get("comfy_model_constructed") is not True
            or model.get("state_dict_parsed_only") is not False
            or not model.get("model_patcher_type")
            or not (1000 <= model.get("tensor_count", 0) <= 100_000)
            or model.get("full_model_forward") is not False
            or model.get("video_generated") is not False
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)
            or len(model.get("gpu_peak_allocated_bytes", [])) != 2
            or any(v > 128 * 1024 * 1024 for v in model["gpu_peak_allocated_bytes"])):
        raise ValueError("CL-20 CPU model construction incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
