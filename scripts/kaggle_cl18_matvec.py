"""CL-18: one real Q2_K weight matrix times synthetic vector on both GPUs."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl17_whole_tensor import KERNEL_SOURCE as CL17_SOURCE

SLUG = "debatt-ai-movie-cl18-single-matrix-vector"
TITLE = "Debatt AI Movie CL18 Single Matrix Vector"
REPORT_NAME = "cl18-single-matvec.json"
RESULTS = Path("kaggle-cl18-results")
SUBMISSION = Path("kaggle-cl18-submission")

_INSERT = '''            del cpu_whole
            del cpu_dequant
            del cpu_sample
        finally:
            # These module-scope references otherwise retain GGUF mmap views.'''

_MATVEC = '''            # First bounded *arithmetic* with real dequantized LTX weights.
            # Synthetic vector only; this is NOT a valid LTX forward pass.
            dims = tuple(int(d) for d in whole_source.shape)
            if (len(dims) != 2 or min(dims) < 2 or max(dims) > 8192
                    or dims[0] * dims[1] != cpu_whole.numel()):
                raise ValueError("Selected Q2_K tensor is not a bounded 2-D matrix")
            rows, cols = dims
            cpu_matrix = cpu_whole.reshape(rows, cols)
            cpu_input = torch.linspace(-1.0, 1.0, steps=cols, dtype=torch.float32)
            cpu_result = torch.mv(cpu_matrix, cpu_input)
            if not torch.isfinite(cpu_result).all():
                raise ValueError("CPU projection returned non-finite values")
            projection_devices = []
            for device_index in range(2):
                gpu_matrix = gpu_input = gpu_result = None
                try:
                    with torch.cuda.device(device_index):
                        free_before, _ = torch.cuda.mem_get_info()
                        allocated_before = torch.cuda.memory_allocated()
                        if free_before < 8_000_000_000:
                            raise ValueError("Insufficient VRAM for projection")
                        gpu_matrix = cpu_matrix.to(f"cuda:{device_index}")
                        gpu_input = cpu_input.to(f"cuda:{device_index}")
                        gpu_result = torch.mv(gpu_matrix, gpu_input)
                        torch.cuda.synchronize(device_index)
                        allocated_delta = torch.cuda.memory_allocated() - allocated_before
                        free_after, _ = torch.cuda.mem_get_info()
                        if not (0 < allocated_delta <= 128 * 1024 * 1024):
                            raise ValueError("Projection allocation exceeded 128 MiB")
                        if not (0 <= free_before - free_after <= 512 * 1024 * 1024):
                            raise ValueError("Projection free-memory change exceeded 512 MiB")
                        actual = gpu_result.cpu()
                        max_abs_error = float((actual - cpu_result).abs().max().item())
                        if not torch.allclose(actual, cpu_result, rtol=1e-3, atol=1e-3):
                            raise ValueError("GPU matrix-vector output differs from CPU")
                        projection_devices.append({
                            "device_index": device_index,
                            "allocated_delta_bytes": allocated_delta,
                            "free_before_bytes": free_before,
                            "free_after_bytes": free_after,
                            "max_absolute_error": max_abs_error,
                            "cpu_gpu_allclose": True,
                        })
                finally:
                    gpu_result = gpu_input = gpu_matrix = None
                    with torch.cuda.device(device_index):
                        torch.cuda.empty_cache()
            first["single_projection"] = {
                "matrix_tensor_name": str(whole_source.name)[:120],
                "rows": rows, "cols": cols,
                "input_type": "synthetic_linspace",
                "operation": "torch.mv",
                "real_dequantized_weight_matrix": True,
                "full_model_forward": False,
                "video_generated": False,
                "devices": projection_devices,
            }
            cpu_result = cpu_input = cpu_matrix = None
'''

_RELEASE = '''            gpu_whole = None
            del reader'''
_EXTRA_RELEASE = '''            gpu_whole = None
            cpu_result = None
            cpu_input = None
            cpu_matrix = None
            gpu_result = None
            gpu_input = None
            gpu_matrix = None
            actual = None
            del reader'''


def make_kernel():
    source = CL17_SOURCE
    assert source.count(_INSERT) == 1
    source = source.replace(_INSERT, _MATVEC + _INSERT)
    assert source.count(_RELEASE) == 1
    source = source.replace(_RELEASE, _EXTRA_RELEASE)
    for old, new in (
        ("cl17-sources", "cl18-sources"),
        ('"cl17_checks"', '"cl18_checks"'),
        ('"cl17_disk_free_after_bytes"', '"cl18_disk_free_after_bytes"'),
        ('"cl17_torch_after"', '"cl18_torch_after"'),
        ('"cl17_cuda_after"', '"cl18_cuda_after"'),
        ("cl17-whole-tensor.json", REPORT_NAME),
        ("cl17_bounded_whole_tensor_complete", "cl18_single_matvec_complete"),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    first = report.get("cl18_checks", {}).get("first_weight", {})
    proj = first.get("single_projection", {})
    devices = proj.get("devices", [])
    rows, cols = proj.get("rows", 0), proj.get("cols", 0)
    whole = first.get("whole_tensor", {})
    if (report.get("git_sha") != sha or report.get("run_id") != run_id
            or report.get("install_completed") is not True
            or report.get("no_deps") is not True or report.get("no_cache") is not True
            or len(report.get("direct_requirement_install_steps", [])) != 12
            or not all(s.get("ok") is True for s in report["direct_requirement_install_steps"])
            or first.get("download_complete") is not True
            or first.get("sha256_verified") is not True
            or first.get("stage_removed") is not True
            or first.get("gguf_structure", {}).get("parsed") is not True
            or len(first.get("gpu_sample", {}).get("devices", [])) != 2
            or len(first.get("dequant_block", {}).get("devices", [])) != 2
            or whole.get("whole_tensor_dequantized") is not True
            or whole.get("whole_model_loaded") is not False
            or not (2 <= rows <= 8192 and 2 <= cols <= 8192)
            or rows * cols != whole.get("decoded_elements")
            or proj.get("operation") != "torch.mv"
            or proj.get("input_type") != "synthetic_linspace"
            or proj.get("real_dequantized_weight_matrix") is not True
            or proj.get("full_model_forward") is not False
            or proj.get("video_generated") is not False
            or len(devices) != 2
            or any(d.get("device_index") != i
                   or d.get("cpu_gpu_allclose") is not True
                   or not (0 <= d.get("max_absolute_error", -1) < float("inf"))
                   or not (0 < d.get("allocated_delta_bytes", 0) <= 128 * 1024 * 1024)
                   or d.get("free_before_bytes", 0) < 8_000_000_000
                   or not (0 <= d.get("free_before_bytes", 0) - d.get("free_after_bytes", 0)
                           <= 512 * 1024 * 1024)
                   for i, d in enumerate(devices))
            or report.get("model_downloaded") is not True
            or report.get("ltx_loaded") is not False
            or report.get("video_generated") is not False
            or report.get("cl18_torch_after") != report.get("torch_before")
            or report.get("cl18_cuda_after") != report.get("cuda_before")
            or report.get("cl18_disk_free_after_bytes", 0) < 19_300_000_000):
        raise ValueError("Incomplete or unsafe CL-18 matrix-vector report")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
