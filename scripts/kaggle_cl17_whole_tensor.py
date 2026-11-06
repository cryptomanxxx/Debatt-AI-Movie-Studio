"""CL-17: dequantize one *complete* bounded Q2_K tensor, not the model."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl16_dequant_block import KERNEL_SOURCE as CL16_SOURCE

SLUG = "debatt-ai-movie-cl17-bounded-whole-tensor"
TITLE = "Debatt AI Movie CL17 Bounded Whole Tensor"
REPORT_NAME = "cl17-whole-tensor.json"
RESULTS = Path("kaggle-cl17-results")
SUBMISSION = Path("kaggle-cl17-submission")

_INSERT = '''            del cpu_dequant
            del cpu_sample
        finally:
            # These module-scope references otherwise retain GGUF mmap views.'''

_FULL_TENSOR = '''            # One COMPLETE, small Q2_K tensor; bounded before any dequantization.
            whole_candidates = []
            for t in reader.tensors:
                n = t.data.nbytes
                decoded_bytes = (n // 84) * 1024 if n % 84 == 0 else 0
                if (t.tensor_type == qtype and 84 <= n <= 4 * 1024 * 1024
                        and 1024 <= decoded_bytes <= 64 * 1024 * 1024):
                    whole_candidates.append(t)
            if not whole_candidates:
                raise ValueError("No whole Q2_K tensor within 64 MiB decoded cap")
            whole_source = min(whole_candidates, key=lambda t: t.data.nbytes)
            compressed_size = whole_source.data.nbytes
            max_decoded_size = compressed_size // 84 * 1024
            if max_decoded_size > 64 * 1024 * 1024:
                raise ValueError("Decoded tensor would exceed 64 MiB")
            whole_decoded = np.asarray(dequantize(whole_source.data, qtype))
            if (whole_decoded.nbytes != max_decoded_size
                    or whole_decoded.dtype != np.float32
                    or not np.isfinite(whole_decoded).all()):
                raise ValueError("Whole-tensor GGUF dequantization mismatch")
            cpu_whole = torch.from_numpy(whole_decoded.copy().reshape(-1))
            whole_devices = []
            for device_index in range(2):
                gpu_whole = None
                try:
                    with torch.cuda.device(device_index):
                        free_before, _ = torch.cuda.mem_get_info()
                        allocated_before = torch.cuda.memory_allocated()
                        if free_before < 8_000_000_000:
                            raise ValueError("Insufficient VRAM for whole-tensor probe")
                        gpu_whole = cpu_whole.to(f"cuda:{device_index}")
                        torch.cuda.synchronize(device_index)
                        allocated_delta = torch.cuda.memory_allocated() - allocated_before
                        free_after, _ = torch.cuda.mem_get_info()
                        if not (0 < allocated_delta <= 128 * 1024 * 1024):
                            raise ValueError("GPU allocated more than 128 MiB")
                        if not (0 <= free_before - free_after <= 512 * 1024 * 1024):
                            raise ValueError("GPU free-memory delta exceeded 512 MiB")
                        whole_devices.append({
                            "device_index": device_index,
                            "allocated_delta_bytes": allocated_delta,
                            "free_before_bytes": free_before,
                            "free_after_bytes": free_after,
                        })
                finally:
                    del gpu_whole
                    with torch.cuda.device(device_index):
                        torch.cuda.empty_cache()
            first["whole_tensor"] = {
                "tensor_name": str(whole_source.name)[:120],
                "quantization": "Q2_K",
                "compressed_bytes": compressed_size,
                "decoded_bytes": cpu_whole.numel() * cpu_whole.element_size(),
                "decoded_elements": cpu_whole.numel(),
                "decoded_dtype": "float32",
                "whole_tensor_dequantized": True,
                "whole_model_loaded": False,
                "inference_performed": False,
                "devices": whole_devices,
            }
            del cpu_whole
'''

_RELEASE = '''            gpu_dequant = None
            del reader'''
_EXTRA_RELEASE = '''            gpu_dequant = None
            t = None
            whole_candidates = None
            whole_source = None
            whole_decoded = None
            cpu_whole = None
            gpu_whole = None
            del reader'''


def make_kernel():
    source = CL16_SOURCE
    assert source.count(_INSERT) == 1
    source = source.replace(_INSERT, _FULL_TENSOR + _INSERT)
    assert source.count(_RELEASE) == 1
    source = source.replace(_RELEASE, _EXTRA_RELEASE)
    for old, new in (
        ("cl16-sources", "cl17-sources"),
        ('"cl16_checks"', '"cl17_checks"'),
        ('"cl16_disk_free_after_bytes"', '"cl17_disk_free_after_bytes"'),
        ('"cl16_torch_after"', '"cl17_torch_after"'),
        ('"cl16_cuda_after"', '"cl17_cuda_after"'),
        ("cl16-q2-block-dequant.json", REPORT_NAME),
        ("cl16_q2_block_dequant_complete", "cl17_bounded_whole_tensor_complete"),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    first = report.get("cl17_checks", {}).get("first_weight", {})
    whole = first.get("whole_tensor", {})
    devices = whole.get("devices", [])
    compressed = whole.get("compressed_bytes", 0)
    decoded = whole.get("decoded_bytes", 0)
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
            or whole.get("quantization") != "Q2_K"
            or not (84 <= compressed <= 4 * 1024 * 1024)
            or compressed % 84 != 0
            or decoded != compressed // 84 * 1024
            or not (1024 <= decoded <= 64 * 1024 * 1024)
            or whole.get("decoded_elements") != decoded // 4
            or whole.get("decoded_dtype") != "float32"
            or whole.get("whole_tensor_dequantized") is not True
            or whole.get("whole_model_loaded") is not False
            or whole.get("inference_performed") is not False
            or len(devices) != 2
            or any(d.get("device_index") != i
                   or not (0 < d.get("allocated_delta_bytes", 0) <= 128 * 1024 * 1024)
                   or d.get("free_before_bytes", 0) < 8_000_000_000
                   or not (0 <= d.get("free_before_bytes", 0) - d.get("free_after_bytes", 0)
                           <= 512 * 1024 * 1024)
                   for i, d in enumerate(devices))
            or report.get("model_downloaded") is not True
            or report.get("ltx_loaded") is not False
            or report.get("video_generated") is not False
            or report.get("cl17_torch_after") != report.get("torch_before")
            or report.get("cl17_cuda_after") != report.get("cuda_before")
            or report.get("cl17_disk_free_after_bytes", 0) < 19_300_000_000):
        raise ValueError("Incomplete or unsafe CL-17 whole-tensor report")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
