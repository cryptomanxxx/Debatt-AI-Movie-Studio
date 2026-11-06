"""CL-15: bounded copy of GGUF packed tensor bytes to each T4, never inference."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl14_gguf_structure import KERNEL_SOURCE as CL14_SOURCE

SLUG = "debatt-ai-movie-cl15-gpu-tensor-sample"
TITLE = "Debatt AI Movie CL15 GPU Tensor Sample"
REPORT_NAME = "cl15-gpu-tensor-sample.json"
RESULTS = Path("kaggle-cl15-results")
SUBMISSION = Path("kaggle-cl15-submission")

_MARKER = '''        finally:
            del reader
            gc.collect()
'''
_GPU_SAMPLE = '''            # Copy <=4 MiB of one raw, packed GGUF tensor onto EACH T4.
            # No dequantization, model layers, inference, or optimizer state.
            import numpy as np
            candidates = [t for t in reader.tensors
                          if 1024 <= t.data.nbytes <= 4 * 1024 * 1024]
            if not candidates:
                raise ValueError("No bounded GGUF tensor sample available")
            sample = min(candidates, key=lambda t: t.data.nbytes)
            sample_bytes = sample.data.tobytes()
            if not (1024 <= len(sample_bytes) <= 4 * 1024 * 1024):
                raise ValueError("Sample exceeds 4 MiB cap")
            if torch.cuda.device_count() != 2:
                raise ValueError("Two CUDA GPUs required")
            cpu_sample = torch.from_numpy(np.frombuffer(sample_bytes, dtype=np.uint8).copy())
            del sample_bytes
            measurements = []
            for device_index in range(2):
                gpu_sample = None
                try:
                    with torch.cuda.device(device_index):
                        free_before, total = torch.cuda.mem_get_info()
                        allocated_before = torch.cuda.memory_allocated()
                        if free_before < 8_000_000_000:
                            raise ValueError("GPU free VRAM below 8 GB guard")
                        gpu_sample = cpu_sample.to(device=f"cuda:{device_index}")
                        torch.cuda.synchronize(device_index)
                        free_during, _ = torch.cuda.mem_get_info()
                        allocated_during = torch.cuda.memory_allocated()
                        if (free_before - free_during > 1_000_000_000
                                or allocated_during - allocated_before > 16_000_000):
                            raise ValueError("GPU sample memory guard exceeded")
                        measurements.append({
                            "device_index": device_index, "device_name": torch.cuda.get_device_name(device_index),
                            "total_vram_bytes": total, "free_before_bytes": free_before,
                            "free_during_bytes": free_during,
                            "allocated_delta_bytes": allocated_during - allocated_before,
                            "raw_sample_bytes": cpu_sample.numel(),
                        })
                finally:
                    del gpu_sample
                    with torch.cuda.device(device_index):
                        torch.cuda.empty_cache()
            first["gpu_sample"] = {
                "raw_packed_bytes_only": True,
                "sample_tensor_name": str(sample.name)[:120],
                "sample_bytes": cpu_sample.numel(),
                "devices": measurements,
                "full_model_loaded": False,
                "inference_performed": False,
            }
            del cpu_sample
'''


def make_kernel():
    source = CL14_SOURCE
    assert source.count(_MARKER) == 1
    source = source.replace(_MARKER, _GPU_SAMPLE + _MARKER)
    for old, new in (
        ("cl14-sources", "cl15-sources"),
        ('"cl14_checks"', '"cl15_checks"'),
        ('"cl14_disk_free_after_bytes"', '"cl15_disk_free_after_bytes"'),
        ('"cl14_torch_after"', '"cl15_torch_after"'),
        ('"cl14_cuda_after"', '"cl15_cuda_after"'),
        ("cl14-gguf-structure.json", REPORT_NAME),
        ("cl14_gguf_structure_complete", "cl15_gpu_tensor_sample_complete"),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    first = report.get("cl15_checks", {}).get("first_weight", {})
    structure = first.get("gguf_structure", {})
    gpu = first.get("gpu_sample", {})
    devices = gpu.get("devices", [])
    if (report.get("git_sha") != sha or report.get("run_id") != run_id
            or report.get("install_completed") is not True
            or report.get("no_deps") is not True or report.get("no_cache") is not True
            or len(report.get("direct_requirement_install_steps", [])) != 12
            or not all(s.get("ok") is True for s in report["direct_requirement_install_steps"])
            or first.get("download_complete") is not True
            or first.get("sha256_verified") is not True
            or first.get("stage_removed") is not True
            or structure.get("parsed") is not True
            or not (1 <= structure.get("tensor_count", 0) <= 100_000)
            or structure.get("neural_weights_loaded") is not False
            or gpu.get("raw_packed_bytes_only") is not True
            or not (1024 <= gpu.get("sample_bytes", 0) <= 4 * 1024 * 1024)
            or gpu.get("full_model_loaded") is not False
            or gpu.get("inference_performed") is not False
            or len(devices) != 2
            or any(d.get("device_index") != i
                   or not (0 < d.get("allocated_delta_bytes", 0) <= 16_000_000)
                   or d.get("raw_sample_bytes") != gpu["sample_bytes"]
                   or d.get("free_before_bytes", 0) < 8_000_000_000
                   or not (0 <= d.get("free_before_bytes", 0) - d.get("free_during_bytes", 0)
                           <= 1_000_000_000)
                   for i, d in enumerate(devices))
            or report.get("model_downloaded") is not True
            or report.get("ltx_loaded") is not False
            or report.get("video_generated") is not False
            or report.get("cl15_torch_after") != report.get("torch_before")
            or report.get("cl15_cuda_after") != report.get("cuda_before")
            or report.get("cl15_disk_free_after_bytes", 0) < 19_300_000_000):
        raise ValueError("Incomplete, stale, or unsafe CL-15 GPU sample report")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
