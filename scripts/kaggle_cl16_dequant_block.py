"""CL-16: one real Q2_K GGUF block dequantized and transferred, no inference."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl15_gpu_sample import KERNEL_SOURCE as CL15_SOURCE

SLUG = "debatt-ai-movie-cl16-q2-block-dequant"
TITLE = "Debatt AI Movie CL16 Q2 Block Dequant"
REPORT_NAME = "cl16-q2-block-dequant.json"
RESULTS = Path("kaggle-cl16-results")
SUBMISSION = Path("kaggle-cl16-submission")

_INSERT = '''            del cpu_sample
        finally:
            # These module-scope references otherwise retain GGUF mmap views.'''

_DEQUANT = '''            # Quantized Q2_K block: 84 packed bytes -> 256 float32 elements.
            # Use the library's actual decoder; never dequantize a full tensor.
            from gguf import GGMLQuantizationType, dequantize
            qtype = GGMLQuantizationType.Q2_K
            block_elements, block_bytes = 256, 84
            q2_candidates = [t for t in reader.tensors
                             if t.tensor_type == qtype and t.data.nbytes >= block_bytes]
            if not q2_candidates:
                raise ValueError("No Q2_K tensor in this checkpoint")
            q2_source = q2_candidates[0]
            packed_block = np.asarray(q2_source.data).view(np.uint8).reshape(-1)[:block_bytes].copy()
            if packed_block.nbytes != block_bytes:
                raise ValueError("Incomplete packed Q2_K block")
            unpacked = np.asarray(dequantize(packed_block.reshape(1, block_bytes), qtype))
            if (unpacked.size != block_elements or unpacked.dtype != np.float32
                    or not np.isfinite(unpacked).all()):
                raise ValueError("Invalid Q2_K dequantization output")
            cpu_dequant = torch.from_numpy(unpacked.copy().reshape(-1))
            dequant_devices = []
            for device_index in range(2):
                gpu_dequant = None
                try:
                    with torch.cuda.device(device_index):
                        before_alloc = torch.cuda.memory_allocated()
                        free_before, _ = torch.cuda.mem_get_info()
                        if free_before < 8_000_000_000:
                            raise ValueError("Insufficient free VRAM before dequant copy")
                        gpu_dequant = cpu_dequant.to(f"cuda:{device_index}")
                        torch.cuda.synchronize(device_index)
                        after_alloc = torch.cuda.memory_allocated()
                        free_after, _ = torch.cuda.mem_get_info()
                        if not (0 < after_alloc - before_alloc <= 16_000_000):
                            raise ValueError("GPU dequant sample allocation outside guard")
                        if not (0 <= free_before - free_after <= 1_000_000_000):
                            raise ValueError("GPU free memory changed beyond guard")
                        dequant_devices.append({
                            "device_index": device_index,
                            "allocated_delta_bytes": after_alloc - before_alloc,
                            "free_before_bytes": free_before,
                            "free_after_bytes": free_after,
                        })
                finally:
                    del gpu_dequant
                    with torch.cuda.device(device_index):
                        torch.cuda.empty_cache()
            first["dequant_block"] = {
                "tensor_name": str(q2_source.name)[:120],
                "quantization": "Q2_K",
                "packed_bytes": block_bytes,
                "decoded_elements": block_elements,
                "decoded_dtype": "float32",
                "decoded_bytes": cpu_dequant.numel() * cpu_dequant.element_size(),
                "devices": dequant_devices,
                "full_model_dequantized": False,
                "neural_model_loaded": False,
                "inference_performed": False,
            }
            del cpu_dequant
'''

_RELEASE = '''            sample = None
            candidates = None
            cpu_sample = None
            sample_bytes = None
            gpu_sample = None
            del reader'''

_EXTRA_RELEASE = '''            sample = None
            candidates = None
            cpu_sample = None
            sample_bytes = None
            gpu_sample = None
            q2_candidates = None
            q2_source = None
            packed_block = None
            unpacked = None
            cpu_dequant = None
            gpu_dequant = None
            del reader'''


def make_kernel():
    source = CL15_SOURCE
    assert source.count(_INSERT) == 1
    source = source.replace(_INSERT, _DEQUANT + _INSERT)
    assert source.count(_RELEASE) == 1
    source = source.replace(_RELEASE, _EXTRA_RELEASE)
    for old, new in (
        ("cl15-sources", "cl16-sources"),
        ('"cl15_checks"', '"cl16_checks"'),
        ('"cl15_disk_free_after_bytes"', '"cl16_disk_free_after_bytes"'),
        ('"cl15_torch_after"', '"cl16_torch_after"'),
        ('"cl15_cuda_after"', '"cl16_cuda_after"'),
        ("cl15-gpu-tensor-sample.json", REPORT_NAME),
        ("cl15_gpu_tensor_sample_complete", "cl16_q2_block_dequant_complete"),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    first = report.get("cl16_checks", {}).get("first_weight", {})
    dequant = first.get("dequant_block", {})
    devices = dequant.get("devices", [])
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
            or dequant.get("quantization") != "Q2_K"
            or dequant.get("packed_bytes") != 84
            or dequant.get("decoded_elements") != 256
            or dequant.get("decoded_bytes") != 1024
            or dequant.get("decoded_dtype") != "float32"
            or dequant.get("full_model_dequantized") is not False
            or dequant.get("neural_model_loaded") is not False
            or dequant.get("inference_performed") is not False
            or len(devices) != 2
            or any(d.get("device_index") != i
                   or not (0 < d.get("allocated_delta_bytes", 0) <= 16_000_000)
                   or d.get("free_before_bytes", 0) < 8_000_000_000
                   or not (0 <= d.get("free_before_bytes", 0) - d.get("free_after_bytes", 0)
                           <= 1_000_000_000)
                   for i, d in enumerate(devices))
            or report.get("model_downloaded") is not True
            or report.get("ltx_loaded") is not False
            or report.get("video_generated") is not False
            or report.get("cl16_torch_after") != report.get("torch_before")
            or report.get("cl16_cuda_after") != report.get("cuda_before")
            or report.get("cl16_disk_free_after_bytes", 0) < 19_300_000_000):
        raise ValueError("Incomplete, stale or unsafe CL-16 dequantization report")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
