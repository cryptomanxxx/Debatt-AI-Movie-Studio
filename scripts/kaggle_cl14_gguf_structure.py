"""CL-14: bounded Q2 structural/mmap probe; no neural network or GPU inference."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl13_first_weight import KERNEL_SOURCE as CL13_SOURCE

SLUG = "debatt-ai-movie-cl14-q2-gguf-structure"
TITLE = "Debatt AI Movie CL14 Q2 GGUF Structure"
REPORT_NAME = "cl14-gguf-structure.json"
RESULTS = Path("kaggle-cl14-results")
SUBMISSION = Path("kaggle-cl14-submission")

_INSERT_BEFORE = '''    except Exception as exc:
        first["error"] = (type(exc).__name__ + ": " + str(exc))[:500]
    finally:
        # Never let Kaggle output retrieval archive giant checkpoints.'''

_STRUCTURE = '''        # Parse only GGUF metadata/tensor index via read-only memory mapping.
        # This is not a torch state_dict, ComfyUI load, CUDA allocation, or inference.
        import gc
        import gguf
        reader = None
        try:
            reader = gguf.GGUFReader(str(stage), "r")
            tensor_count = len(reader.tensors)
            if not (1 <= tensor_count <= 100_000):
                raise ValueError("Invalid GGUF tensor count")
            first["gguf_structure"] = {
                "parsed": True,
                "tensor_count": tensor_count,
                "metadata_field_count": len(reader.fields),
                "reader_mode": "read_only_mmap",
                "neural_weights_loaded": False,
                "gpu_allocated": False,
            }
        finally:
            del reader
            gc.collect()
'''


def make_kernel():
    source = CL13_SOURCE
    assert source.count(_INSERT_BEFORE) == 1
    source = source.replace(_INSERT_BEFORE, _STRUCTURE + _INSERT_BEFORE)
    for old, new in (
        ("cl13-sources", "cl14-sources"),
        ('"cl13_checks"', '"cl14_checks"'),
        ('"cl13_disk_free_after_bytes"', '"cl14_disk_free_after_bytes"'),
        ('"cl13_torch_after"', '"cl14_torch_after"'),
        ('"cl13_cuda_after"', '"cl14_cuda_after"'),
        ("cl13-first-weight.json", REPORT_NAME),
        ("cl13_first_weight_complete", "cl14_gguf_structure_complete"),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    first = report.get("cl14_checks", {}).get("first_weight", {})
    structure = first.get("gguf_structure", {})
    if (report.get("git_sha") != sha or report.get("run_id") != run_id
            or report.get("install_completed") is not True
            or report.get("no_deps") is not True or report.get("no_cache") is not True
            or len(report.get("direct_requirement_install_steps", [])) != 12
            or not all(s.get("ok") is True for s in report["direct_requirement_install_steps"])
            or first.get("target_filename") != "ltx-2.5-22b-distilled-Q2_K.gguf"
            or first.get("download_complete") is not True
            or first.get("sha256_verified") is not True
            or first.get("stage_removed") is not True
            or structure.get("parsed") is not True
            or not (1 <= structure.get("tensor_count", 0) <= 100_000)
            or structure.get("reader_mode") != "read_only_mmap"
            or structure.get("neural_weights_loaded") is not False
            or structure.get("gpu_allocated") is not False
            or report.get("model_downloaded") is not True
            or report.get("ltx_loaded") is not False
            or report.get("video_generated") is not False
            or report.get("cl14_torch_after") != report.get("torch_before")
            or report.get("cl14_cuda_after") != report.get("cuda_before")
            or report.get("cl14_disk_free_after_bytes", 0) < 19_300_000_000):
        raise ValueError("Incomplete or unsafe CL-14 structural probe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
