"""CL-42: inspect pinned GGUF video-block 8 metadata only; no tensor evaluation."""
import re
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl14_gguf_structure import (
    KERNEL_SOURCE as CL14_SOURCE,
    _STRUCTURE as CL14_STRUCTURE,
    validate_report as validate_cl14,
)

SLUG = "debatt-ai-movie-cl42-block8-metadata"
TITLE = "Debatt AI Movie CL42 Block8 Metadata"
REPORT_NAME = "cl42-block8-metadata.json"
RESULTS = Path("kaggle-cl42-results")
SUBMISSION = Path("kaggle-cl42-submission")

_BLOCK8_SCAN = '''            # Names/shapes only: never touch tensor.data or construct a model.
            import re
            pattern = re.compile(r"(?:^|\\.)transformer_blocks\\.(\\d+)\\.")
            # Isolate ReaderTensor references even if scanning throws an error;
            # otherwise a loop-local mmap view survives the reader cleanup.
            def scan_tensor_index(reader):
                all_block_indices = set()
                block8 = []
                for tensor in reader.tensors:
                    name = str(tensor.name)
                    match = pattern.search(name)
                    if match is None:
                        continue
                    index = int(match.group(1))
                    if index > 1024:
                        raise ValueError("Implausible video block index")
                    all_block_indices.add(index)
                    if index == 8:
                        shape = [int(n) for n in tensor.shape]
                        if not shape or len(shape) > 8 or any(n < 1 for n in shape):
                            raise ValueError("Invalid block 8 GGUF shape")
                        logical = math.prod(shape)
                        if not (1 <= logical <= 1_000_000_000):
                            raise ValueError("Unbounded block 8 tensor shape")
                        if len(block8) >= 300:
                            raise ValueError("Too many block 8 tensors")
                        qtype = str(tensor.tensor_type)
                        if not re.fullmatch(r"(?:[0-9]{1,3}|(?:GGMLQuantizationType\\.)?[A-Z][A-Z0-9_]*)", qtype):
                            raise ValueError("Invalid GGUF quantization metadata")
                        block8.append({"name": name[:200], "shape": shape,
                                       "logical_elements": logical,
                                       "quantization_type": qtype,
                                       "under_projection_80m_guard": logical <= 80_000_000})
                return block8, all_block_indices
            block8, all_block_indices = scan_tensor_index(reader)
            block8.sort(key=lambda item: item["name"])
            first["block8_metadata"] = {
                "block8_exists_in_gguf_index": bool(block8),
                "block8_tensor_count": len(block8),
                "observed_video_block_indices": sorted(all_block_indices),
                "tensors": block8,
                "all_block8_tensors_under_80m": bool(block8) and all(
                    item["under_projection_80m_guard"] for item in block8),
                "source": "read_only_gguf_tensor_index",
                "comfy_model_constructed": False,
                "gpu_inference": False,
                "full_model_forward": False,
                "video_generated": False,
            }
'''


def make_kernel():
    source = CL14_SOURCE
    assert source.count(CL14_STRUCTURE) == 1
    extended = CL14_STRUCTURE.replace(
        '            first["gguf_structure"] = {',
        _BLOCK8_SCAN + '            first["gguf_structure"] = {'
    )
    assert extended != CL14_STRUCTURE
    # The structural block already imports gc; add math for shape products.
    extended = extended.replace('        import gc\n', '        import gc\n        import math\n')
    source = source.replace(CL14_STRUCTURE, extended)
    for old, new in (
        ('cl14-sources', 'cl42-sources'),
        ('"cl14_checks"', '"cl42_checks"'),
        ('"cl14_disk_free_after_bytes"', '"cl42_disk_free_after_bytes"'),
        ('"cl14_torch_after"', '"cl42_torch_after"'),
        ('"cl14_cuda_after"', '"cl42_cuda_after"'),
        ('cl14-gguf-structure.json', REPORT_NAME),
        ('cl14_gguf_structure_complete', 'cl42_block8_metadata_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    prior = dict(report)
    prior["cl14_checks"] = report.get("cl42_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        prior["cl14_" + suffix] = report.get("cl42_" + suffix)
    validate_cl14(prior, sha, run_id)
    metadata = report.get("cl42_checks", {}).get("first_weight", {}).get("block8_metadata", {})
    tensors = metadata.get("tensors")
    indices = metadata.get("observed_video_block_indices")
    if (metadata.get("source") != "read_only_gguf_tensor_index"
            or metadata.get("comfy_model_constructed") is not False
            or metadata.get("gpu_inference") is not False
            or metadata.get("full_model_forward") is not False
            or metadata.get("video_generated") is not False
            or not isinstance(tensors, list) or len(tensors) > 300
            or metadata.get("block8_tensor_count") != len(tensors)
            or metadata.get("block8_exists_in_gguf_index") is not bool(tensors)
            or not isinstance(indices, list) or indices != sorted(set(indices))
            or not all(isinstance(i, int) and 0 <= i <= 1024 for i in indices)
            or (bool(tensors) != (8 in indices))
            or any(not isinstance(t, dict) or not isinstance(t.get("name"), str)
                   or not re.search(r"(?:^|\.)transformer_blocks\.8\.", t["name"])
                   or not isinstance(t.get("shape"), list)
                   or not 1 <= len(t["shape"]) <= 8
                   or not all(isinstance(n, int) and n > 0 for n in t["shape"])
                   or t.get("logical_elements") != __import__("math").prod(t["shape"])
                   or not isinstance(t.get("quantization_type"), str)
                   or not re.fullmatch(r"(?:[0-9]{1,3}|(?:GGMLQuantizationType\.)?[A-Z][A-Z0-9_]*)", t["quantization_type"])
                   or t.get("under_projection_80m_guard") is not (t["logical_elements"] <= 80_000_000)
                   for t in tensors)
            or metadata.get("all_block8_tensors_under_80m") is not (
                bool(tensors) and all(t["under_projection_80m_guard"] for t in tensors))):
        raise ValueError("Incomplete or misleading CL-42 metadata-only evidence")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
