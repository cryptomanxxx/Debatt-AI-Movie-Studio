"""CL-41: CPU-only evidence gate for planning the next real LTX video block.

The previous proof uses blocks 0..7. This gate does not inspect model weights,
infer total model depth, or authorize any Kaggle work.
"""
import argparse
import json
import math
from pathlib import Path

SOURCE_RUN = 42215353262
SOURCE_SHA = "3b76f5c667998e008bdcd7a21be5be8cd1e3ec39"
PROVEN_BLOCKS = tuple(range(8))
NEXT_INDEX = 8
TORCH_LIMIT = 4 * 1024**3
RSS_LIMIT = 12_000_000_000
SAMPLED_TORCH_BYTES = 1_521_324_032
SAMPLED_CHILD_RSS_BYTES = 2_791_383_040


def build_gate():
    """Construct a plan from the reviewed CL-40 GitHub Actions log measurements."""
    assert PROVEN_BLOCKS == tuple(range(NEXT_INDEX))
    assert 0 < SAMPLED_TORCH_BYTES < TORCH_LIMIT
    assert 0 < SAMPLED_CHILD_RSS_BYTES < RSS_LIMIT
    return {
        "experiment": "cl41_next_block_expansion_gate",
        "source": {"github_actions_run": SOURCE_RUN, "commit": SOURCE_SHA},
        "evidence": {
            "proven_video_block_indices": list(PROVEN_BLOCKS),
            "three_native_model_evaluations": True,
            "torch_peak_allocated_bytes": SAMPLED_TORCH_BYTES,
            "sampled_child_peak_rss_bytes": SAMPLED_CHILD_RSS_BYTES,
            "torch_allocation_cap_bytes": TORCH_LIMIT,
            "child_rss_cap_bytes": RSS_LIMIT,
            "torch_headroom_bytes": TORCH_LIMIT - SAMPLED_TORCH_BYTES,
            "sampled_rss_headroom_bytes": RSS_LIMIT - SAMPLED_CHILD_RSS_BYTES,
        },
        "next_candidate_block": NEXT_INDEX,
        "next_candidate_weights_verified": False,
        "total_video_block_count_verified": False,
        "next_block_weight_sizes_known": False,
        "next_block_memory_fits": None,
        "next_kaggle_run_authorized": False,
        "gpu_run_triggered": False,
        "model_weights_downloaded": False,
        "full_model_forward": False,
        "video_generated": False,
        "requirements_before_candidate_execution": [
            "Confirm that video transformer block 8 exists in the pinned model config",
            "Enumerate every required quantized block-8 projection and enforce <=80M logical elements per weight",
            "Abort before tensor materialization if any new weight exceeds the existing projection limit",
            "Retain 4GiB PyTorch, 12GB sampled RSS, 180s watchdog and explicit manual Kaggle authorization",
            "Test only one extra block before considering a multi-block expansion",
        ],
        "decision": "METADATA_PREFLIGHT_REQUIRED",
    }


def validate_gate(report):
    if report != build_gate():
        raise ValueError("CL-41 evidence gate inconsistent")
    numeric = report["evidence"]
    if not all(isinstance(v, int) and math.isfinite(v) for v in numeric.values()
               if not isinstance(v, (list, bool))):
        raise ValueError("Invalid numeric evidence")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cl41-results/report.json"))
    args = parser.parse_args(argv)
    report = build_gate()
    validate_gate(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
