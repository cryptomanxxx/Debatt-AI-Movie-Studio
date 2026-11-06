"""CL-10: metadata-only disk/VRAM scenario arithmetic, never downloads weights."""
import argparse
import json
from pathlib import Path

from scripts.ltx25_gguf_preflight import COMPONENTS, PACKS, REPOS, evaluate

# Measured *free* GPU bytes after empty ComfyUI+GGUF startup in CL-9 run.
CL9_RUN = "42147482064"
CL9_FREE_VRAM = (15_419_113_472, 15_526_068_224)
CL9_FREE_DISK = 20_940_529_664
DISK_RESERVE = 3_000_000_000


def analyze(pack_report, free_gpu=CL9_FREE_VRAM, free_disk=CL9_FREE_DISK):
    """Compare checkpoint file sizes as illustrative memory proxies, not VRAM proofs."""
    if (len(free_gpu) != 2 or any(type(x) is not int or x <= 0 for x in free_gpu)
            or type(free_disk) is not int or free_disk <= DISK_RESERVE):
        raise ValueError("Invalid measured hardware budget")
    if not isinstance(pack_report, dict) or not isinstance(pack_report.get("files"), dict):
        raise ValueError("Missing GGUF metadata")
    files = pack_report["files"]
    scenarios = {}
    for name, components in PACKS.items():
        sizes = {}
        for key in components:
            entry = files.get(key)
            if not isinstance(entry, dict) or type(entry.get("bytes")) is not int or entry["bytes"] <= 0:
                raise ValueError(f"Missing verified checkpoint metadata: {key}")
            sizes[key] = entry["bytes"]
        total = sum(sizes.values())
        # Only one plausible 2-GPU *hypothesis*, never assert actual backend placement.
        gpu0_keys = [key for key in components if key != "encoder_q2"]
        gpu1_keys = [key for key in components if key == "encoder_q2"]
        allocations = [sum(sizes[key] for key in gpu0_keys),
                       sum(sizes[key] for key in gpu1_keys)]
        scenarios[name] = {
            "components": sizes,
            "total_file_bytes": total,
            "disk_bytes_after_weights_and_reserve": free_disk - total - DISK_RESERVE,
            "disk_weights_only_fit": total <= free_disk - DISK_RESERVE,
            "single_gpu_file_proxy_margins_bytes": [free - total for free in free_gpu],
            "hypothetical_two_gpu_placement": {
                "gpu0_components": gpu0_keys, "gpu1_components": gpu1_keys,
                "file_proxy_bytes_per_gpu": allocations,
                "file_proxy_margin_per_gpu_bytes": [free_gpu[i] - allocations[i] for i in range(2)],
                "backend_support_verified": False,
                "both_file_proxy_margins_positive": all(allocations[i] <= free_gpu[i]
                                                       for i in range(2)),
            },
            "cpu_offload_feasibility": "unknown",
            "runtime_vram_feasibility": "unknown",
            "real_gpu_memory_footprint_measured": False,
            "model_weights_loaded": False,
        }
    return {
        "source_metadata_revisions": pack_report.get("repositories"),
        "source_kaggle_startup_run": CL9_RUN,
        "free_vram_after_comfyui_startup_bytes": list(free_gpu),
        "reported_free_disk_after_startup_bytes": free_disk,
        "disk_reserve_bytes": DISK_RESERVE,
        "packs": scenarios,
        "model_downloaded": False, "gpu_used": False, "video_generated": False,
        "interpretation": "Disk file bytes are NOT GPU allocation measurements. GPU memory can include "
                          "dequantization, activations, caches, temporary buffers and offload. "
                          "The two GPUs are NOT pooled; the split is illustrative and unverified. "
                          "Disk check ignores package, model download staging and runtime space. "
                          "No checkpoint compatibility, license access or video inference was tested.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, RepoFile

    api = HfApi()
    repo_files, revisions, gated = {}, {}, {}
    for key, repo in REPOS.items():
        info = api.model_info(repo)
        if not info.sha:
            raise ValueError(f"Missing revision for {repo}")
        revisions[key], gated[key] = info.sha, bool(info.gated)
        repo_files[key] = {item.path: item.size for item in
                           api.list_repo_tree(repo, recursive=True, revision=info.sha)
                           if isinstance(item, RepoFile)}
    metadata = evaluate(repo_files, revisions, gated)
    report = analyze(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, scenario in report["packs"].items():
        print(name, "file proxy GB:", round(scenario["total_file_bytes"] / 1e9, 2),
              "disk weights-only fit:", scenario["disk_weights_only_fit"],
              "runtime GPU feasibility: UNKNOWN")
    print("Metadata only: no GPU time, model weights, or video generation.")


if __name__ == "__main__":
    main()
