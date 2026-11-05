"""Metadata-only GGUF storage feasibility check; no model downloads or GPU."""
import argparse
import json
from pathlib import Path

from scripts.ltx25_preflight import DISK_FREE, RESERVE

REPOS = {
    "transformer": "unsloth/LTX-2.5-GGUF",
    "encoder": "elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF",
    "vae": "Lightricks/LTX-2.5",
}
# Exact basenames, with unique-match enforcement: no accidentally counting a
# model patch, quantization of the wrong architecture or an unrelated file.
COMPONENTS = {
    "transformer_q2": ("transformer", "ltx-2.5-22b-distilled-Q2_K.gguf"),
    "transformer_ud_q2": ("transformer", "ltx-2.5-22b-distilled-UD-Q2_K_XL.gguf"),
    "encoder_q2": ("encoder", "gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf"),
    "video_vae": ("vae", "ltx-2.5-video-vae-conv-bf16.safetensors"),
    "audio_vae": ("vae", "ltx-2.5-audio-vae-bf16.safetensors"),
}
PACKS = {
    "distilled_q2": ("transformer_q2", "encoder_q2", "video_vae", "audio_vae"),
    "distilled_ud_q2": ("transformer_ud_q2", "encoder_q2", "video_vae", "audio_vae"),
}


def evaluate(repo_files, revisions, gated):
    """Evaluate complete pack weights against measured *disk*, not VRAM."""
    files = {}
    for component, (repo_key, basename) in COMPONENTS.items():
        matches = [(path, size) for path, size in repo_files[repo_key].items()
                   if path.rsplit("/", 1)[-1] == basename]
        if len(matches) != 1:
            raise ValueError(f"Expected one {component} in {REPOS[repo_key]}, found {len(matches)}")
        path, size = matches[0]
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid metadata size for {component}")
        files[component] = {"repository": REPOS[repo_key], "path": path,
                            "revision": revisions[repo_key], "bytes": size}

    available = DISK_FREE - RESERVE
    packs = {}
    for name, components in PACKS.items():
        total = sum(files[key]["bytes"] for key in components)
        packs[name] = {
            "components": list(components), "weights_bytes": total,
            "available_disk_after_reserve_bytes": available,
            "weights_fit_disk_only": total <= available,
            "bytes_remaining_after_reserve": available - total,
            "additional_runtime_space_accounted_for": False,
            "runtime_feasibility": "not tested",
        }
    return {
        "source_kaggle_probe_run": "42128365700",
        "measured_free_disk_bytes": DISK_FREE,
        "disk_reserve_bytes": RESERVE,
        "kaggle_gpu_count": 2,
        "vram_per_gpu_bytes": 15_636_037_632,
        "repositories": {key: {"id": REPOS[key], "revision": revisions[key],
                                "gated": gated[key]} for key in REPOS},
        "files": files, "packs": packs,
        "gpu_used": False, "model_downloaded": False, "video_generated": False,
        "warning": "Weights-only disk fit does not prove a runnable pack. ComfyUI, temporary downloads, "
                   "swap, cache, extra model patches, license access, T4/CUDA compatibility, "
                   "VRAM and inference speed remain untested. Some repositories require "
                   "separate license acceptance. Do not fetch weights automatically.",
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
        revisions[key] = info.sha
        gated[key] = bool(info.gated)
        repo_files[key] = {
            item.path: item.size for item in api.list_repo_tree(repo, recursive=True,
                                                                 revision=info.sha)
            if isinstance(item, RepoFile)
        }
    report = evaluate(repo_files, revisions, gated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, pack in report["packs"].items():
        print(f"{name}: {pack['weights_bytes'] / 1e9:.2f} GB; "
              f"weights fit disk only: {pack['weights_fit_disk_only']}; "
              f"remaining: {pack['bytes_remaining_after_reserve'] / 1e9:.2f} GB")
    print("No weights downloaded. No Kaggle GPU used. Runtime compatibility NOT tested.")


if __name__ == "__main__":
    main()
