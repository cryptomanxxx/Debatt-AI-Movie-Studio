"""Read-only LTX 2.5 model file-size check; never downloads weights."""
import argparse
import json
from pathlib import Path

MODEL = "Lightricks/LTX-2.5"
# Verified Kaggle probe: GitHub Actions run 42128365700.
DISK_FREE = 20_940_562_432
RESERVE = 3_000_000_000
FILES = {
    "transformer_bf16": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "transformer_int8": "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "encoder_bf16": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "encoder_int8": "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
}
PACKS = {
    "distilled_bf16": ["transformer_bf16", "encoder_bf16", "video_vae", "audio_vae"],
    "distilled_comfy_int8": ["transformer_int8", "encoder_int8", "video_vae", "audio_vae"],
}


def evaluate(sizes, revision):
    missing = set(FILES.values()) - set(sizes)
    if missing:
        raise ValueError(f"Missing official checkpoint metadata: {sorted(missing)}")
    if any(type(sizes[path]) is not int or sizes[path] <= 0 for path in FILES.values()):
        raise ValueError("Invalid checkpoint sizes")
    available = DISK_FREE - RESERVE
    packs = {}
    for name, keys in PACKS.items():
        total = sum(sizes[FILES[key]] for key in keys)
        packs[name] = {
            "download_bytes": total,
            "available_disk_with_reserve_bytes": available,
            "fits_disk": total <= available,
            "shortfall_bytes": max(total - available, 0),
            "file_bytes": {key: sizes[FILES[key]] for key in keys},
        }
    return {
        "model": MODEL, "revision": revision,
        "source_hardware_run": "42128365700",
        "measured_disk_free_bytes": DISK_FREE, "disk_reserve_bytes": RESERVE,
        "vram_per_gpu_bytes": 15_636_037_632, "gpu_count": 2,
        "packs": packs,
        "gpu_used": False, "model_downloaded": False, "video_generated": False,
        "warning": "Passing disk check does not establish GPU compatibility or runtime feasibility.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, RepoFile
    api = HfApi()
    revision = api.model_info(MODEL).sha
    if not revision:
        raise ValueError("Model revision unavailable")
    sizes = {f.path: f.size for f in api.list_repo_tree(MODEL, recursive=True, revision=revision)
             if isinstance(f, RepoFile)}
    report = evaluate(sizes, revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, pack in report["packs"].items():
        print(f"{name}: {pack['download_bytes'] / 1e9:.2f} GB; disk fit: {pack['fits_disk']}")
    print("No weights downloaded; no Kaggle GPU consumed.")


if __name__ == "__main__":
    main()
