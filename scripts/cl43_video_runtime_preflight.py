"""CL-43: actionable local ComfyUI/LTX-2.5 clip preflight, no downloads or GPU.

This does NOT use the project's existing Q2_K GGUF probe as a drop-in
replacement for the official INT8 ConvRot LTX-2.5 pipeline.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ASSETS = (
    ("transformer", "diffusion_models", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
    ("text_encoder", "text_encoders", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    ("video_vae", "vae", "ltx-2.5-video-vae-conv-bf16.safetensors"),
    ("audio_vae", "vae", "ltx-2.5-audio-vae-bf16.safetensors"),
)
WORKFLOW = "video_ltx2_5_flf2v"
OFFICIAL_GUIDE = "https://docs.ltx.io/open-source-model/integration-tools/comfy-ui/"
MIN_FREE_BYTES = 20_000_000_000


def preflight(comfy_root):
    """Inspect configured files and machine prerequisites without side effects."""
    comfy_root = Path(comfy_root).expanduser().resolve()
    files = []
    for role, subdir, filename in ASSETS:
        path = comfy_root / "models" / subdir / filename
        exists = path.is_file()
        files.append({"role": role, "relative_path": str(path.relative_to(comfy_root)),
                      "exists": exists, "size_bytes": path.stat().st_size if exists else None})
    main_file = comfy_root / "main.py"
    expected_nodes = comfy_root / "custom_nodes" / "ComfyUI-LTXVideo"
    ffmpeg_path = shutil.which("ffmpeg")
    storage_anchor = comfy_root if comfy_root.exists() else comfy_root.parent
    while not storage_anchor.exists() and storage_anchor != storage_anchor.parent:
        storage_anchor = storage_anchor.parent
    free_bytes = shutil.disk_usage(storage_anchor).free if storage_anchor.exists() else None
    # Existing checkpoints are not verified by name alone; hashes/licensing
    # and runtime compatibility still require a separate real launch test.
    ready_for_manual_workflow_import = bool(
        main_file.is_file() and expected_nodes.is_dir() and all(f["exists"] for f in files)
        and ffmpeg_path and free_bytes is not None and free_bytes >= MIN_FREE_BYTES)
    return {
        "experiment": "cl43_video_runtime_preflight",
        "runtime": "ComfyUI",
        "workflow": WORKFLOW,
        "workflow_source": "official_ComfyUI_template",
        "video_vae_template_override": "ltx-2.5-video-vae-conv-bf16.safetensors",
        "template_requires_video_vae_override": True,
        "official_documentation": OFFICIAL_GUIDE,
        "strategy": "official_lower_vram_int8_convrot_not_gguf_substitution",
        "comfy_root": str(comfy_root),
        "comfy_main_exists": main_file.is_file(),
        "ltx_video_custom_nodes_exist": expected_nodes.is_dir(),
        "ffmpeg_found": bool(ffmpeg_path),
        "free_disk_bytes": free_bytes,
        "min_free_disk_bytes": MIN_FREE_BYTES,
        "required_assets": files,
        "all_required_assets_present": all(f["exists"] for f in files),
        "ready_for_manual_workflow_import": ready_for_manual_workflow_import,
        "model_hashes_verified": False,
        "gpu_memory_fit_verified": False,
        "comfyui_workflow_executed": False,
        "video_generated": False,
        "automatic_downloads": False,
        "gpu_used": False,
        "next_action": "Verify model hashes, import First/Last-Frame template, then change its video_vae selector from BF16 DiffVAE to the Conv VAE before queuing",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=Path(os.environ.get("COMFYUI_ROOT", "ComfyUI")))
    parser.add_argument("--output", type=Path, default=Path("cl43-results/report.json"))
    args = parser.parse_args(argv)
    report = preflight(args.comfy_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
