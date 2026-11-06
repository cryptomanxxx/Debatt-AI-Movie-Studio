"""CL-11: inspect upstream source text for offload features; never run ComfyUI."""
import argparse
import json
import re
import subprocess
from pathlib import Path

SOURCES = {
    "comfyui": "https://github.com/Comfy-Org/ComfyUI.git",
    "gguf": "https://github.com/city96/ComfyUI-GGUF.git",
}
PATTERNS = {
    "cli_lowvram": ("comfyui", "comfy/cli_args.py", r"--lowvram"),
    "cli_cpu": ("comfyui", "comfy/cli_args.py", r"--cpu"),
    "cli_reserve_vram": ("comfyui", "comfy/cli_args.py", r"--reserve-vram"),
    "cli_cpu_vae": ("comfyui", "comfy/cli_args.py", r"--cpu-vae"),
    "cli_disable_smart_memory": ("comfyui", "comfy/cli_args.py", r"--disable-smart-memory"),
    "cli_dynamic_vram": ("comfyui", "comfy/cli_args.py", r"dynamic.vram"),
    "model_management_offload": ("comfyui", "comfy/model_management.py", r"offload_device"),
    "model_patcher_partial_unload": ("comfyui", "comfy/model_patcher.py", r"partially_unload"),
    "gguf_loader": ("gguf", "nodes.py", r"UnetLoaderGGUF"),
    "gguf_cpu_or_offload": ("gguf", "nodes.py", r"cpu|offload"),
}
MAX_SOURCE_BYTES = 150_000_000


def inspect_sources(roots):
    """Return locations and snippets. A source match is NOT successful inference."""
    checks = {}
    for name, (repo, relative, needle) in PATTERNS.items():
        path = roots[repo] / relative
        if not path.is_file():
            checks[name] = {"present": False, "reason": "source file missing", "file": relative}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if re.search(needle, line, re.IGNORECASE):
                matches.append({"line": lineno, "snippet": line.strip()[:200]})
                if len(matches) >= 5:
                    break
        checks[name] = {"present": bool(matches), "file": relative,
                        "pattern": needle, "matches": matches,
                        "evidence_level": "source text only"}
    return checks


def checked_clone(repo, target):
    command = ["git", "clone", "--quiet", "--depth", "1", "--single-branch",
               "--filter=blob:none", SOURCES[repo], str(target)]
    subprocess.run(command, check=True, timeout=120)
    total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    if total > MAX_SOURCE_BYTES:
        raise ValueError(f"{repo} source checkout exceeded ceiling")
    revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"],
                                       text=True, timeout=10).strip()
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Unverifiable source revision")
    return {"revision": revision, "source_bytes": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    args.source_root.mkdir(parents=True, exist_ok=True)
    roots = {key: args.source_root / key for key in SOURCES}
    revisions = {repo: checked_clone(repo, roots[repo]) for repo in SOURCES}
    report = {
        "source_repositories": SOURCES, "source_revisions": revisions,
        "source_checks": inspect_sources(roots),
        "kaggle_gpu_used": False, "model_downloaded": False,
        "comfyui_executed": False, "video_generated": False,
        "verified_cpu_offload_for_ltx25_gguf": False,
        "verified_multi_gpu_model_parallelism": False,
        "verified_q2_pack_compatibility": False,
        "next_step": "Review source evidence; design bounded no-weight configuration checks. "
                     "Do not infer LTX 2.5 or multi-GPU runtime support from text matches.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("CL-11 source evidence written; no GPU, weights, or inference.")


if __name__ == "__main__":
    main()
