"""Manually triggered, bounded Kaggle ComfyUI source/dependency inventory."""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.kaggle_gpu_probe import execute, MAX_POLLS, POLL_SECONDS

SLUG = "debatt-ai-movie-comfyui-preflight"
TITLE = "Debatt AI Movie ComfyUI Preflight"
RESULTS = Path("kaggle-comfy-results")
SUBMISSION = Path("kaggle-comfy-submission")

KERNEL_SOURCE = r'''import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch

run_id = __RUN_ID__
git_sha = __GIT_SHA__
work = Path("/kaggle/working")
initial_free = shutil.disk_usage(work).free
repos = {
    "comfyui": ("https://github.com/Comfy-Org/ComfyUI.git", "ComfyUI"),
    "gguf_plugin": ("https://github.com/city96/ComfyUI-GGUF.git", "ComfyUI-GGUF"),
}
MIN_DISK_AFTER_SOURCE_BYTES = 19_000_000_000
results = {}
for key, (url, directory) in repos.items():
    dest = work / directory
    clone = subprocess.run(["git", "-c", "credential.interactive=never", "clone",
                            "--quiet", "--depth", "1", url, str(dest)],
                           capture_output=True, text=True, timeout=180)
    if clone.returncode:
        results[key] = {"cloned": False, "error": clone.stderr[-1000:]}
        break
    revision = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                       text=True, timeout=15).strip()
    results[key] = {"cloned": True, "revision": revision}
    if shutil.disk_usage(work).free < MIN_DISK_AFTER_SOURCE_BYTES:
        results[key]["low_disk_abort"] = True
        break

requirements = {}
for key, (_, directory) in repos.items():
    req_file = work / directory / "requirements.txt"
    if not req_file.is_file():
        requirements[key] = {"available": False}
        continue
    # Inspect installed distributions; do not invoke a package installer.
    from packaging.requirements import Requirement
    parsed, missing, invalid = 0, [], []
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "git+")):
            continue
        try:
            requirement = Requirement(line)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            parsed += 1
            try:
                installed = importlib.metadata.version(requirement.name)
                if requirement.specifier and installed not in requirement.specifier:
                    missing.append({"name": requirement.name, "installed": installed,
                                    "required": str(requirement.specifier)})
            except importlib.metadata.PackageNotFoundError:
                missing.append({"name": requirement.name, "installed": None,
                                "required": str(requirement.specifier)})
        except Exception as exc:
            invalid.append({"line": line[:120], "error": type(exc).__name__})
    requirements[key] = {"available": True, "evaluated": parsed,
                         "missing_or_incompatible": missing, "unparsed": invalid}

free_after = shutil.disk_usage(work).free
devices = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    devices.append({"name": p.name, "vram_bytes": p.total_memory,
                    "compute_capability": [p.major, p.minor]})
report = {"run_id": run_id, "git_sha": git_sha, "source_repos": results,
          "requirements": requirements, "disk_free_before_bytes": initial_free,
          "disk_free_after_bytes": free_after,
          "disk_consumed_bytes": initial_free - free_after,
          "gpu_devices": devices, "torch_version": torch.__version__,
          "cuda_version": torch.version.cuda,
          "source_only": True, "full_comfyui_installed": False,
          "comfyui_started": False, "model_downloaded": False,
          "ltx_loaded": False, "video_generated": False}
with (work / "comfy-preflight.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("PROGRESS run=" + run_id + " phase=comfy_source_inventory_complete", flush=True)
if len(results) != 2 or not all(v.get("cloned") and not v.get("low_disk_abort")
                                for v in results.values()):
    raise RuntimeError("ComfyUI source clone failed or disk safety threshold reached")
'''


def parse_published_kernel(output, username):
    matches = re.findall(r"https://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)", output)
    if len(matches) != 1 or matches[0][0].lower() != username.lower() or matches[0][1] != SLUG:
        raise ValueError("Kaggle returned an unexpected ComfyUI notebook URL")
    return f"{matches[0][0]}/{matches[0][1]}"


def validate_report(report, sha, run_id):
    if (report.get("git_sha") != sha or report.get("run_id") != run_id
            or report.get("source_only") is not True
            or report.get("full_comfyui_installed") is not False
            or report.get("comfyui_started") is not False
            or report.get("model_downloaded") is not False
            or report.get("ltx_loaded") is not False
            or report.get("video_generated") is not False
            or not report.get("gpu_devices")
            or set(report.get("source_repos", {})) != {"comfyui", "gguf_plugin"}
            or not all(item.get("cloned") for item in report["source_repos"].values())):
        raise ValueError("Wrong, stale or unsafe ComfyUI source inventory")


def get_output(kernel):
    try:
        result = execute(["kaggle", "kernels", "output", kernel,
                          "-p", str(RESULTS), "--force"], timeout=180)
        (RESULTS / "download.log").write_text(result.stdout, encoding="utf-8")
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        (RESULTS / "download.log").write_text(f"Diagnostics retrieval failed: {exc}\n",
                                                  encoding="utf-8")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sha", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--username", required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        p.error("Expected full git commit SHA")
    if not re.fullmatch(r"[0-9]+-[0-9]+", args.run_id):
        p.error("Expected GitHub run ID and attempt")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        p.error("Invalid Kaggle username")
    RESULTS.mkdir(exist_ok=True)
    SUBMISSION.mkdir(exist_ok=True)
    (SUBMISSION / "run.py").write_text(KERNEL_SOURCE.replace("__RUN_ID__", repr(args.run_id))
                                      .replace("__GIT_SHA__", repr(args.sha)), encoding="utf-8")
    metadata = {"id": f"{args.username}/{SLUG}", "title": TITLE,
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (SUBMISSION / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    push = execute(["kaggle", "kernels", "push", "-p", str(SUBMISSION),
                    "--accelerator", "NvidiaTeslaT4", "--timeout", "900"], timeout=180)
    (RESULTS / "submission.log").write_text(push.stdout, encoding="utf-8")
    if (push.returncode or not re.search(r"kernel version \d+ (successfully )?pushed", push.stdout, re.I)):
        raise RuntimeError("Kaggle rejected the ComfyUI source inventory notebook")
    kernel = parse_published_kernel(push.stdout, args.username)
    complete = False
    for _ in range(MAX_POLLS):
        state = execute(["kaggle", "kernels", "status", kernel], timeout=60)
        with (RESULTS / "status.log").open("a", encoding="utf-8") as f:
            f.write(state.stdout + "\n")
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", state.stdout)
        if state.returncode or not match:
            get_output(kernel)
            raise RuntimeError("Unexpected Kaggle status; diagnostics retrieved if available")
        status = match.group(1)
        if status == "COMPLETE":
            complete = True
            break
        if status in {"ERROR", "FAILED", "CANCELED", "CANCELLED", "CANCELLING", "CANCELING"}:
            get_output(kernel)
            raise RuntimeError(f"Kaggle preflight failed: {status}")
        if status not in {"QUEUED", "RUNNING", "STARTING", "INITIALIZING"}:
            get_output(kernel)
            raise RuntimeError(f"Unexpected Kaggle status: {status}")
        time.sleep(POLL_SECONDS)
    if not complete:
        get_output(kernel)
        raise RuntimeError("Bounded ComfyUI preflight polling timed out")
    if not get_output(kernel):
        raise RuntimeError("Could not download Kaggle preflight output")
    path = RESULTS / "comfy-preflight.json"
    if not path.is_file():
        raise RuntimeError("ComfyUI source inventory report missing")
    report = json.loads(path.read_text(encoding="utf-8"))
    validate_report(report, args.sha, args.run_id)
    print("Validated ComfyUI source inventory:", json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
