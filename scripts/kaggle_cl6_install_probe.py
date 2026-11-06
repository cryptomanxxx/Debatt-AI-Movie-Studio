"""Bounded, opt-in Kaggle direct-dependency installation probe; no LTX models."""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.kaggle_gpu_probe import MAX_POLLS, POLL_SECONDS, execute

SLUG = "debatt-ai-movie-cl6-dependency-probe"
TITLE = "Debatt AI Movie CL6 Dependency Probe"
RESULTS = Path("kaggle-cl6-results")
SUBMISSION = Path("kaggle-cl6-submission")

KERNEL_SOURCE = r'''import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
import torch

run_id = __RUN_ID__
git_sha = __GIT_SHA__
working = Path("/kaggle/working")
# CL-4 missing direct requirements; no transitive dependency installation.
requirements = [
    "comfyui-frontend-package==1.58.5",
    "comfyui-workflow-templates==0.11.102",
    "comfyui-embedded-docs==0.5.16",
    "torchsde",
    "av>=17.0.0",
    "comfy-aimdo==0.6.0",
    "comfy-kitchen==0.2.41",
    "simpleeval>=1.0.0",
    "spandrel",
    "pydantic-settings~=2.0",
    "comfy-angle",
    "gguf>=0.13.0",
]
# Preserve room for the estimated Q2 model weights plus 3 GB reserve.
MIN_FREE_DISK_BYTES = 19_300_000_000
before_bytes = shutil.disk_usage(working).free
torch_before = torch.__version__
cuda_before = torch.version.cuda
steps = []
for requirement in requirements:
    if shutil.disk_usage(working).free < MIN_FREE_DISK_BYTES:
        steps.append({"requirement": requirement, "error": "disk safety threshold before install"})
        break
    command = [sys.executable, "-m", "pip", "install", "--no-input", "--no-deps",
               "--no-cache-dir", "--only-binary=:all:", "--disable-pip-version-check",
               "--timeout", "15", "--retries", "1", requirement]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        item = {"requirement": requirement, "ok": result.returncode == 0,
                "output_tail": (result.stdout + "\n" + result.stderr)[-900:]}
    except subprocess.TimeoutExpired:
        item = {"requirement": requirement, "ok": False, "error": "120s timeout"}
    item["disk_free_after_bytes"] = shutil.disk_usage(working).free
    steps.append(item)
    if not item.get("ok") or item["disk_free_after_bytes"] < MIN_FREE_DISK_BYTES:
        break

devices = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    devices.append({"name": p.name, "vram_bytes": p.total_memory,
                    "compute_capability": [p.major, p.minor]})
successful = (len(steps) == len(requirements)
              and all(item.get("ok") for item in steps)
              and shutil.disk_usage(working).free >= MIN_FREE_DISK_BYTES
              and torch.__version__ == torch_before
              and torch.version.cuda == cuda_before)
report = {
    "run_id": run_id, "git_sha": git_sha, "direct_requirement_install_steps": steps,
    "install_completed": successful,
    "disk_free_before_bytes": before_bytes,
    "disk_free_after_bytes": shutil.disk_usage(working).free,
    "disk_safety_min_bytes": MIN_FREE_DISK_BYTES,
    "torch_before": torch_before, "torch_after": torch.__version__,
    "cuda_before": cuda_before, "cuda_after": torch.version.cuda,
    "gpu_devices": devices,
    "no_deps": True, "no_cache": True,
    "full_comfyui_installed": False, "comfyui_started": False,
    "model_downloaded": False, "ltx_loaded": False, "video_generated": False,
}
with (working / "cl6-install-probe.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("PROGRESS run=" + run_id + " phase=cl6_direct_dependency_install_complete", flush=True)
if not successful:
    raise RuntimeError("Direct dependency install incomplete or safety threshold exceeded")
'''


def published_kernel(output, username):
    matches = re.findall(r"https://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)", output)
    if len(matches) != 1 or matches[0][0].lower() != username.lower() or matches[0][1] != SLUG:
        raise ValueError("Unexpected Kaggle notebook URL")
    return f"{matches[0][0]}/{matches[0][1]}"


def validate_report(report, sha, run_id):
    if (report.get("git_sha") != sha or report.get("run_id") != run_id
            or report.get("install_completed") is not True
            or report.get("no_deps") is not True or report.get("no_cache") is not True
            or report.get("torch_before") != report.get("torch_after")
            or report.get("cuda_before") != report.get("cuda_after")
            or report.get("disk_free_after_bytes", 0) < 19_300_000_000
            or report.get("full_comfyui_installed") is not False
            or report.get("comfyui_started") is not False
            or report.get("model_downloaded") is not False
            or report.get("ltx_loaded") is not False
            or report.get("video_generated") is not False
            or not report.get("gpu_devices")
            or len(report.get("direct_requirement_install_steps", [])) != 12
            or not all(step.get("ok") is True for step in report["direct_requirement_install_steps"])):
        raise ValueError("Stale, incomplete or unsafe CL-6 Kaggle report")


def retrieve_output(kernel):
    try:
        result = execute(["kaggle", "kernels", "output", kernel,
                          "-p", str(RESULTS), "--force"], timeout=180)
        (RESULTS / "download.log").write_text(result.stdout, encoding="utf-8")
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        (RESULTS / "download.log").write_text(f"Output retrieval failed: {exc}\n",
                                                  encoding="utf-8")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        parser.error("Full Git commit SHA required")
    if not re.fullmatch(r"[0-9]+-[0-9]+", args.run_id):
        parser.error("Expected GitHub run ID-attempt")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        parser.error("Invalid Kaggle username")
    RESULTS.mkdir(exist_ok=True)
    SUBMISSION.mkdir(exist_ok=True)
    source = KERNEL_SOURCE.replace("__RUN_ID__", repr(args.run_id)).replace(
        "__GIT_SHA__", repr(args.sha))
    (SUBMISSION / "run.py").write_text(source, encoding="utf-8")
    metadata = {"id": f"{args.username}/{SLUG}", "title": TITLE,
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (SUBMISSION / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pushed = execute(["kaggle", "kernels", "push", "-p", str(SUBMISSION),
                      "--accelerator", "NvidiaTeslaT4", "--timeout", "900"], timeout=180)
    (RESULTS / "submission.log").write_text(pushed.stdout, encoding="utf-8")
    if (pushed.returncode or not re.search(r"kernel version \d+ (successfully )?pushed", pushed.stdout, re.I)):
        raise RuntimeError("Kaggle rejected the CL-6 notebook")
    kernel = published_kernel(pushed.stdout, args.username)
    complete = False
    for _ in range(MAX_POLLS):
        state = execute(["kaggle", "kernels", "status", kernel], timeout=60)
        with (RESULTS / "status.log").open("a", encoding="utf-8") as log:
            log.write(state.stdout + "\n")
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", state.stdout)
        if state.returncode or not match:
            retrieve_output(kernel)
            raise RuntimeError("Unrecognized Kaggle status")
        status = match.group(1)
        if status == "COMPLETE":
            complete = True
            break
        if status in {"ERROR", "FAILED", "CANCELED", "CANCELLED", "CANCELLING", "CANCELING"}:
            retrieve_output(kernel)
            raise RuntimeError(f"Kaggle installation probe failed: {status}")
        if status not in {"QUEUED", "RUNNING", "STARTING", "INITIALIZING"}:
            retrieve_output(kernel)
            raise RuntimeError(f"Unexpected Kaggle status: {status}")
        time.sleep(POLL_SECONDS)
    if not complete:
        retrieve_output(kernel)
        raise RuntimeError("Bounded Kaggle installation polling timed out")
    if not retrieve_output(kernel):
        raise RuntimeError("Could not retrieve CL-6 Kaggle report")
    report_path = RESULTS / "cl6-install-probe.json"
    if not report_path.is_file():
        raise RuntimeError("CL-6 JSON report missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report, args.sha, args.run_id)
    print("Validated CL-6 dependency installation:", json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
