"""Bounded, explicitly opted-in Kaggle GGUF import probe. No model files."""
import argparse
import json
import re
import sys
from pathlib import Path

from scripts.kaggle_gpu_probe import execute, MAX_POLLS, POLL_SECONDS

import time

SLUG = "debatt-ai-movie-gguf-probe"
RESULTS = Path("kaggle-gguf-results")
SUBMISSION = Path("kaggle-gguf-submission")

KERNEL_SOURCE = r'''import json
import os
import shutil
import subprocess
import sys
import torch

run_id = __RUN_ID__
git_sha = __GIT_SHA__
disk_before = shutil.disk_usage("/kaggle/working").free
# Only a small metadata-format library; no ComfyUI or model weights.
install = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                          "--no-input", "--no-deps", "--timeout", "15", "--retries", "1",
                          "gguf>=0.13,<1"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, timeout=180)
gguf_version = None
if install.returncode == 0:
    import importlib.metadata
    import gguf
    gguf_version = importlib.metadata.version("gguf")
devices = []
for index in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(index)
    devices.append({"name": p.name, "vram_bytes": p.total_memory,
                    "capability": [p.major, p.minor]})
report = {
    "run_id": run_id, "git_sha": git_sha, "gguf_install_succeeded": install.returncode == 0,
    "gguf_version": gguf_version, "gguf_install_log_tail": install.stdout[-2000:],
    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    "gpu_devices": devices, "disk_free_before_bytes": disk_before,
    "disk_free_after_bytes": shutil.disk_usage("/kaggle/working").free,
    "comfyui_installed": False, "model_downloaded": False,
    "ltx_loaded": False, "video_generated": False,
}
with open("/kaggle/working/gguf-probe.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("PROGRESS run=" + run_id + " phase=gguf_probe_complete", flush=True)
if install.returncode != 0:
    raise RuntimeError("Minimal gguf package installation failed; see report")
'''


def validate_report(record, sha, run_id):
    if (record.get("git_sha") != sha or record.get("run_id") != run_id
            or record.get("gguf_install_succeeded") is not True
            or not record.get("gguf_version") or not record.get("gpu_devices")
            or record.get("model_downloaded") is not False
            or record.get("ltx_loaded") is not False
            or record.get("video_generated") is not False):
        raise ValueError("Invalid or stale GGUF environment probe result")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sha", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--username", required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        p.error("Full git commit SHA required")
    if not re.fullmatch(r"[0-9]+-[0-9]+", args.run_id):
        p.error("Expected GitHub run ID and attempt")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        p.error("Invalid Kaggle username")
    RESULTS.mkdir(exist_ok=True)
    SUBMISSION.mkdir(exist_ok=True)
    kernel = f"{args.username}/{SLUG}"
    source = KERNEL_SOURCE.replace("__RUN_ID__", repr(args.run_id)).replace(
        "__GIT_SHA__", repr(args.sha))
    (SUBMISSION / "run.py").write_text(source, encoding="utf-8")
    metadata = {
        "id": kernel, "title": "Debatt AI Movie GGUF Environment Probe",
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (SUBMISSION / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pushed = execute(["kaggle", "kernels", "push", "-p", str(SUBMISSION),
                      "--accelerator", "NvidiaTeslaT4", "--timeout", "900"], timeout=180)
    (RESULTS / "submission.log").write_text(pushed.stdout, encoding="utf-8")
    if (pushed.returncode or re.search(r"(?im)^\s*(kernel push error|error pushing|failed:|traceback)", pushed.stdout)
            or not re.search(r"kernel version \d+ (successfully )?pushed", pushed.stdout, re.I)):
        raise RuntimeError("Kaggle did not confirm GGUF probe submission")
    complete = False
    for _ in range(MAX_POLLS):
        state = execute(["kaggle", "kernels", "status", kernel], timeout=60)
        with (RESULTS / "status.log").open("a", encoding="utf-8") as f:
            f.write(state.stdout + "\n")
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", state.stdout)
        if state.returncode or not match:
            raise RuntimeError("Unrecognized Kaggle status")
        status = match.group(1)
        if status == "COMPLETE":
            complete = True
            break
        if status in {"ERROR", "FAILED", "CANCELED", "CANCELLED", "CANCELLING", "CANCELING"}:
            raise RuntimeError(f"Kaggle GGUF probe failed: {status}")
        if status not in {"QUEUED", "RUNNING", "STARTING", "INITIALIZING"}:
            raise RuntimeError(f"Unknown Kaggle status: {status}")
        time.sleep(POLL_SECONDS)
    if not complete:
        raise RuntimeError("Bounded GGUF probe polling timed out")
    output = execute(["kaggle", "kernels", "output", kernel, "-p", str(RESULTS), "--force"], timeout=180)
    if output.returncode:
        raise RuntimeError("Failed to retrieve Kaggle GGUF probe output")
    path = RESULTS / "gguf-probe.json"
    if not path.is_file():
        raise RuntimeError("GGUF report not returned")
    record = json.loads(path.read_text(encoding="utf-8"))
    validate_report(record, args.sha, args.run_id)
    print("Validated GGUF environment report:", json.dumps(record, indent=2))


if __name__ == "__main__":
    sys.exit(main())
