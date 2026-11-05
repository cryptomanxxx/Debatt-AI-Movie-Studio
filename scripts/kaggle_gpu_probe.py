"""Submit a bounded, provenance-checked Kaggle GPU capability probe.

This intentionally does not download or execute LTX weights.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SLUG = "debatt-ai-movie-gpu-probe"
RESULTS = Path("kaggle-results")
SUBMISSION = Path("kaggle-submission")
MAX_POLLS = 20
POLL_SECONDS = 45

KERNEL_SOURCE = r'''import json
import os
import shutil
import time

import torch

run_id = __RUN_ID__
git_sha = __GIT_SHA__
devices = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    devices.append({
        "name": props.name,
        "vram_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor],
    })
free = shutil.disk_usage("/kaggle/working")
record = {
    "run_id": run_id,
    "git_sha": git_sha,
    "cuda_available": torch.cuda.is_available(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "gpu_devices": devices,
    "system_ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    "disk_free_bytes": free.free,
    "timestamp_utc": int(time.time()),
    "ltx_loaded": False,
    "video_generated": False,
}
with open("/kaggle/working/probe.json", "w", encoding="utf-8") as output:
    json.dump(record, output, indent=2)
print("PROGRESS run=" + run_id + " phase=probe_complete", flush=True)
'''


def execute(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, check=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sha", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--username", required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        p.error("Full Git commit SHA required")
    if not re.fullmatch(r"[0-9]+-[0-9]+", args.run_id):
        p.error("Expected GitHub run ID and attempt")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        p.error("Invalid Kaggle username")

    RESULTS.mkdir(exist_ok=True)
    SUBMISSION.mkdir(exist_ok=True)
    kernel_name = f"{args.username}/{SLUG}"
    source = KERNEL_SOURCE.replace("__RUN_ID__", repr(args.run_id)).replace(
        "__GIT_SHA__", repr(args.sha)
    )
    (SUBMISSION / "run.py").write_text(source, encoding="utf-8")
    metadata = {
        "id": kernel_name,
        "title": "Debatt AI Movie Studio GPU Probe",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (SUBMISSION / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    submitted = execute(["kaggle", "kernels", "push", "-p", str(SUBMISSION),
                         "--accelerator", "NvidiaTeslaT4", "--timeout", "900"], timeout=180)
    (RESULTS / "submission.log").write_text(submitted.stdout, encoding="utf-8")
    print(submitted.stdout)
    if submitted.returncode != 0 or re.search(
        r"(?im)^\s*(kernel push error|error pushing|failed:|traceback)", submitted.stdout
    ):
        raise RuntimeError("Kaggle rejected the submission")
    if not re.search(r"kernel version \d+ (successfully )?pushed", submitted.stdout, re.I):
        raise RuntimeError("Kaggle CLI did not confirm a new version")

    completed = False
    for _ in range(MAX_POLLS):
        state = execute(["kaggle", "kernels", "status", kernel_name], timeout=60)
        with (RESULTS / "status.log").open("a", encoding="utf-8") as log:
            log.write(state.stdout + "\n")
        print(state.stdout, flush=True)
        if state.returncode != 0:
            raise RuntimeError("Kaggle status command failed")
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", state.stdout)
        if not match:
            raise RuntimeError("Unexpected Kaggle worker status")
        status = match.group(1)
        if status == "COMPLETE":
            completed = True
            break
        if status in {"ERROR", "FAILED", "CANCELED", "CANCELLED", "CANCELLING", "CANCELING"}:
            raise RuntimeError(f"Kaggle kernel failed: {status}")
        if status not in {"QUEUED", "RUNNING", "STARTING", "INITIALIZING"}:
            raise RuntimeError(f"Unknown Kaggle status: {status}")
        time.sleep(POLL_SECONDS)
    if not completed:
        raise RuntimeError("Kaggle GPU probe exceeded the bounded polling window")

    output = execute(["kaggle", "kernels", "output", kernel_name,
                      "-p", str(RESULTS), "--force"], timeout=180)
    (RESULTS / "download.log").write_text(output.stdout, encoding="utf-8")
    if output.returncode != 0:
        raise RuntimeError("Failed to download Kaggle output")
    result_path = RESULTS / "probe.json"
    if not result_path.is_file():
        raise RuntimeError("Kaggle did not return probe.json")
    record = json.loads(result_path.read_text(encoding="utf-8"))
    if (record.get("git_sha") != args.sha or record.get("run_id") != args.run_id
            or not record.get("cuda_available") or not record.get("gpu_devices")
            or record.get("ltx_loaded") is not False):
        raise RuntimeError("Wrong, stale, or invalid Kaggle GPU probe result")
    print("Validated Kaggle GPU probe:", json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
