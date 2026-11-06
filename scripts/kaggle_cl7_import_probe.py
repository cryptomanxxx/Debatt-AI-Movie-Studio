"""CL-7: opt-in Kaggle import smoke test, without model weights or inference."""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.kaggle_cl6_install_probe import KERNEL_SOURCE as CL6_SOURCE
from scripts.kaggle_gpu_probe import MAX_POLLS, POLL_SECONDS, execute

SLUG = "debatt-ai-movie-cl7-import-probe"
RESULTS = Path("kaggle-cl7-results")
SUBMISSION = Path("kaggle-cl7-submission")

# Run only after CL-6's installation and its disk, torch and CUDA guards pass.
KERNEL_SOURCE = CL6_SOURCE + r'''
import os
import shutil
from pathlib import Path

sources = Path("/kaggle/working/cl7-sources")
sources.mkdir(exist_ok=True)

def checked_command(command, timeout=90, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True,
                                text=True, timeout=timeout)
        return {"ok": result.returncode == 0,
                "exit_code": result.returncode,
                "output_tail": (result.stdout + "\n" + result.stderr)[-2000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout {timeout}s"}

checks = {"pip_check": checked_command([sys.executable, "-m", "pip", "check"], 40)}
repositories = {
    "ComfyUI": "https://github.com/comfyanonymous/ComfyUI.git",
    "ComfyUI-GGUF": "https://github.com/city96/ComfyUI-GGUF.git",
}
for name, url in repositories.items():
    target = sources / name
    if shutil.disk_usage(sources).free < MIN_FREE_DISK_BYTES:
        checks[name] = {"ok": False, "error": "disk safety threshold"}
        continue
    checks[name] = checked_command(["git", "clone", "--depth", "1", "--single-branch",
                                     "--filter=blob:none", url, str(target)], 90)
    if target.exists():
        checks[name]["source_bytes"] = sum(p.stat().st_size for p in target.rglob("*")
                                             if p.is_file())
        if checks[name]["source_bytes"] > 150_000_000:
            checks[name] = {"ok": False, "error": "source exceeded 150MB ceiling"}
            shutil.rmtree(target)

comfy = sources / "ComfyUI"
if (comfy / "nodes.py").is_file():
    # Import smoke test in a child process, never launch the ComfyUI server.
    checks["comfy_nodes_import"] = checked_command(
        [sys.executable, "-c", "import nodes; print('ComfyUI nodes import OK')"],
        45, comfy)
else:
    checks["comfy_nodes_import"] = {"ok": False, "error": "ComfyUI source absent"}

report["cl7_checks"] = checks
report["cl7_disk_free_after_bytes"] = shutil.disk_usage(working).free
report["cl7_torch_after"] = torch.__version__
report["cl7_cuda_after"] = torch.version.cuda
report["comfyui_started"] = False
report["model_downloaded"] = False
report["ltx_loaded"] = False
report["video_generated"] = False
with (working / "cl7-import-probe.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("PROGRESS run=" + run_id + " phase=cl7_import_probe_complete", flush=True)
'''


def validate_report(report, sha, run_id):
    checks = report.get("cl7_checks")
    if (report.get("git_sha") != sha or report.get("run_id") != run_id
            or report.get("install_completed") is not True
            or report.get("no_deps") is not True or report.get("no_cache") is not True
            or len(report.get("direct_requirement_install_steps", [])) != 12
            or not all(x.get("ok") is True for x in report["direct_requirement_install_steps"])
            or not isinstance(checks, dict)
            or not {"pip_check", "ComfyUI", "ComfyUI-GGUF", "comfy_nodes_import"} <= checks.keys()
            or report.get("cl7_torch_after") != report.get("torch_before")
            or report.get("cl7_cuda_after") != report.get("cuda_before")
            or report.get("cl7_disk_free_after_bytes", 0) < 19_300_000_000
            or report.get("comfyui_started") is not False
            or report.get("model_downloaded") is not False
            or report.get("ltx_loaded") is not False
            or report.get("video_generated") is not False):
        raise ValueError("Incomplete, stale, or unsafe CL-7 report")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        parser.error("Full SHA required")
    if not re.fullmatch(r"[0-9]+-[0-9]+", args.run_id):
        parser.error("Invalid run id")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        parser.error("Invalid username")
    RESULTS.mkdir(exist_ok=True)
    SUBMISSION.mkdir(exist_ok=True)
    (SUBMISSION / "run.py").write_text(
        KERNEL_SOURCE.replace("__RUN_ID__", repr(args.run_id)).replace("__GIT_SHA__", repr(args.sha)),
        encoding="utf-8")
    metadata = {"id": f"{args.username}/{SLUG}", "title": "Debatt AI Movie CL7 Import Probe",
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (SUBMISSION / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pushed = execute(["kaggle", "kernels", "push", "-p", str(SUBMISSION),
                      "--accelerator", "NvidiaTeslaT4", "--timeout", "900"], timeout=180)
    (RESULTS / "submission.log").write_text(pushed.stdout, encoding="utf-8")
    if pushed.returncode or not re.search(r"kernel version \d+ (successfully )?pushed", pushed.stdout, re.I):
        raise RuntimeError("Kaggle rejected CL-7 submission")
    matches = re.findall(r"https://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)", pushed.stdout)
    if len(matches) != 1 or matches[0][0].lower() != args.username.lower() or matches[0][1] != SLUG:
        raise ValueError("Unexpected Kaggle notebook URL")
    kernel = "/".join(matches[0])
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
            break
        if status not in {"QUEUED", "RUNNING", "STARTING", "INITIALIZING"}:
            raise RuntimeError(f"Unexpected Kaggle status: {status}")
        time.sleep(POLL_SECONDS)
    try:
        output = execute(["kaggle", "kernels", "output", kernel,
                          "-p", str(RESULTS), "--force"], timeout=180)
        (RESULTS / "download.log").write_text(output.stdout, encoding="utf-8")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot retrieve Kaggle diagnostics: {exc}") from exc
    path = RESULTS / "cl7-import-probe.json"
    if not path.is_file():
        raise RuntimeError("CL-7 report missing; see Kaggle output")
    report = json.loads(path.read_text(encoding="utf-8"))
    validate_report(report, args.sha, args.run_id)
    if not complete:
        raise RuntimeError("Kaggle CL-7 run did not complete")
    print("Validated CL-7 import diagnostics:", json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
