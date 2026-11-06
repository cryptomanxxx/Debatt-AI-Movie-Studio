"""CL-8: bounded remediation of the observed CL-7 trampoline import failure."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7

SLUG = "debatt-ai-movie-cl8-dependency-probe"
RESULTS = Path("kaggle-cl8-results")
SUBMISSION = Path("kaggle-cl8-submission")

_OLD_CHECK = 'checks = {"pip_check": checked_command([sys.executable, "-m", "pip", "check"], 40)}'
_NEW_CHECK = r'''
checks = {}
# CL-7 showed torchsde -> trampoline is missing; install only this small leaf.
if shutil.disk_usage(working).free < MIN_FREE_DISK_BYTES + 150_000_000:
    checks["trampoline_install"] = {"ok": False, "error": "insufficient disk headroom"}
else:
    checks["trampoline_install"] = checked_command(
        [sys.executable, "-m", "pip", "install", "--no-input", "--no-deps",
         "--no-cache-dir", "--only-binary=:all:", "--disable-pip-version-check",
         "--timeout", "15", "--retries", "1", "trampoline"], 90)
checks["trampoline_disk_free_bytes"] = shutil.disk_usage(working).free
checks["pip_check"] = checked_command([sys.executable, "-m", "pip", "check"], 40)

# Metadata-only inspection: never download the 8 template wheels.
import urllib.request
template_packages = [
    "comfyui-workflow-templates-core", "comfyui-workflow-templates-json",
    "comfyui-workflow-templates-media-api",
    "comfyui-workflow-templates-media-assets-01",
    "comfyui-workflow-templates-media-assets-02",
    "comfyui-workflow-templates-media-image",
    "comfyui-workflow-templates-media-other",
    "comfyui-workflow-templates-media-video",
]
checks["template_wheel_metadata"] = {}
for package in template_packages:
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/" + package + "/json", timeout=5) as response:
            payload = json.load(response)
        files = [entry for entry in payload.get("urls", [])
                 if entry.get("filename", "").endswith(".whl")]
        checks["template_wheel_metadata"][package] = {
            "release": payload.get("info", {}).get("version"),
            "wheel_count": len(files),
            "wheel_min_bytes": min((entry["size"] for entry in files), default=None),
            "wheel_max_bytes": max((entry["size"] for entry in files), default=None),
            "metadata_only": True,
        }
    except (OSError, ValueError, KeyError) as exc:
        checks["template_wheel_metadata"][package] = {"error": str(exc)[:200],
                                                       "metadata_only": True}
'''


def _make_kernel():
    source = cl7.KERNEL_SOURCE
    assert source.count(_OLD_CHECK) == 1, "CL-7 source marker has changed"
    source = source.replace(_OLD_CHECK, _NEW_CHECK)
    for old, new in (
        ("cl7-sources", "cl8-sources"),
        ('"cl7_checks"', '"cl8_checks"'),
        ('"cl7_disk_free_after_bytes"', '"cl8_disk_free_after_bytes"'),
        ('"cl7_torch_after"', '"cl8_torch_after"'),
        ('"cl7_cuda_after"', '"cl8_cuda_after"'),
        ("cl7-import-probe.json", "cl8-dependency-probe.json"),
        ("cl7_import_probe_complete", "cl8_dependency_probe_complete"),
    ):
        assert old in source, f"CL-7 marker missing: {old}"
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = _make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl8_checks", {})
    if (report.get("git_sha") != sha or report.get("run_id") != run_id
            or report.get("install_completed") is not True
            or report.get("no_deps") is not True or report.get("no_cache") is not True
            or len(report.get("direct_requirement_install_steps", [])) != 12
            or not all(step.get("ok") for step in report["direct_requirement_install_steps"])
            or not isinstance(checks, dict)
            or not {"trampoline_install", "pip_check", "ComfyUI", "ComfyUI-GGUF",
                    "comfy_nodes_import", "template_wheel_metadata"} <= checks.keys()
            or not isinstance(checks.get("template_wheel_metadata"), dict)
            or len(checks["template_wheel_metadata"]) != 8
            or report.get("cl8_torch_after") != report.get("torch_before")
            or report.get("cl8_cuda_after") != report.get("cuda_before")
            or report.get("cl8_disk_free_after_bytes", 0) < 19_300_000_000
            or checks.get("trampoline_disk_free_bytes", 0) < 19_300_000_000
            or any(report.get(key) is not False for key in
                   ("comfyui_started", "model_downloaded", "ltx_loaded", "video_generated"))):
        raise ValueError("Incomplete, stale or unsafe CL-8 diagnostic report")


def main():
    # Reuse CL-7's proven GitHub Actions -> Kaggle submit/status/output machinery.
    cl7.SLUG = SLUG
    cl7.TITLE = "Debatt AI Movie CL8 Dependency Probe"
    cl7.REPORT_NAME = "cl8-dependency-probe.json"
    cl7.RESULTS = RESULTS
    cl7.SUBMISSION = SUBMISSION
    cl7.KERNEL_SOURCE = KERNEL_SOURCE
    cl7.validate_report = validate_report
    cl7.main()


if __name__ == "__main__":
    main()
