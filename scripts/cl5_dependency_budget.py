"""Metadata-only size estimate for missing CL-4 dependencies (no downloads)."""
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import InvalidVersion, Version

# Verified missing-or-mismatched requirements in Kaggle run 42137390963.
# The list is a snapshot, not a resolver. Kaggle's installed packages can change.
MISSING = {
    "comfyui-frontend-package": "==1.58.5",
    "comfyui-workflow-templates": "==0.11.102",
    "comfyui-embedded-docs": "==0.5.16",
    "torchsde": "",
    "av": ">=17.0.0",
    "comfy-aimdo": "==0.6.0",
    "comfy-kitchen": "==0.2.41",
    "simpleeval": ">=1.0.0",
    "spandrel": "",
    "pydantic-settings": "~=2.0",
    "comfy-angle": "",
    "gguf": ">=0.13.0",
}

# Approximation from CL-2 log, *not* an exact byte count; do not use to
# approve downloading model files.
Q2_WEIGHT_ESTIMATE_BYTES = 16_230_000_000
KAGGLE_FREE_DISK_BYTES = 20_940_554_240
RESERVE_BYTES = 3_000_000_000
CL4_SOURCE_BYTES = 67_420_160


def choose_compatible_wheel(releases, specifier, tags):
    """Pick a release's smallest compatible non-yanked wheel, or return None."""
    from packaging.specifiers import SpecifierSet
    expected = SpecifierSet(specifier)
    found = []
    for version, files in releases.items():
        try:
            parsed_version = Version(version)
        except InvalidVersion:
            continue
        if parsed_version.is_prerelease or parsed_version.is_devrelease:
            continue
        if not expected.contains(parsed_version, prereleases=False):
            continue
        compatible = []
        for candidate in files:
            if candidate.get("packagetype") != "bdist_wheel" or candidate.get("yanked"):
                continue
            try:
                _, _, _, wheel_tags = parse_wheel_filename(candidate["filename"])
            except (ValueError, KeyError):
                continue
            if wheel_tags.isdisjoint(tags):
                continue
            size = candidate.get("size")
            if type(size) is int and size > 0:
                compatible.append((size, candidate["filename"]))
        if compatible:
            size, filename = min(compatible)
            found.append((parsed_version, size, filename))
    if not found:
        return None
    version, size, filename = max(found, key=lambda x: x[0])
    return {"version": str(version), "wheel_bytes": size, "wheel_filename": filename}


def report_budget(package_data, errors, tag_names):
    """Estimate *direct wheel download bytes*, excluding dependencies and caches."""
    rows, total, unknown = {}, 0, []
    for package, spec in MISSING.items():
        if package in errors:
            rows[package] = {"requirement": spec, "error": errors[package]}
            unknown.append(package)
            continue
        wheel = choose_compatible_wheel(package_data[package], spec, tag_names)
        rows[package] = {"requirement": spec, "selection": wheel}
        if wheel is None:
            unknown.append(package)
        else:
            total += wheel["wheel_bytes"]
    budget = KAGGLE_FREE_DISK_BYTES - Q2_WEIGHT_ESTIMATE_BYTES - RESERVE_BYTES
    return {
        "source_cl4_run": "42137390963", "source_cl2_run": "42131530963",
        "kaggle_disk_free_bytes": KAGGLE_FREE_DISK_BYTES,
        "q2_model_weight_estimate_bytes": Q2_WEIGHT_ESTIMATE_BYTES,
        "reserve_bytes": RESERVE_BYTES,
        "remaining_before_runtime_bytes_estimate": budget,
        "source_checkout_bytes_if_retained": CL4_SOURCE_BYTES,
        "packages": rows, "direct_wheel_bytes_known": total,
        "unknown_wheel_packages": unknown,
        "direct_wheels_fit_estimated_margin": (total <= budget) if not unknown else None,
        "python_compatibility_environment": "GitHub Ubuntu CPython 3.11 proxy, not measured Kaggle tags",
        "dependency_resolution_performed": False,
        "model_downloaded": False, "gpu_used": False,
        "conclusion": "INCOMPLETE ESTIMATE — do not authorize model download or inference",
        "warnings": [
            "A wheel's compressed download size is not its installed size.",
            "This excludes recursive dependencies, cache, unpacking, swap, outputs, ComfyUI extras and model patches.",
            "Selected wheels use GitHub's Python/Linux tags; Kaggle compatibility was not verified.",
            "Kaggle's observed installed-package state is a snapshot and may change.",
            "The Q2 weight size is rounded; exact Hugging Face metadata must be used for a final gate.",
        ],
    }


def fetch_releases(package):
    # Official PyPI JSON metadata endpoint; JSON only, never wheel downloads.
    req = urllib.request.Request(
        f"https://pypi.org/pypi/{package}/json",
        headers={"User-Agent": "debatt-ai-movie-studio-cl5-metadata/1.0",
                 "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError(f"PyPI HTTP {response.status}")
        payload = response.read(8_000_001)
        if len(payload) > 8_000_000:
            raise ValueError("PyPI project metadata exceeds 8 MB")
    return json.loads(payload)["releases"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, errors = {}, {}
    for package in MISSING:
        try:
            data[package] = fetch_releases(package)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, RuntimeError) as exc:
            errors[package] = type(exc).__name__
    report = report_budget(data, errors, set(sys_tags()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Known direct compatible wheel downloads:", report["direct_wheel_bytes_known"])
    print("Unresolved packages:", report["unknown_wheel_packages"])
    print("This is NOT an installed-size or complete-environment estimate; no GPU, no model weights.")


if __name__ == "__main__":
    main()
