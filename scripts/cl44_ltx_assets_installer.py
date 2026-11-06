"""CL-44: explicit, size-checked installation of official LTX-2.5 ComfyUI assets.

Dry-run by default. Real downloads require a pinned Hugging Face revision,
explicit CLI confirmation, space guards, and the huggingface_hub dependency.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from scripts.cl43_video_runtime_preflight import ASSETS, preflight

REPOSITORY = "Lightricks/LTX-2.5"
CONFIRM = "DOWNLOAD_LTX25_MODELS"
EXTRA_FREE_BYTES = 10_000_000_000
MAX_TOTAL_BYTES = 80_000_000_000


def build_manifest(comfy_root):
    root = Path(comfy_root).expanduser().resolve()
    return [{"role": role, "repository_path": f"{subdir}/{filename}",
             "destination": str(root / "models" / subdir / filename)}
            for role, subdir, filename in ASSETS]


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(comfy_root, revision, confirm, *, api=None, download=None):
    """Validate *all* remote file sizes before any remote download begins."""
    if confirm != CONFIRM or not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Requires exact confirmation and pinned 40-hex revision")
    root = Path(comfy_root).expanduser().resolve()
    if not (root / "main.py").is_file():
        raise ValueError("Existing ComfyUI installation required")
    if api is None or download is None:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        download = hf_hub_download
    info = api.model_info(REPOSITORY, revision=revision, files_metadata=True)
    if getattr(info, "sha", None) != revision:
        raise ValueError("Remote model commit does not match pinned revision")
    siblings = {f.rfilename: f for f in info.siblings}
    manifest = build_manifest(root)
    required_total = 0
    missing_total = 0
    for item in manifest:
        sibling = siblings.get(item["repository_path"])
        size = getattr(sibling, "size", None)
        if not isinstance(size, int) or not (1_000_000 <= size <= MAX_TOTAL_BYTES):
            raise ValueError("Missing or unbounded asset size: " + item["repository_path"])
        lfs = getattr(sibling, "lfs", None)
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Missing LFS SHA-256: " + item["repository_path"])
        item["expected_size_bytes"] = size
        item["expected_sha256"] = digest
        required_total += size
        destination = Path(item["destination"])
        item["already_verified"] = (destination.is_file()
                                    and destination.stat().st_size == size
                                    and _hash_file(destination) == digest)
        if not item["already_verified"]:
            missing_total += size
    if required_total > MAX_TOTAL_BYTES:
        raise ValueError("Model stack exceeds configured disk budget")
    # local_dir stores Hub transfer metadata and staging files below models/.
    # Asset directories may be independent mounts or symlinks: budget each
    # actual destination device, not merely the ComfyUI installation device.
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    device_paths = {models.stat().st_dev: models}
    pending_by_device = {}
    for item in manifest:
        if item["already_verified"]:
            continue
        destination = Path(item["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        actual_parent = destination.parent.resolve()
        device = actual_parent.stat().st_dev
        device_paths.setdefault(device, actual_parent)
        pending_by_device[device] = pending_by_device.get(device, 0) + item["expected_size_bytes"]
    for device, path in device_paths.items():
        # On a different destination mount, Hub staging still needs room for
        # the largest single transfer on the models filesystem.
        pending = pending_by_device.get(device, 0)
        if device == models.stat().st_dev and missing_total > pending:
            pending += max((item["expected_size_bytes"] for item in manifest
                            if not item["already_verified"]), default=0)
        if shutil.disk_usage(path).free < pending + EXTRA_FREE_BYTES:
            raise ValueError("Not enough free space on actual model destination or staging filesystem")

    completed = []
    for item in manifest:
        destination = Path(item["destination"])
        if item["already_verified"]:
            completed.append(item["role"])
            continue
        # The Hub library handles resumable transfer and cache bookkeeping.
        # Never log access tokens or write credentials into the report.
        # Invalidate stale Hub local_dir metadata as well: a previous
        # size-correct but SHA-invalid file must never be reused on retry.
        download(repo_id=REPOSITORY, filename=item["repository_path"],
                 revision=revision, local_dir=str(models), force_download=True)
        if (not destination.is_file()
                or destination.stat().st_size != item["expected_size_bytes"]
                or _hash_file(destination) != item["expected_sha256"]):
            raise ValueError("Downloaded asset failed SHA-256/size verification: " + item["role"])
        completed.append(item["role"])
    return {"repository": REPOSITORY, "pinned_revision": revision,
            "installed_and_sha256_verified_roles": completed,
            "verified_total_bytes": required_total,
            "video_generated": False, "gpu_used": False,
            "workflow_template_override_still_required": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--confirm", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("cl44-results/report.json"))
    args = parser.parse_args(argv)
    if args.apply:
        report = install(args.comfy_root, args.revision, args.confirm)
    else:
        report = {"mode": "dry_run", "repository": REPOSITORY,
                  "required_assets": build_manifest(args.comfy_root),
                  "current_preflight": preflight(args.comfy_root),
                  "explicit_download_authorized": False,
                  "gpu_used": False, "video_generated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
