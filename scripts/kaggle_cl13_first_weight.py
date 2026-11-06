"""CL-13: explicitly opt-in, bounded download of ONE LTX 2.5 Q2 checkpoint."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl8_dependency_probe import KERNEL_SOURCE as CL8_SOURCE

SLUG = "debatt-ai-movie-cl13-first-weight"
TITLE = "Debatt AI Movie CL13 First Q2 Weight"
REPORT_NAME = "cl13-first-weight.json"
RESULTS = Path("kaggle-cl13-results")
SUBMISSION = Path("kaggle-cl13-submission")

_MARKER = '''finally:
    # Kaggle output downloads must contain only diagnostics, never cloned repositories.'''
_DOWNLOAD = r'''
    # Only one transformer file. No text encoder, VAE, GPU loading or inference.
    import hashlib
    import signal
    import urllib.request
    from urllib.error import HTTPError, URLError

    repo_id = "unsloth/LTX-2.5-GGUF"
    filename = "ltx-2.5-22b-distilled-Q2_K.gguf"
    # Kaggle publishes /kaggle/working as outputs. Keep any partial model OUTSIDE it.
    temporary = Path("/kaggle/temp")
    temporary.mkdir(parents=True, exist_ok=True)
    stage = temporary / "cl13-q2-download.part"
    first = {"attempted": False, "download_complete": False,
             "sha256_verified": False, "stage_removed": False,
             "target_repository": repo_id, "target_filename": filename}
    checks["first_weight"] = first
    digest = hashlib.sha256()
    try:
        from huggingface_hub import HfApi, RepoFile, hf_hub_url
        api = HfApi()
        info = api.model_info(repo_id)
        if not info.sha or info.gated:
            raise ValueError("Repository revision missing or gated; no automated license bypass")
        matches = [entry for entry in api.list_repo_tree(repo_id, recursive=True,
                                                           revision=info.sha)
                   if isinstance(entry, RepoFile) and entry.path.rsplit("/", 1)[-1] == filename]
        if len(matches) != 1 or type(matches[0].size) is not int or matches[0].size <= 0:
            raise ValueError("Ambiguous or missing Q2 checkpoint metadata")
        entry = matches[0]
        first["revision"] = info.sha
        first["expected_bytes"] = entry.size
        first["path"] = entry.path
        lfs = entry.lfs or {}
        checksum = lfs.get("sha256") if isinstance(lfs, dict) else None
        if not checksum or len(checksum) != 64 or not all(c in "0123456789abcdef" for c in checksum):
            raise ValueError("No verifiable LFS SHA-256: refuse large download")
        first["expected_sha256"] = checksum
        # Leave 3 GB disk reserve and an extra 1 GB cushion for staging/runtime.
        if shutil.disk_usage(temporary).free - entry.size < 4_000_000_000:
            raise ValueError("Insufficient disk space for file plus 4 GB reserve")
        if entry.size > 13_000_000_000:
            raise ValueError("Single-file download cap 13 GB exceeded")
        first["attempted"] = True
        url = hf_hub_url(repo_id, entry.path, revision=info.sha)
        downloaded = 0
        def deadline_exceeded(signum, frame):
            raise TimeoutError("12-minute weight download deadline")
        previous_handler = signal.signal(signal.SIGALRM, deadline_exceeded)
        signal.setitimer(signal.ITIMER_REAL, 720)
        try:
            with urllib.request.urlopen(url, timeout=30) as response, stage.open("xb") as out:
                if response.status != 200:
                    raise ValueError("Unexpected download HTTP status")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != entry.size:
                    raise ValueError("Content-Length mismatch; download aborted")
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > entry.size or shutil.disk_usage(temporary).free < 4_000_000_000:
                        raise ValueError("Download size or disk-reserve limit exceeded")
                    out.write(chunk)
                    digest.update(chunk)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        first["downloaded_bytes"] = downloaded
        first["sha256_verified"] = downloaded == entry.size and digest.hexdigest() == checksum
        first["download_complete"] = first["sha256_verified"]
        if not first["sha256_verified"]:
            raise ValueError("Downloaded checkpoint failed size or SHA-256 verification")
    except Exception as exc:
        first["error"] = (type(exc).__name__ + ": " + str(exc))[:500]
    finally:
        # Never let Kaggle output retrieval archive giant checkpoints.
        if stage.is_file():
            stage.unlink()
        first["stage_removed"] = not stage.exists()
        first["disk_free_after_cleanup_bytes"] = shutil.disk_usage(temporary).free
'''


def make_kernel():
    source = CL8_SOURCE
    assert source.count(_MARKER) == 1
    source = source.replace(_MARKER, _DOWNLOAD + _MARKER)
    assert source.count('report["model_downloaded"] = False') == 1
    source = source.replace('report["model_downloaded"] = False',
                            'report["model_downloaded"] = checks["first_weight"]["download_complete"]')
    for old, new in (
        ("cl8-sources", "cl13-sources"),
        ('"cl8_checks"', '"cl13_checks"'),
        ('"cl8_disk_free_after_bytes"', '"cl13_disk_free_after_bytes"'),
        ('"cl8_torch_after"', '"cl13_torch_after"'),
        ('"cl8_cuda_after"', '"cl13_cuda_after"'),
        ("cl8-dependency-probe.json", REPORT_NAME),
        ("cl8_dependency_probe_complete", "cl13_first_weight_complete"),
    ):
        assert old in source
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl13_checks")
    if (report.get("git_sha") != sha or report.get("run_id") != run_id
            or report.get("install_completed") is not True
            or report.get("no_deps") is not True or report.get("no_cache") is not True
            or len(report.get("direct_requirement_install_steps", [])) != 12
            or not all(x.get("ok") is True for x in report["direct_requirement_install_steps"])
            or not isinstance(checks, dict)
            or not isinstance(checks.get("first_weight"), dict)
            or checks["first_weight"].get("target_repository") != "unsloth/LTX-2.5-GGUF"
            or checks["first_weight"].get("target_filename") != "ltx-2.5-22b-distilled-Q2_K.gguf"
            or checks["first_weight"].get("attempted") is not True
            or checks["first_weight"].get("sha256_verified") is not True
            or checks["first_weight"].get("download_complete") is not True
            or checks["first_weight"].get("stage_removed") is not True
            or checks["first_weight"].get("disk_free_after_cleanup_bytes", 0) < 4_000_000_000
            or report.get("model_downloaded") is not checks["first_weight"].get("download_complete")
            or report.get("cl13_torch_after") != report.get("torch_before")
            or report.get("cl13_cuda_after") != report.get("cuda_before")
            or report.get("cl13_disk_free_after_bytes", 0) < 19_300_000_000
            or any(report.get(key) is not False for key in ("ltx_loaded", "video_generated"))):
        raise ValueError("Incomplete, stale, or unsafe CL-13 report")


def main():
    cl7.SLUG = SLUG
    cl7.TITLE = TITLE
    cl7.REPORT_NAME = REPORT_NAME
    cl7.RESULTS = RESULTS
    cl7.SUBMISSION = SUBMISSION
    cl7.KERNEL_SOURCE = KERNEL_SOURCE
    cl7.validate_report = validate_report
    cl7.main()


if __name__ == "__main__":
    main()
