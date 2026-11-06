"""CL-9: opt-in bounded ComfyUI HTTP/GGUF startup diagnostics, no model weights."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl8_dependency_probe import KERNEL_SOURCE as CL8_SOURCE

SLUG = "debatt-ai-movie-cl9-startup-probe"
TITLE = "Debatt AI Movie CL9 Startup Probe"
REPORT_NAME = "cl9-startup-probe.json"
RESULTS = Path("kaggle-cl9-results")
SUBMISSION = Path("kaggle-cl9-submission")

_MARKER = '''finally:
    # Kaggle output downloads must contain only diagnostics, never cloned repositories.'''
_STARTUP = r'''
    # The successful CL-8 import is a precondition, not proof of HTTP startup.
    startup = {"attempted": False, "http_ok": False, "gguf_registered": False}
    checks["startup_probe"] = startup
    if checks.get("comfy_nodes_import", {}).get("ok") and (sources / "ComfyUI-GGUF").is_dir():
        import urllib.request
        addon = comfy / "custom_nodes" / "ComfyUI-GGUF"
        server_log = working / "cl9-comfy-server.log"
        server = None
        startup["attempted"] = True
        try:
            addon.parent.mkdir(exist_ok=True)
            addon.symlink_to(sources / "ComfyUI-GGUF", target_is_directory=True)
            startup["gpu_mem_before_bytes"] = [torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())]
            if shutil.disk_usage(working).free < MIN_FREE_DISK_BYTES + 150_000_000:
                startup["error"] = "disk safety threshold before startup"
            else:
                with server_log.open("wb") as log:
                    server = subprocess.Popen(
                        [sys.executable, "main.py", "--listen", "127.0.0.1",
                         "--port", "8188", "--disable-auto-launch"],
                        cwd=comfy, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                endpoint = "http://127.0.0.1:8188/object_info"
                begin = time.monotonic()
                while time.monotonic() - begin < 90:
                    if server.poll() is not None:
                        startup["error"] = "server exited before HTTP readiness"
                        break
                    if shutil.disk_usage(working).free < MIN_FREE_DISK_BYTES:
                        startup["error"] = "disk safety threshold during server startup"
                        break
                    try:
                        with urllib.request.urlopen(endpoint, timeout=3) as response:
                            nodes_data = json.load(response)
                        startup["http_ok"] = response.status == 200
                        names = [name for name in nodes_data if "gguf" in name.lower()]
                        startup["gguf_node_names"] = names[:30]
                        startup["gguf_registered"] = bool(names)
                        break
                    except (OSError, ValueError):
                        time.sleep(1)
                else:
                    startup["error"] = "90-second HTTP startup timeout"
            startup["gpu_mem_after_bytes"] = [torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())]
        except (OSError, ValueError) as exc:
            startup["error"] = (type(exc).__name__ + ": " + str(exc))[:500]
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
            if server_log.is_file():
                startup["server_log_tail"] = server_log.read_bytes()[-12000:].decode("utf-8", "replace")
                server_log.unlink()
            # The Git checkout is removed by the outer finally block.
    else:
        startup["error"] = "CL-8 imports or GGUF checkout unavailable"
'''


def make_kernel():
    source = CL8_SOURCE
    assert source.count(_MARKER) == 1
    source = source.replace(_MARKER, _STARTUP + _MARKER)
    assert source.count('report["comfyui_started"] = False') == 1
    source = source.replace('report["comfyui_started"] = False',
                            'report["comfyui_started"] = checks["startup_probe"]["http_ok"]')
    for old, new in (
        ("cl8-sources", "cl9-sources"),
        ('"cl8_checks"', '"cl9_checks"'),
        ('"cl8_disk_free_after_bytes"', '"cl9_disk_free_after_bytes"'),
        ('"cl8_torch_after"', '"cl9_torch_after"'),
        ('"cl8_cuda_after"', '"cl9_cuda_after"'),
        ("cl8-dependency-probe.json", REPORT_NAME),
        ("cl8_dependency_probe_complete", "cl9_startup_probe_complete"),
    ):
        assert old in source, f"Missing CL-8 source marker {old}"
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl9_checks")
    if (report.get("git_sha") != sha or report.get("run_id") != run_id
            or report.get("install_completed") is not True
            or report.get("no_deps") is not True or report.get("no_cache") is not True
            or not isinstance(checks, dict)
            or not {"trampoline_install", "ComfyUI", "ComfyUI-GGUF", "comfy_nodes_import",
                    "startup_probe"} <= checks.keys()
            or not isinstance(checks.get("startup_probe"), dict)
            or report.get("comfyui_started") is not checks["startup_probe"].get("http_ok")
            or not isinstance(checks["startup_probe"].get("gguf_registered"), bool)
            or report.get("cl9_torch_after") != report.get("torch_before")
            or report.get("cl9_cuda_after") != report.get("cuda_before")
            or report.get("cl9_disk_free_after_bytes", 0) < 19_300_000_000
            or any(report.get(key) is not False for key in
                   ("model_downloaded", "ltx_loaded", "video_generated"))):
        raise ValueError("Incomplete, stale, or unsafe CL-9 report")


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
