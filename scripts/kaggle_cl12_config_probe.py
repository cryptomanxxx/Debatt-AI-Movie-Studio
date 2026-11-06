"""CL-12: opt-in no-model ComfyUI memory-configuration startup comparison."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl8_dependency_probe import KERNEL_SOURCE as CL8_SOURCE

SLUG = "debatt-ai-movie-cl12-config-probe"
TITLE = "Debatt AI Movie CL12 Config Probe"
REPORT_NAME = "cl12-config-probe.json"
RESULTS = Path("kaggle-cl12-results")
SUBMISSION = Path("kaggle-cl12-submission")

_MARKER = '''finally:
    # Kaggle output downloads must contain only diagnostics, never cloned repositories.'''
_CONFIGS = r'''
    # No checkpoints or queues: only start a local HTTP server for each variant.
    import urllib.request
    import psutil
    configs = [
        ("default", []),
        ("lowvram_no_dynamic", ["--disable-dynamic-vram", "--lowvram"]),
        ("cpu_vae", ["--cpu-vae"]),
        ("aggressive_ram_offload", ["--disable-smart-memory"]),
    ]
    checks["configuration_probes"] = {}
    addon = comfy / "custom_nodes" / "ComfyUI-GGUF"
    if (comfy / "main.py").is_file() and (sources / "ComfyUI-GGUF").is_dir() \
            and checks.get("comfy_nodes_import", {}).get("ok"):
        addon.parent.mkdir(exist_ok=True)
        addon.symlink_to(sources / "ComfyUI-GGUF", target_is_directory=True)
        for idx, (name, flags) in enumerate(configs):
            entry = {"attempted": False, "http_ok": False, "gguf_registered": False,
                     "flags": flags, "model_loaded": False}
            checks["configuration_probes"][name] = entry
            if shutil.disk_usage(working).free < MIN_FREE_DISK_BYTES + 150_000_000:
                entry["error"] = "disk safety threshold before startup"
                continue
            if psutil.virtual_memory().available < 3_000_000_000:
                entry["error"] = "RAM safety threshold before startup"
                continue
            port = 8188 + idx
            log_path = working / ("cl12-server-" + name + ".log")
            server = None
            entry["attempted"] = True
            entry["gpu_free_before_bytes"] = [torch.cuda.mem_get_info(i)[0]
                                              for i in range(torch.cuda.device_count())]
            entry["ram_free_before_bytes"] = psutil.virtual_memory().available
            try:
                with log_path.open("wb") as log:
                    server = subprocess.Popen(
                        [sys.executable, "main.py", "--listen", "127.0.0.1",
                         "--port", str(port), "--disable-auto-launch", *flags],
                        cwd=comfy, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 75
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        entry["error"] = "server exited before readiness"
                        break
                    if shutil.disk_usage(working).free < MIN_FREE_DISK_BYTES:
                        entry["error"] = "disk safety threshold during startup"
                        break
                    if psutil.virtual_memory().available < 3_000_000_000:
                        entry["error"] = "RAM safety threshold during startup"
                        break
                    try:
                        with urllib.request.urlopen(
                                "http://127.0.0.1:" + str(port) + "/object_info",
                                timeout=3) as response:
                            node_info = json.load(response)
                        entry["http_ok"] = response.status == 200
                        entry["gguf_registered"] = any("GGUF" in key for key in node_info)
                        entry["registered_gguf_nodes"] = sorted(
                            key for key in node_info if "GGUF" in key)[:20]
                        break
                    except (OSError, ValueError):
                        time.sleep(1)
                else:
                    entry["error"] = "75-second startup timeout"
                if server.poll() is None:
                    entry["server_rss_bytes"] = psutil.Process(server.pid).memory_info().rss
                entry["gpu_free_after_bytes"] = [torch.cuda.mem_get_info(i)[0]
                                                 for i in range(torch.cuda.device_count())]
                entry["ram_free_after_bytes"] = psutil.virtual_memory().available
            except Exception as exc:
                entry["error"] = (type(exc).__name__ + ": " + str(exc))[:400]
            finally:
                if server is not None:
                    if server.poll() is None:
                        server.terminate()
                    try:
                        server.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
                if log_path.is_file():
                    entry["log_tail"] = log_path.read_bytes()[-6000:].decode("utf-8", "replace")
                    log_path.unlink()
                time.sleep(1)  # Allow the previous process's GPU context to release.
    else:
        for name, flags in configs:
            checks["configuration_probes"][name] = {
                "attempted": False, "http_ok": False, "gguf_registered": False,
                "flags": flags, "model_loaded": False,
                "error": "ComfyUI/ComfyUI-GGUF source or import unavailable"}
'''


def make_kernel():
    source = CL8_SOURCE
    assert source.count(_MARKER) == 1
    source = source.replace(_MARKER, _CONFIGS + _MARKER)
    assert source.count('report["comfyui_started"] = False') == 1
    source = source.replace('report["comfyui_started"] = False',
                            'report["comfyui_started"] = any(\n'
                            '    p["http_ok"] for p in checks["configuration_probes"].values())')
    for old, new in (
        ("cl8-sources", "cl12-sources"),
        ('"cl8_checks"', '"cl12_checks"'),
        ('"cl8_disk_free_after_bytes"', '"cl12_disk_free_after_bytes"'),
        ('"cl8_torch_after"', '"cl12_torch_after"'),
        ('"cl8_cuda_after"', '"cl12_cuda_after"'),
        ("cl8-dependency-probe.json", REPORT_NAME),
        ("cl8_dependency_probe_complete", "cl12_config_probe_complete"),
    ):
        assert old in source, f"CL-8 marker changed: {old}"
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl12_checks")
    expected = {"default", "lowvram_no_dynamic", "cpu_vae", "aggressive_ram_offload"}
    if (report.get("git_sha") != sha or report.get("run_id") != run_id
            or report.get("install_completed") is not True
            or report.get("no_deps") is not True or report.get("no_cache") is not True
            or not isinstance(checks, dict)
            or not isinstance(checks.get("configuration_probes"), dict)
            or set(checks["configuration_probes"]) != expected
            or report.get("comfyui_started") is not any(
                p.get("http_ok") is True for p in checks["configuration_probes"].values())
            or any(x.get("model_loaded") is not False for x in
                   checks["configuration_probes"].values())
            or report.get("cl12_torch_after") != report.get("torch_before")
            or report.get("cl12_cuda_after") != report.get("cuda_before")
            or report.get("cl12_disk_free_after_bytes", 0) < 19_300_000_000
            or any(report.get(key) is not False for key in
                   ("model_downloaded", "ltx_loaded", "video_generated"))):
        raise ValueError("Incomplete, stale or unsafe CL-12 report")


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
