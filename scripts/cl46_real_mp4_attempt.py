"""CL-46: explicitly queue a real ComfyUI LTX-2.5 graph and retrieve MP4.

No mock media, hidden downloads or synthetic success. Read-only probe by
default; real GPU execution requires an explicit confirmation string.
"""
import argparse
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from scripts.cl43_video_runtime_preflight import preflight

CONFIRM = "QUEUE_REAL_LTX25_VIDEO"
MAX_VIDEO_BYTES = 250_000_000


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


LOCAL_HTTP = urllib.request.build_opener(NoRedirect)


def _request(base, endpoint, *, payload=None, timeout=15):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base + endpoint, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET")
    try:
        with LOCAL_HTTP.open(request, timeout=timeout) as response:
            return json.loads(response.read(5_000_000).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(50_000).decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI HTTP {exc.code}: {detail}") from exc


def probe(comfy_root, base, *, client=_request):
    if base not in ("http://127.0.0.1:8188", "http://localhost:8188"):
        raise ValueError("Only the local ComfyUI endpoint is allowed")
    checks = preflight(comfy_root)
    try:
        stats = client(base, "/system_stats")
        online = isinstance(stats, dict) and isinstance(stats.get("devices"), list)
        devices = [{"name": str(d.get("name", "unknown")),
                    "vram_total": d.get("vram_total")}
                   for d in stats.get("devices", []) if isinstance(d, dict)] if online else []
        error = None
    except (OSError, ValueError, TimeoutError) as exc:
        online, devices, error = False, [], type(exc).__name__
    return {"comfyui_online": online, "runtime_preflight": checks,
            "devices": devices, "connection_error": error,
            "gpu_used": False, "video_generated": False}


def find_video(history_entry):
    if not isinstance(history_entry, dict):
        return None
    for output in history_entry.get("outputs", {}).values():
        if not isinstance(output, dict):
            continue
        for group in ("videos", "gifs", "images"):
            for item in output.get(group, []):
                if isinstance(item, dict) and isinstance(item.get("filename"), str) \
                        and item["filename"].lower().endswith(".mp4") \
                        and item.get("type", "output") == "output":
                    return {"filename": item["filename"],
                            "subfolder": item.get("subfolder", ""), "type": "output"}
    return None


def attempt(comfy_root, workflow_path, output_path, *, base, confirm,
            timeout_seconds=1800, client=_request, downloader=None,
            clock=time.monotonic, pause=time.sleep, progress=None):
    if confirm != CONFIRM:
        raise ValueError("Explicit real-GPU queue confirmation required")
    readiness = probe(comfy_root, base, client=client)
    if not readiness["comfyui_online"]:
        raise RuntimeError("ComfyUI server is unavailable")
    if not readiness["runtime_preflight"]["ready_for_manual_workflow_import"]:
        raise RuntimeError("LTX-2.5 model/runtime preflight failed; see probe report")
    if not readiness["devices"]:
        raise RuntimeError("ComfyUI reports no compute devices")
    graph = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not graph:
        raise ValueError("Prepared ComfyUI API workflow required")
    if not all(isinstance(v, dict) and isinstance(v.get("inputs"), dict)
               and isinstance(v.get("class_type"), str) for v in graph.values()):
        raise ValueError("Invalid ComfyUI API graph")

    queued = client(base, "/prompt", payload={"prompt": graph})
    prompt_id = queued.get("prompt_id") if isinstance(queued, dict) else None
    if not isinstance(prompt_id, str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", prompt_id):
        raise RuntimeError("ComfyUI rejected graph: " + str(queued)[:1500])
    if progress:
        progress({"status": "queued", "prompt_id": prompt_id,
                  "gpu_queue_submitted": True, "video_generated": False})
    deadline = clock() + timeout_seconds
    while clock() < deadline:
        history = client(base, "/history/" + prompt_id)
        entry = history.get(prompt_id) if isinstance(history, dict) else None
        if entry:
            status = entry.get("status", {})
            if not isinstance(status, dict):
                raise RuntimeError("ComfyUI returned invalid execution status: " + prompt_id)
            if status.get("status_str") == "error" or status.get("completed") is False:
                detail = json.dumps(status.get("messages", []), ensure_ascii=False)[:2000]
                raise RuntimeError("ComfyUI execution failed: " + detail)
            if status.get("status_str") != "success" or status.get("completed") is not True:
                raise RuntimeError("ComfyUI history lacks confirmed success: " + prompt_id)
            video = find_video(entry)
            if not video:
                raise RuntimeError("ComfyUI completed but produced no output MP4: " + prompt_id)
            if downloader is None:
                downloader = download_video
            size = downloader(base, video, Path(output_path))
            return {"status": "real_mp4_downloaded", "prompt_id": prompt_id,
                    "path": str(output_path), "bytes": size,
                    "gpu_queue_submitted": True, "video_generated": True}
        pause(5)
    raise TimeoutError("ComfyUI prompt still pending at timeout: " + prompt_id)


def download_video(base, descriptor, output_path):
    filename = descriptor["filename"]
    if Path(filename).name != filename or not filename.lower().endswith(".mp4"):
        raise ValueError("Unexpected remote video filename")
    if Path(descriptor.get("subfolder", "")).is_absolute() or ".." in Path(descriptor.get("subfolder", "")).parts:
        raise ValueError("Unexpected remote video subfolder")
    query = urllib.parse.urlencode(descriptor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    size = 0
    try:
        with LOCAL_HTTP.open(base + "/view?" + query, timeout=60) as response, temporary.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise ValueError("Video exceeded 250 MB retrieval limit")
                file.write(chunk)
        if size < 32:
            raise ValueError("Output is too small to be a real MP4")
        with temporary.open("rb") as file:
            header = file.read(32)
        if b"ftyp" not in header[4:16]:
            raise ValueError("Output is not an MP4 container")
        if not shutil.which("ffprobe"):
            raise RuntimeError("FFprobe required to verify video frames")
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
             "stream=codec_type,width,height,nb_read_frames", "-of", "json", str(temporary)],
            capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise ValueError("FFprobe rejected MP4: " + result.stderr[:400])
        streams = json.loads(result.stdout).get("streams", [])
        if not any(s.get("codec_type") == "video" and s.get("width", 0) > 0
                   and s.get("height", 0) > 0
                   and int(s.get("nb_read_frames") or 0) > 0 for s in streams):
            raise ValueError("MP4 lacks decodable video frames")
        temporary.replace(output_path)
        return size
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8188")
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--output", type=Path, default=Path("cl46-results/first-real-clip.mp4"))
    parser.add_argument("--report", type=Path, default=Path("cl46-results/attempt.json"))
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def record(report):
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    record({"status": "starting", "video_generated": False})
    try:
        if args.submit:
            if not args.workflow:
                parser.error("--submit requires --workflow")
            report = attempt(args.comfy_root, args.workflow, args.output,
                             base=args.base, confirm=args.confirm, progress=record)
        else:
            report = probe(args.comfy_root, args.base)
    except Exception as exc:
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        record({**previous, "status": "failed", "video_generated": False,
                "error": str(exc)[:2000]})
        raise
    record(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
