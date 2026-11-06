"""CL-47: render one short text-to-video scene with Wan 2.1 T2V 1.3B on Kaggle.

GitHub Actions submits a private Kaggle T4 script, polls it, downloads the
MP4 and report, and verifies the MP4 with FFprobe. Nothing is synthetic: if
the model cannot be installed, loaded or run, the job fails and the Kaggle
error report is kept as diagnostics.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SLUG = "debatt-ai-movie-wan21-scene"
TITLE = "Debatt AI Movie Wan21 Scene"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REPORT_NAME = "wan21-scene.json"
VIDEO_NAME = "wan21-scene.mp4"
RESULTS = Path("kaggle-cl47-results")
SUBMISSION = Path("kaggle-cl47-submission")
KAGGLE_TIMEOUT_SECONDS = 5400
MAX_POLLS = 120
POLL_SECONDS = 60
FPS = 16
MAX_VIDEO_BYTES = 250_000_000

# Pinned Python packages installed on top of Kaggle's own CUDA PyTorch.
# Kaggle's torch is held fixed with a constraints file so pip cannot replace it.
PACKAGES = (
    "diffusers==0.40.0",
    "transformers==5.19.0",
    "accelerate==1.15.0",
    "ftfy==6.3.1",
    "sentencepiece==0.2.1",
    "imageio==2.37.0",
    "imageio-ffmpeg==0.6.0",
)
RESOLUTIONS = {"832x480": (480, 832), "480x832": (832, 480), "576x320": (320, 576)}
FRAME_CHOICES = (17, 33, 49, 81)
DTYPES = ("float16", "float32")
DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, deformed, jpeg artifacts, watermark, text, "
    "subtitles, static image, worst quality"
)

KERNEL_SOURCE = r'''import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

CONFIG = json.loads(__CONFIG_JSON__)
WORK = "/kaggle/working"
REPORT = os.path.join(WORK, CONFIG["report_name"])
VIDEO = os.path.join(WORK, CONFIG["video_name"])
CACHE = "/tmp/wan21-model"
MIN_FREE_DISK = 30_000_000_000


def progress(phase, **extra):
    fields = " ".join(f"{key}={value}" for key, value in extra.items())
    print(f"PROGRESS run={CONFIG['run_id']} phase={phase} {fields}".rstrip(), flush=True)


def install(packages):
    from importlib import metadata
    constraints = "/tmp/cl47-constraints.txt"
    with open(constraints, "w", encoding="utf-8") as handle:
        # Pin Kaggle's installed CUDA builds by distribution version so pip keeps them.
        for name in ("torch", "torchvision", "torchaudio"):
            try:
                handle.write(name + "==" + metadata.version(name) + "\n")
            except metadata.PackageNotFoundError:
                pass
    command = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
               "-c", constraints, *packages]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=900, check=False)
    print(result.stdout[-6000:], flush=True)
    if result.returncode != 0:
        raise RuntimeError("pip install failed with exit code " + str(result.returncode))


def frame_stats(frames, np):
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise RuntimeError("Unexpected frame array shape " + str(array.shape))
    finite = bool(np.isfinite(array).all())
    stats = {
        "shape": list(array.shape),
        "finite": finite,
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "std": float(np.nanstd(array)),
        "mean_abs_frame_diff": float(np.nanmean(np.abs(np.diff(array, axis=0))))
        if array.shape[0] > 1 else 0.0,
    }
    if not finite:
        raise RuntimeError("Generated frames contain NaN/Inf values")
    if stats["std"] < 0.01:
        raise RuntimeError("Generated frames are blank or constant (std " + str(stats["std"]) + ")")
    return stats


def load_pipeline(local, cfg, torch, offload=True):
    """Load Wan 2.1 from a local snapshot. Shared with offline tests."""
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
    from transformers import UMT5EncoderModel
    # T4 has no fast bf16, but UMT5 overflows in fp16; it runs once, so bf16 is fine.
    text_encoder = UMT5EncoderModel.from_pretrained(
        local, subfolder="text_encoder", dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(local, subfolder="vae", dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        local, text_encoder=text_encoder, vae=vae,
        dtype=getattr(torch, cfg["transformer_dtype"]))
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=cfg["flow_shift"])
    if offload:
        pipe.enable_model_cpu_offload()
    return pipe


def render(pipe, cfg, torch, np, export_to_video, video_path):
    """Run the pipeline and write the MP4. Shared with offline tests."""
    total = cfg["steps"]

    def on_step(_pipe, step, _timestep, callback_kwargs):
        progress("denoise", step=step + 1, total=total)
        return callback_kwargs

    generator = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    started = time.time()
    output = pipe(
        prompt=cfg["prompt"],
        negative_prompt=cfg["negative_prompt"],
        height=cfg["height"],
        width=cfg["width"],
        num_frames=cfg["num_frames"],
        num_inference_steps=total,
        guidance_scale=cfg["guidance_scale"],
        generator=generator,
        output_type="np",
        callback_on_step_end=on_step,
    )
    frames = output.frames[0]
    generation_seconds = time.time() - started
    stats = frame_stats(frames, np)
    if stats["shape"][0] != cfg["num_frames"]:
        raise RuntimeError("Expected " + str(cfg["num_frames"]) + " frames, got "
                           + str(stats["shape"][0]))
    export_to_video(list(frames), video_path, fps=cfg["fps"])
    if not os.path.isfile(video_path) or os.path.getsize(video_path) < 1000:
        raise RuntimeError("MP4 export produced no usable file")
    return {"generation_seconds": round(generation_seconds, 1), "frames": stats,
            "video_bytes": os.path.getsize(video_path)}


def main():
    started = time.time()
    report = {
        "run_id": CONFIG["run_id"], "git_sha": CONFIG["git_sha"],
        "model_id": CONFIG["model_id"], "requested_revision": CONFIG["revision"],
        "settings": {key: CONFIG[key] for key in (
            "prompt", "negative_prompt", "seed", "height", "width", "num_frames",
            "fps", "steps", "guidance_scale", "transformer_dtype")},
        "packages": CONFIG["packages"], "video_generated": False, "error": None,
    }
    try:
        import torch
        report["torch_version"] = torch.__version__
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available")
        report["gpu"] = [{"name": torch.cuda.get_device_properties(i).name,
                          "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
                         for i in range(torch.cuda.device_count())]
        report["system_ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        os.makedirs(CACHE, exist_ok=True)
        free = shutil.disk_usage(CACHE).free
        report["model_disk_free_bytes"] = free
        if free < MIN_FREE_DISK:
            raise RuntimeError("Only " + str(free) + " bytes free for model download")

        progress("install")
        install(CONFIG["packages"])
        import numpy as np
        import huggingface_hub
        from diffusers.utils import export_to_video

        progress("download")
        t0 = time.time()
        info = huggingface_hub.model_info(CONFIG["model_id"], revision=CONFIG["revision"])
        report["resolved_revision"] = info.sha
        local = huggingface_hub.snapshot_download(
            CONFIG["model_id"], revision=info.sha, local_dir=CACHE,
            allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                            "text_encoder/*", "transformer/*", "vae/*"])
        report["download_seconds"] = round(time.time() - t0, 1)
        report["model_bytes"] = sum(
            os.path.getsize(os.path.join(root, name))
            for root, _dirs, names in os.walk(local) for name in names)

        progress("load")
        t0 = time.time()
        pipe = load_pipeline(local, CONFIG, torch)
        report["load_seconds"] = round(time.time() - t0, 1)
        torch.cuda.reset_peak_memory_stats()

        progress("generate")
        report.update(render(pipe, CONFIG, torch, np, export_to_video, VIDEO))
        report["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        report["video_generated"] = True
        progress("done", video_bytes=report["video_bytes"])
    except BaseException as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)[:4000]
        report["traceback"] = traceback.format_exc()[-8000:]
        if os.path.exists(VIDEO):
            os.remove(VIDEO)
        raise
    finally:
        report["total_seconds"] = round(time.time() - started, 1)
        with open(REPORT, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        shutil.rmtree(CACHE, ignore_errors=True)
        gc.collect()


if __name__ == "__main__":
    main()
'''


def build_config(*, sha, run_id, prompt, seed, resolution, num_frames, steps,
                 guidance_scale, transformer_dtype, revision="main",
                 negative_prompt=DEFAULT_NEGATIVE):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Full Git commit SHA required")
    if not re.fullmatch(r"[0-9]+-[0-9]+", run_id):
        raise ValueError("Expected GitHub run ID and attempt")
    prompt = " ".join(str(prompt).split())
    if not 3 <= len(prompt) <= 800:
        raise ValueError("Prompt must be 3-800 characters")
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,64}", revision):
        raise ValueError("Invalid model revision")
    if resolution not in RESOLUTIONS:
        raise ValueError("Resolution must be one of " + ", ".join(RESOLUTIONS))
    if int(num_frames) not in FRAME_CHOICES:
        raise ValueError("num_frames must be one of " + str(FRAME_CHOICES))
    if not 10 <= int(steps) <= 40:
        raise ValueError("steps must be between 10 and 40")
    if not 1.0 <= float(guidance_scale) <= 10.0:
        raise ValueError("guidance_scale must be between 1 and 10")
    if transformer_dtype not in DTYPES:
        raise ValueError("transformer_dtype must be one of " + str(DTYPES))
    if not 0 <= int(seed) < 2**32:
        raise ValueError("seed must be a 32-bit unsigned integer")
    height, width = RESOLUTIONS[resolution]
    return {
        "git_sha": sha, "run_id": run_id, "model_id": MODEL_ID, "revision": revision,
        "prompt": prompt, "negative_prompt": negative_prompt, "seed": int(seed),
        "height": height, "width": width, "num_frames": int(num_frames), "fps": FPS,
        "steps": int(steps), "guidance_scale": float(guidance_scale),
        "transformer_dtype": transformer_dtype,
        "flow_shift": 3.0 if max(height, width) <= 832 else 5.0,
        "packages": list(PACKAGES), "report_name": REPORT_NAME, "video_name": VIDEO_NAME,
    }


def make_kernel(config):
    literal = repr(json.dumps(config))
    return KERNEL_SOURCE.replace("__CONFIG_JSON__", literal)


def validate_report(report, config):
    if not isinstance(report, dict):
        raise ValueError("Report is not a JSON object")
    if report.get("git_sha") != config["git_sha"] or report.get("run_id") != config["run_id"]:
        raise ValueError("Stale or foreign Kaggle report")
    if report.get("error"):
        raise ValueError("Kaggle render failed: " + str(report["error"]))
    if report.get("video_generated") is not True:
        raise ValueError("Kaggle report does not confirm a generated video")
    frames = report.get("frames") or {}
    if not frames.get("finite") or (frames.get("shape") or [None])[0] != config["num_frames"]:
        raise ValueError("Kaggle report frame statistics are invalid")
    return report


def probe_video(path, config, *, runner=subprocess.run):
    path = Path(path)
    if not path.is_file():
        raise ValueError("MP4 missing")
    size = path.stat().st_size
    if not 1000 <= size <= MAX_VIDEO_BYTES:
        raise ValueError("MP4 size out of bounds: " + str(size))
    result = runner(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,width,height,nb_read_frames",
         "-of", "json", str(path)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120, check=False)
    if result.returncode != 0:
        raise ValueError("FFprobe could not read the MP4: " + result.stdout[-2000:])
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise ValueError("MP4 has no video stream")
    stream = streams[0]
    frames = int(stream.get("nb_read_frames") or 0)
    if frames != config["num_frames"]:
        raise ValueError(f"MP4 has {frames} frames, expected {config['num_frames']}")
    if (int(stream.get("width", 0)), int(stream.get("height", 0))) != (config["width"], config["height"]):
        raise ValueError("MP4 resolution does not match the request")
    return {"codec": stream.get("codec_name"), "width": int(stream["width"]),
            "height": int(stream["height"]), "frames": frames, "bytes": size,
            "seconds": round(frames / config["fps"], 2)}


def _execute(args, *, timeout=120):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, check=False)


def _download_output(kernel_name):
    output = _execute(["kaggle", "kernels", "output", kernel_name,
                       "-p", str(RESULTS), "--force"], timeout=600)
    (RESULTS / "download.log").write_text(output.stdout, encoding="utf-8")
    return output.returncode == 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sha", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--username", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", default="832x480")
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--transformer-dtype", default="float16")
    p.add_argument("--revision", default="main")
    args = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        p.error("Invalid Kaggle username")
    try:
        config = build_config(
            sha=args.sha, run_id=args.run_id, prompt=args.prompt, seed=args.seed,
            resolution=args.resolution, num_frames=args.num_frames, steps=args.steps,
            guidance_scale=args.guidance_scale, transformer_dtype=args.transformer_dtype,
            revision=args.revision)
    except ValueError as exc:
        p.error(str(exc))

    shutil.rmtree(RESULTS, ignore_errors=True)
    RESULTS.mkdir()
    SUBMISSION.mkdir(exist_ok=True)
    kernel_name = f"{args.username}/{SLUG}"
    (SUBMISSION / "run.py").write_text(make_kernel(config), encoding="utf-8")
    metadata = {
        "id": kernel_name, "title": TITLE, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "model_sources": [],
    }
    (SUBMISSION / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    submitted = _execute(["kaggle", "kernels", "push", "-p", str(SUBMISSION),
                          "--accelerator", "NvidiaTeslaT4",
                          "--timeout", str(KAGGLE_TIMEOUT_SECONDS)], timeout=180)
    (RESULTS / "submission.log").write_text(submitted.stdout, encoding="utf-8")
    print(submitted.stdout, flush=True)
    if submitted.returncode != 0 or re.search(
            r"(?im)^\s*(kernel push error|error pushing|failed:|traceback)", submitted.stdout):
        raise RuntimeError("Kaggle rejected the submission")
    if not re.search(r"kernel version \d+ (successfully )?pushed", submitted.stdout, re.I):
        raise RuntimeError("Kaggle CLI did not confirm a new version")

    status = None
    for _ in range(MAX_POLLS):
        time.sleep(POLL_SECONDS)
        state = _execute(["kaggle", "kernels", "status", kernel_name], timeout=60)
        with (RESULTS / "status.log").open("a", encoding="utf-8") as log:
            log.write(state.stdout + "\n")
        print(state.stdout.strip(), flush=True)
        if state.returncode != 0:
            continue  # transient API failure; the poll budget stays bounded
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", state.stdout)
        status = match.group(1) if match else None
        if status in {"COMPLETE", "ERROR", "FAILED", "CANCELED", "CANCELLED",
                      "CANCELLING", "CANCELING"}:
            break
        if status not in {"QUEUED", "RUNNING", "STARTING", "INITIALIZING", "NEW_SCRIPT", None}:
            raise RuntimeError(f"Unknown Kaggle status: {status}")
    else:
        raise RuntimeError("Kaggle render exceeded the bounded polling window")

    downloaded = _download_output(kernel_name)
    report_path = RESULTS / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    if status != "COMPLETE":
        detail = (report or {}).get("error") or "no report returned"
        raise RuntimeError(f"Kaggle kernel ended with {status}: {detail}")
    if not downloaded or report is None:
        raise RuntimeError("Could not download the Kaggle report")
    validate_report(report, config)
    video = probe_video(RESULTS / VIDEO_NAME, config)
    report["verified_mp4"] = video
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Verified Wan 2.1 scene:", json.dumps(video), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
