"""CL-47: render one short text-to-video scene with a Wan model on Kaggle.

The module is shared by several model profiles: CL-47 (Wan 2.1 T2V 1.3B) and
CL-48 (Wan 2.2 TI2V 5B, see kaggle_cl48_wan22_scene.py). GitHub Actions submits a private Kaggle T4 script, polls it, downloads the
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
from dataclasses import dataclass
from pathlib import Path

POLL_SECONDS = 60
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
VAE_TILING_ABOVE_PIXELS = 832 * 480
# CL-48b used diffusers' default 256 px tiles (stride 192): 28 tiles per 720p frame and a
# 9.2 min decode. 512 px tiles with the same 64 px overlap need 6 tiles per frame.
VAE_TILE_KWARGS = {"tile_sample_min_height": 512, "tile_sample_min_width": 512,
                   "tile_sample_stride_height": 448, "tile_sample_stride_width": 448}
# CL-49: Real-ESRGAN x2plus, pinned by SHA-256, loaded with spandrel's restricted unpickler.
UPSCALER = {
    "name": "RealESRGAN_x2plus",
    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "sha256": "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
    "package": "spandrel==0.4.2",
}
UPSCALE_TARGETS = {"1080p": (1920, 1080)}
DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, deformed, jpeg artifacts, watermark, text, "
    "subtitles, static image, worst quality"
)


@dataclass(frozen=True)
class Profile:
    """Everything that differs between the Wan models rendered on Kaggle."""
    name: str
    slug: str
    title: str
    model_id: str
    stem: str
    results: Path
    submission: Path
    fps: int
    resolutions: dict
    frame_choices: tuple
    dtypes: tuple
    max_steps: int
    flow_shift: object  # float, or None to keep the model repository's scheduler shift
    max_latent_tokens: int
    kaggle_timeout_seconds: int
    max_polls: int

    @property
    def report_name(self):
        return self.stem + ".json"

    @property
    def video_name(self):
        return self.stem + ".mp4"


WAN21_1_3B = Profile(
    name="Wan 2.1 T2V 1.3B", slug="debatt-ai-movie-wan21-scene",
    title="Debatt AI Movie Wan21 Scene", model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    stem="wan21-scene", results=Path("kaggle-cl47-results"),
    submission=Path("kaggle-cl47-submission"), fps=16,
    resolutions={"832x480": (480, 832), "480x832": (832, 480), "576x320": (320, 576)},
    frame_choices=(17, 33, 49, 81), dtypes=("float16", "float32"), max_steps=40,
    flow_shift=3.0, max_latent_tokens=40_000, kaggle_timeout_seconds=5400, max_polls=120)

# Backwards-compatible CL-47 names.
PROFILE = WAN21_1_3B
SLUG, TITLE, MODEL_ID = PROFILE.slug, PROFILE.title, PROFILE.model_id
REPORT_NAME, VIDEO_NAME = PROFILE.report_name, PROFILE.video_name
RESULTS, SUBMISSION, FPS = PROFILE.results, PROFILE.submission, PROFILE.fps
RESOLUTIONS, FRAME_CHOICES, DTYPES = PROFILE.resolutions, PROFILE.frame_choices, PROFILE.dtypes

KERNEL_SOURCE = r'''import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

# Must be set before torch initialises CUDA. Reduces fragmentation: the first
# 720p attempt failed with 2.8 GiB reserved-but-unallocated during VAE decode.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CONFIG = json.loads(__CONFIG_JSON__)
WORK = "/kaggle/working"
REPORT = os.path.join(WORK, CONFIG["report_name"])
VIDEO = os.path.join(WORK, CONFIG["video_name"])
UPSCALED = os.path.join(WORK, CONFIG["upscale"]["video_name"]) if CONFIG.get("upscale") else None
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
    """Load a Wan model from a local snapshot. Shared with offline tests."""
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
    from transformers import UMT5EncoderModel
    # T4 has no fast bf16, but UMT5 overflows in fp16; it runs once, so bf16 is fine.
    text_encoder = UMT5EncoderModel.from_pretrained(
        local, subfolder="text_encoder", dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(local, subfolder="vae", dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        local, text_encoder=text_encoder, vae=vae,
        dtype=getattr(torch, cfg["transformer_dtype"]))
    if cfg["flow_shift"] is not None:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=cfg["flow_shift"])
    if cfg.get("vae_tiling"):
        # Decode in overlapping spatial tiles; full-frame 720p decode ran out of memory.
        pipe.vae.enable_tiling(**cfg.get("vae_tile_kwargs", {}))
    if offload:
        pipe.enable_model_cpu_offload()
    return pipe


def render(pipe, cfg, torch, np, export_to_video, video_path, keep=None):
    """Run the pipeline and write the MP4. Shared with offline tests.

    If ``keep`` is a list, the float frames are appended to it for upscaling.
    """
    total = cfg["steps"]
    marks = {}

    def on_step(_pipe, step, _timestep, callback_kwargs):
        progress("denoise", step=step + 1, total=total)
        if step + 1 == total:
            marks["decode_start"] = time.time()
            progress("vae_decode", tiling=bool(cfg.get("vae_tiling")))
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
    finished = time.time()
    generation_seconds = finished - started
    stats = frame_stats(frames, np)
    if stats["shape"][0] != cfg["num_frames"]:
        raise RuntimeError("Expected " + str(cfg["num_frames"]) + " frames, got "
                           + str(stats["shape"][0]))
    export_to_video(list(frames), video_path, fps=cfg["fps"])
    if not os.path.isfile(video_path) or os.path.getsize(video_path) < 1000:
        raise RuntimeError("MP4 export produced no usable file")
    if keep is not None:
        keep.append(frames)
    result = {"generation_seconds": round(generation_seconds, 1), "frames": stats,
              "video_bytes": os.path.getsize(video_path)}
    if "decode_start" in marks:
        result["denoise_seconds"] = round(marks["decode_start"] - started, 1)
        result["vae_decode_seconds"] = round(finished - marks["decode_start"], 1)
    return result


def fit_and_crop(image, target_w, target_h, torch):
    """Scale a (1, 3, H, W) tensor to cover target_w x target_h, then centre-crop."""
    _, _, h, w = image.shape
    scale = max(target_w / w, target_h / h)
    new_w, new_h = max(target_w, round(w * scale)), max(target_h, round(h * scale))
    resized = torch.nn.functional.interpolate(
        image, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
    top, left = (new_h - target_h) // 2, (new_w - target_w) // 2
    return resized[:, :, top:top + target_h, left:left + target_w].clamp(0, 1)


def upscale_frames(frames, cfg, torch, np, model, device, dtype):
    """Real-ESRGAN each frame, then fit to the target size. Shared with offline tests."""
    up = cfg["upscale"]
    out = []
    with torch.inference_mode():
        for index, frame in enumerate(frames):
            tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None]
            tensor = tensor.to(device=device, dtype=dtype)
            sr = model(tensor).float().clamp(0, 1)
            fitted = fit_and_crop(sr, up["width"], up["height"], torch)
            out.append(fitted[0].permute(1, 2, 0).cpu().numpy())
            if index % 8 == 0:
                progress("upscale", frame=index + 1, total=len(frames))
    return out


def export_exact(frames, video_path, fps, export_to_video):
    """Export without imageio's default macro_block_size=16, which pads 1080 to 1088."""
    export_to_video(frames, video_path, fps=fps, macro_block_size=1)


def run_upscale(frames, cfg, torch, np, export_to_video, video_path):
    import hashlib
    import urllib.request
    up = cfg["upscale"]
    install([up["package"]])
    from spandrel import ModelLoader
    weights = "/tmp/" + up["model"] + ".pth"
    started = time.time()
    urllib.request.urlretrieve(up["url"], weights)
    digest = hashlib.sha256(open(weights, "rb").read()).hexdigest()
    if digest != up["sha256"]:
        raise RuntimeError("Upscaler weights SHA-256 mismatch: " + digest)
    model = ModelLoader().load_from_file(weights).model.eval()
    device = torch.device("cuda")
    model = model.to(device=device, dtype=torch.float16)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    upscaled = upscale_frames(frames, cfg, torch, np, model, device, torch.float16)
    stats = frame_stats(upscaled, np)
    export_exact(upscaled, video_path, cfg["fps"], export_to_video)
    if not os.path.isfile(video_path) or os.path.getsize(video_path) < 1000:
        raise RuntimeError("Upscaled MP4 export produced no usable file")
    return {"model": up["model"], "seconds": round(time.time() - started, 1),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "frames": stats, "video_bytes": os.path.getsize(video_path)}


def main():
    started = time.time()
    report = {
        "run_id": CONFIG["run_id"], "git_sha": CONFIG["git_sha"],
        "model_id": CONFIG["model_id"], "requested_revision": CONFIG["revision"],
        "settings": {key: CONFIG[key] for key in (
            "prompt", "negative_prompt", "seed", "height", "width", "num_frames",
            "fps", "steps", "guidance_scale", "transformer_dtype", "vae_tiling",
            "upscale")},
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
        kept = []
        report.update(render(pipe, CONFIG, torch, np, export_to_video, VIDEO, keep=kept))
        report["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        report["video_generated"] = True
        if UPSCALED:
            # Free the video pipeline before loading the upscaler.
            pipe.maybe_free_model_hooks()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            progress("upscale", target=CONFIG["upscale"]["label"])
            report["upscale_result"] = run_upscale(
                kept[0], CONFIG, torch, np, export_to_video, UPSCALED)
        progress("done", video_bytes=report["video_bytes"])
    except BaseException as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)[:4000]
        report["traceback"] = traceback.format_exc()[-8000:]
        for path in (VIDEO, UPSCALED):
            if path and os.path.exists(path):
                os.remove(path)
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


def latent_tokens(profile, height, width, num_frames):
    """Transformer sequence length: Wan VAE compresses time by 4, patches are 2x2."""
    spatial = 16 if profile.model_id.startswith("Wan-AI/Wan2.2-TI2V") else 8
    return ((num_frames - 1) // 4 + 1) * (height // spatial // 2) * (width // spatial // 2)


def upscale_config(upscale, height, width, profile):
    if upscale in (None, "none"):
        return None
    if upscale not in UPSCALE_TARGETS:
        raise ValueError("upscale must be 'none' or one of " + ", ".join(UPSCALE_TARGETS))
    long_side, short_side = UPSCALE_TARGETS[upscale]
    target_w, target_h = (long_side, short_side) if width >= height else (short_side, long_side)
    return {"label": upscale, "width": target_w, "height": target_h,
            "model": UPSCALER["name"], "url": UPSCALER["url"], "sha256": UPSCALER["sha256"],
            "package": UPSCALER["package"],
            "video_name": profile.stem + "-" + upscale + ".mp4"}


def build_config(*, sha, run_id, prompt, seed, resolution, num_frames, steps,
                 guidance_scale, transformer_dtype, revision="main",
                 negative_prompt=DEFAULT_NEGATIVE, profile=WAN21_1_3B, upscale="none"):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Full Git commit SHA required")
    if not re.fullmatch(r"[0-9]+-[0-9]+", run_id):
        raise ValueError("Expected GitHub run ID and attempt")
    prompt = " ".join(str(prompt).split())
    if not 3 <= len(prompt) <= 800:
        raise ValueError("Prompt must be 3-800 characters")
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,64}", revision):
        raise ValueError("Invalid model revision")
    if resolution not in profile.resolutions:
        raise ValueError("Resolution must be one of " + ", ".join(profile.resolutions))
    if int(num_frames) not in profile.frame_choices:
        raise ValueError("num_frames must be one of " + str(profile.frame_choices))
    if not 10 <= int(steps) <= profile.max_steps:
        raise ValueError(f"steps must be between 10 and {profile.max_steps}")
    if not 1.0 <= float(guidance_scale) <= 10.0:
        raise ValueError("guidance_scale must be between 1 and 10")
    if transformer_dtype not in profile.dtypes:
        raise ValueError("transformer_dtype must be one of " + str(profile.dtypes))
    if not 0 <= int(seed) < 2**32:
        raise ValueError("seed must be a 32-bit unsigned integer")
    height, width = profile.resolutions[resolution]
    tokens = latent_tokens(profile, height, width, int(num_frames))
    if tokens > profile.max_latent_tokens:
        raise ValueError(f"{resolution} with {num_frames} frames is {tokens} latent tokens; "
                         f"the limit for {profile.name} on a T4 is {profile.max_latent_tokens}. "
                         "Use fewer frames or a lower resolution.")
    return {
        "git_sha": sha, "run_id": run_id, "model_id": profile.model_id, "revision": revision,
        "prompt": prompt, "negative_prompt": negative_prompt, "seed": int(seed),
        "height": height, "width": width, "num_frames": int(num_frames), "fps": profile.fps,
        "steps": int(steps), "guidance_scale": float(guidance_scale),
        "transformer_dtype": transformer_dtype, "latent_tokens": tokens,
        "flow_shift": profile.flow_shift,
        # Full-frame decode is proven up to 832x480 (CL-47, CL-48); tile anything larger.
        "vae_tiling": height * width > VAE_TILING_ABOVE_PIXELS,
        "vae_tile_kwargs": dict(VAE_TILE_KWARGS),
        "upscale": upscale_config(upscale, height, width, profile),
        "packages": list(PACKAGES), "report_name": profile.report_name,
        "video_name": profile.video_name,
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


def _download_output(kernel_name, results):
    output = _execute(["kaggle", "kernels", "output", kernel_name,
                       "-p", str(results), "--force"], timeout=600)
    (results / "download.log").write_text(output.stdout, encoding="utf-8")
    return output.returncode == 0


def main(argv=None, profile=WAN21_1_3B, defaults=None):
    defaults = {"resolution": "832x480", "num_frames": 33, "steps": 25, **(defaults or {})}
    results, submission = profile.results, profile.submission
    p = argparse.ArgumentParser(description=f"Render a {profile.name} scene on Kaggle")
    p.add_argument("--sha", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--username", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", default=defaults["resolution"])
    p.add_argument("--num-frames", type=int, default=defaults["num_frames"])
    p.add_argument("--steps", type=int, default=defaults["steps"])
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--transformer-dtype", default="float16")
    p.add_argument("--revision", default="main")
    p.add_argument("--upscale", default="none", help="none or 1080p (Real-ESRGAN x2 + fit)")
    args = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        p.error("Invalid Kaggle username")
    try:
        config = build_config(
            sha=args.sha, run_id=args.run_id, prompt=args.prompt, seed=args.seed,
            resolution=args.resolution, num_frames=args.num_frames, steps=args.steps,
            guidance_scale=args.guidance_scale, transformer_dtype=args.transformer_dtype,
            revision=args.revision, profile=profile, upscale=args.upscale)
    except ValueError as exc:
        p.error(str(exc))

    shutil.rmtree(results, ignore_errors=True)
    results.mkdir()
    submission.mkdir(exist_ok=True)
    kernel_name = f"{args.username}/{profile.slug}"
    (submission / "run.py").write_text(make_kernel(config), encoding="utf-8")
    metadata = {
        "id": kernel_name, "title": profile.title, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "model_sources": [],
    }
    (submission / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    submitted = _execute(["kaggle", "kernels", "push", "-p", str(submission),
                          "--accelerator", "NvidiaTeslaT4",
                          "--timeout", str(profile.kaggle_timeout_seconds)], timeout=180)
    (results / "submission.log").write_text(submitted.stdout, encoding="utf-8")
    print(submitted.stdout, flush=True)
    if submitted.returncode != 0 or re.search(
            r"(?im)^\s*(kernel push error|error pushing|failed:|traceback)", submitted.stdout):
        raise RuntimeError("Kaggle rejected the submission")
    if not re.search(r"kernel version \d+ (successfully )?pushed", submitted.stdout, re.I):
        raise RuntimeError("Kaggle CLI did not confirm a new version")

    status = None
    for _ in range(profile.max_polls):
        time.sleep(POLL_SECONDS)
        state = _execute(["kaggle", "kernels", "status", kernel_name], timeout=60)
        with (results / "status.log").open("a", encoding="utf-8") as log:
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

    downloaded = _download_output(kernel_name, results)
    report_path = results / profile.report_name
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    if status != "COMPLETE":
        detail = (report or {}).get("error") or "no report returned"
        raise RuntimeError(f"Kaggle kernel ended with {status}: {detail}")
    if not downloaded or report is None:
        raise RuntimeError("Could not download the Kaggle report")
    validate_report(report, config)
    video = probe_video(results / profile.video_name, config)
    report["verified_mp4"] = video
    if config["upscale"]:
        up = config["upscale"]
        upscaled = probe_video(results / up["video_name"],
                               {**config, "width": up["width"], "height": up["height"]})
        report["verified_upscaled_mp4"] = upscaled
        print(f"Verified {up['label']} upscale:", json.dumps(upscaled), flush=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified {profile.name} scene:", json.dumps(video), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
