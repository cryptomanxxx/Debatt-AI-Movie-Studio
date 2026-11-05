"""Command-line entry point; only local, non-generative operations."""

import argparse
import json
from pathlib import Path
import subprocess
import shutil

from .core import Scene, budget, plan, save_plan


def assemble(manifest: Path, destination: Path) -> None:
    """Assemble an explicitly ordered list of trusted local MP4 paths."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    paths = [Path(p).resolve() for p in data["clips"]]
    if not paths or any(not p.is_file() or p.suffix.lower() != ".mp4" for p in paths):
        raise ValueError("manifest must list existing MP4 files")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for assembly")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Filter-based concatenation re-encodes and handles distinct clip timestamps.
    inputs = [item for p in paths for item in ("-i", str(p))]
    streams = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(len(paths)))
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", *inputs,
               "-filter_complex", f"{streams}concat=n={len(paths)}:v=1:a=1[v][a]",
               "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac",
               str(destination)]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debatt-AI Movie Studio (offline tools)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("script", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--clip-seconds", type=float, default=2)
    b = sub.add_parser("budget")
    b.add_argument("plan", type=Path)
    b.add_argument("--gpu-seconds-per-video-second", type=float, required=True)
    b.add_argument("--daily-quota-seconds", type=float, default=300)
    b.add_argument("--quota-multiplier", type=float, default=2)
    a = sub.add_parser("assemble")
    a.add_argument("manifest", type=Path)
    a.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        scenes = plan(args.script.read_text(encoding="utf-8"), args.clip_seconds)
        save_plan(scenes, args.out)
        print(f"Wrote {len(scenes)} scenes to {args.out}")
    elif args.command == "budget":
        data = json.loads(args.plan.read_text(encoding="utf-8"))
        scenes = [Scene(**s) for s in data["scenes"]]
        print(json.dumps(budget(scenes, args.gpu_seconds_per_video_second,
                                args.daily_quota_seconds, args.quota_multiplier), indent=2))
    else:
        assemble(args.manifest, args.out)
        print(f"Assembled {args.out}")


if __name__ == "__main__":
    main()
