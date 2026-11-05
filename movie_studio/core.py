"""Deterministic scene planning and conservative compute estimates."""

from dataclasses import asdict, dataclass
import json
from math import ceil
from pathlib import Path


@dataclass(frozen=True)
class Scene:
    id: str
    prompt: str
    duration_seconds: float


def plan(script: str, max_clip_seconds: float = 2.0) -> list[Scene]:
    """Split nonempty lines into shots; each line is one planned clip.

    The script intentionally does not invoke an LLM or a GPU provider.
    """
    if not 0 < max_clip_seconds <= 20:
        raise ValueError("max_clip_seconds must be between 0 and 20")
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    if not lines:
        raise ValueError("script must contain at least one scene line")
    return [Scene(f"scene-{i:04d}", line, max_clip_seconds) for i, line in enumerate(lines, 1)]


def budget(scenes: list[Scene], gpu_seconds_per_video_second: float,
           quota_seconds_per_day: float = 300, quota_multiplier: float = 2) -> dict:
    if not scenes or gpu_seconds_per_video_second <= 0 or quota_seconds_per_day <= 0 or quota_multiplier <= 0:
        raise ValueError("positive estimate inputs and at least one scene required")
    video_seconds = sum(scene.duration_seconds for scene in scenes)
    gpu_seconds = video_seconds * gpu_seconds_per_video_second
    effective_daily_gpu_seconds = quota_seconds_per_day / quota_multiplier
    return {
        "scenes": len(scenes), "video_seconds": video_seconds,
        "estimated_gpu_seconds": gpu_seconds,
        "effective_daily_gpu_seconds": effective_daily_gpu_seconds,
        "optimistic_minimum_days": ceil(gpu_seconds / effective_daily_gpu_seconds),
        "assumptions": "Illustrative only; excludes queues, failures, retries and variable resolution costs. No provider calls made.",
    }


def save_plan(scenes: list[Scene], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"version": 1, "scenes": [asdict(s) for s in scenes]}, indent=2) + "\n", encoding="utf-8")
