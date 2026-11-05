"""Provider-neutral, offline production queue.

This module records planned jobs but deliberately has no network or GPU client.
"""

import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1
PROVIDERS = {"unassigned", "ltx-local", "ltx-huggingface-zerogpu"}


def _key(scene_id: str, prompt: str, duration: float) -> str:
    payload = json.dumps([scene_id, prompt, duration], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_scene(scene_id: object, prompt: object, duration: object) -> None:
    if (not isinstance(scene_id, str) or not scene_id or
            not isinstance(prompt, str) or not prompt.strip() or
            not isinstance(duration, (int, float)) or isinstance(duration, bool) or
            not 0 < duration <= 20):
        raise ValueError("invalid scene id, prompt or duration")


def build_queue(plan: dict, *, provider: str = "unassigned") -> dict:
    if plan.get("version") != 1 or not isinstance(plan.get("scenes"), list) or not plan["scenes"]:
        raise ValueError("expected nonempty scene plan version 1")
    if provider not in PROVIDERS:
        raise ValueError("unsupported provider label")
    ids = set()
    jobs = []
    for scene in plan["scenes"]:
        scene_id = scene.get("id")
        prompt = scene.get("prompt")
        duration = scene.get("duration_seconds")
        _validate_scene(scene_id, prompt, duration)
        if scene_id in ids:
            raise ValueError("duplicate scene")
        ids.add(scene_id)
        # Stable idempotency token lets a future provider avoid duplicate spend.
        token = _key(scene_id, prompt, duration)
        jobs.append({"scene_id": scene_id, "prompt": prompt, "duration_seconds": duration,
                     "provider": provider, "status": "pending", "idempotency_key": token,
                     "output": None})
    return {"version": SCHEMA_VERSION, "generation_enabled": False, "jobs": jobs}


def validate_queue(queue: dict) -> int:
    if queue.get("version") != SCHEMA_VERSION or queue.get("generation_enabled") is not False:
        raise ValueError("queue version mismatch or generation enabled")
    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("queue has no jobs")
    keys = set()
    ids = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("queue job must be an object")
        scene_id, prompt, duration = (job.get("scene_id"), job.get("prompt"),
                                      job.get("duration_seconds"))
        _validate_scene(scene_id, prompt, duration)
        if scene_id in ids:
            raise ValueError("duplicate scene id")
        ids.add(scene_id)
        if job.get("provider") not in PROVIDERS:
            raise ValueError("unsupported provider")
        if job.get("status") != "pending" or job.get("output") is not None:
            raise ValueError("only unprocessed queue jobs are accepted")
        key = job.get("idempotency_key")
        if key != _key(scene_id, prompt, duration) or key in keys:
            raise ValueError("incorrect or duplicate idempotency key")
        keys.add(key)
    return len(jobs)


def write_queue(queue: dict, destination: Path) -> None:
    validate_queue(queue)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing queue: {destination}")
    destination.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
