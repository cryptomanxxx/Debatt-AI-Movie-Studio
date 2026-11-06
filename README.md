# Debatt-AI-Movie-Studio
An open-source AI filmmaking research platform for automated screenplay generation, multi-scene video production, GPU orchestration, and long-form movie creation using LTX-2.5 and other open-weight generative models.

## Milestone 1: offline-first proof of concept

The initial pipeline **does not generate video, invoke LTX, contact Hugging Face, or consume GPU credits**. It converts newline-separated shot descriptions into an ordered scene plan, estimates compute needs from an explicitly supplied rate, and can assemble pre-existing local MP4 clips.

```bash
python -m unittest discover -s tests -v
python -m movie_studio.cli plan examples/first-film.txt --out /tmp/movie-plan.json
python -m movie_studio.cli budget /tmp/movie-plan.json --gpu-seconds-per-video-second 45
```

The rate of 45 GPU seconds per generated video second is **illustrative**, not a benchmark. Free ZeroGPU quota is commonly expressed as 300 quota seconds per day, and the 96 GB `xlarge` allocation consumes 2× quota. Actual availability, per-call caps, model performance and provider policies must be checked independently. The estimate is an optimistic lower bound: queues, failed outputs, retries and variable resolution costs are excluded.

### Assemble existing clips

Install `ffmpeg` and create a JSON file with an ordered `clips` array of absolute MP4 paths, as shown in `examples/clips.example.json`. Clips **must each contain video and audio streams** and should share compatible dimensions/frame rates and audio parameters. Then run:

```bash
python -m movie_studio.cli assemble /path/to/clips.json --out /path/to/movie.mp4
```

The command refuses to overwrite output files. Store generated videos outside Git history (e.g. object storage); GitHub Actions artifacts are not a permanent film library.

### Safety and roadmap

- No remote generation API or scheduled generation workflow is enabled. GitHub Actions runs unit tests and estimates only.
- A future provider interface must have explicit opt-in, authentication, budget limits, rate limiting, and persistence before any GPU call.
- This repository's MIT license covers original code only. LTX weights and generated media are governed by their own applicable terms.
- Next: characterize real GPU seconds per shot, create a provider-independent job manifest and implement a user-approved GPU connector.

## Milestone 2: offline job queue and temporary artifacts

Create a reproducible queue of pending scenes without calling any provider:

```bash
python -m movie_studio.cli queue /tmp/movie-plan.json --out /tmp/movie-queue.json
python -m movie_studio.cli validate-queue /tmp/movie-queue.json
```

The queue contains deterministic idempotency keys and `generation_enabled: false`. Labels such as `ltx-huggingface-zerogpu` are **metadata only**, not working integrations. There is no authenticated provider client, payment capability or scheduling in this milestone.

The manually triggered **Synthetic video artifact demo** workflow produces a two-second blue test video with a tone using FFmpeg on a standard GitHub Actions runner. It does **not** generate AI video. It uploads the MP4 as a downloadable Actions artifact retained for **seven days**. Trigger it from the Actions tab using *Run workflow*. Artifacts are temporary and subject to GitHub's current retention/storage limits; download anything you wish to keep. Do not commit real MP4s or media credentials to this public repository.

## CL-47: first real AI scene (Wan 2.1 on Kaggle)

**Actions → Kaggle CL-47 Wan 2.1 Scene (manual)** renders a short text-to-video clip (1–5 s, 480p, no sound) with the open-weight Wan 2.1 T2V 1.3B model on a private Kaggle T4. The MP4 is verified with FFprobe and uploaded as a 7-day Actions artifact, together with a copy that has a silent audio track for `assemble`. See [docs/kaggle-cl47-wan21-scene.md](docs/kaggle-cl47-wan21-scene.md). The first real run (2026-11-06) produced a verified 2-second 832x480 clip in about 11 minutes of Kaggle time (6.2 minutes of generation, 11.7 GB peak VRAM).

## CL-48: larger model (Wan 2.2 TI2V 5B on Kaggle)

**Actions → Kaggle CL-48 Wan 2.2 TI2V 5B Scene (manual)** renders a 1–5 s clip at 24 fps with the open-weight Wan 2.2 TI2V 5B model, which is about four times larger than CL-47's model. It uses the same verified Kaggle flow, and a token budget guard rejects settings too slow for a T4. See [docs/kaggle-cl48-wan22-scene.md](docs/kaggle-cl48-wan22-scene.md). The first real run (2026-11-06) produced a verified 2-second 832x480 clip in about 13 minutes of Kaggle time (~8.3 s per step, 13.1 GB peak VRAM). Its quality was clearly better than CL-47.
