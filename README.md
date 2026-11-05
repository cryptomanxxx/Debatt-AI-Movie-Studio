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
