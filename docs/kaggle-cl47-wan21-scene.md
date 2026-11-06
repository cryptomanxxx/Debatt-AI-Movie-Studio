# CL-47 — first real AI scene with Wan 2.1 T2V 1.3B on Kaggle

LTX-2.5 (22B) is too large to be a reliable first target for a Kaggle T4, which has 16 GB of VRAM per card. CL-47 renders a short text-to-video scene with the much smaller open-weight **Wan 2.1 T2V 1.3B** model (Apache 2.0, `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`). It uses the same GitHub → Kaggle CLI route as the earlier experiments.

## Run it

1. The repository secrets `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` must already exist (see `kaggle-gpu.md`).
2. Open **Actions → Kaggle CL-47 Wan 2.1 Scene (manual) → Run workflow**.
3. Type `YES` and `YES_DOWNLOAD_WAN21_1_3B`, then write a scene description. English prompts work best.
4. When the run finishes, download the `movie-studio-cl47-wan21-scene-<run id>` artifact (kept for 7 days). It contains:
   - `wan21-scene.mp4`: the generated clip with no sound.
   - `wan21-scene-with-silent-audio.mp4`: the same clip with a silent audio track, which `movie_studio.cli assemble` requires.
   - `wan21-scene.json`: settings, model revision, timings, peak VRAM and frame statistics.

| Input | Default | Notes |
| --- | --- | --- |
| `num_frames` | 33 | Rendered at 16 fps: 17 = 1 s, 33 = 2 s, 49 = 3 s, 81 = 5 s |
| `resolution` | 832x480 | 480x832 for portrait; 576x320 is a faster, lower-quality draft |
| `steps` | 25 | 10–40 |
| `seed` | 42 | Use the same seed to reproduce a result |
| `transformer_dtype` | float16 | Try float32 if the frames come out black or blank |

## What happens on Kaggle

The private Kaggle script (T4, internet on, 90-minute limit) does the following:

1. Installs pinned `diffusers==0.40.0`, `transformers==5.19.0` and `accelerate==1.15.0`, plus small helpers. Kaggle's own CUDA PyTorch is held fixed with a pip constraints file.
2. Downloads the model snapshot (~30 GB of free disk is required in `/tmp`) and records the resolved Hugging Face commit.
3. Loads the UMT5 text encoder in bf16, because UMT5 overflows in fp16. The transformer uses the selected dtype and the VAE uses fp32. The scheduler is UniPC with `flow_shift=3.0` for 480p. Model CPU offload keeps only one component on the GPU at a time.
4. Generates the frames and rejects NaN or blank output, then writes the MP4 and JSON report. The model cache is deleted afterwards.

The GitHub runner then checks the commit SHA and run ID in the report. It also uses FFprobe to confirm that the MP4 has exactly the requested frame count and resolution. If any step fails, the job fails and keeps the Kaggle error report. No placeholder video is produced.

## Verified so far, and what is not

- Offline unit tests cover input validation, safe prompt embedding, report and video validation, and blank/NaN frame rejection.
- Before this PR, the kernel's `load_pipeline` and `render` functions ran on a CPU against the pinned package set. They used a tiny, randomly initialised Wan pipeline (real diffusers code, fake weights). That test produced a 17-frame H.264 MP4 that passed `probe_video`.
- **Not yet verified:** a real Kaggle run. Download time, runtime per step on a T4, whether fp16 gives clean frames and whether RAM is sufficient are all unknown until the first run. Expect roughly 10–30 minutes for a 2-second clip at 832x480. This is an estimate, not a measurement.

## Limits

- Wan 2.1 1.3B produces 480p clips without sound. Quality is clearly below large models, and text and faces are often poor.
- Every run consumes free Kaggle GPU quota that is shared with Model Lab. The workflow has no schedule and only runs manually.
- Artifacts are temporary. Download clips you want to keep and do not commit MP4 files to this repository.
