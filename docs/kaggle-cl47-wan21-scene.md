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

## First real run (2026-11-06)

[Run 42233991439](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42233991439) rendered and verified the project's first real AI scene. It used the default settings: 33 frames at 832x480, 25 steps, seed 42, float16 transformer. The prompt was *"Two people debate passionately on a dimly lit TV studio stage, cinematic lighting, slow camera push-in"*.

**Kaggle machine (measured)**

| Resource | Value |
| --- | --- |
| GPU | 2 × Tesla T4, 15.6 GB VRAM each (only GPU 0 used) |
| System RAM | 33.7 GB |
| Free disk for the model cache | ~1 TB |
| PyTorch | 2.11.0+cu130 |

**Timings and memory (measured)**

| Phase | Result |
| --- | --- |
| Model download (28.9 GB, revision `0fad780a534b6463e45facd96134c9f345acfa5b`) | 122 s |
| Pipeline load | 127 s |
| Generation, 25 steps | 372 s (~15 s per step) |
| Total Kaggle script | 673 s (~11 min) |
| GitHub workflow end to end | ~13 min |
| Peak VRAM (model CPU offload) | 11.7 GB of 15.6 GB |

**Output:** an H.264 MP4 at 832x480 with 33 frames (2.06 s at 16 fps), 58 KB. FFprobe verified it, and all frames were finite with no NaN or blank output. `float16` gave clean frames, so `float32` was not needed.

**Visual check:** the clip shows a coherent, dark TV studio with red lighting, a round table and a slow camera push-in, which matches the prompt. It shows four people instead of the requested two, the figures are small, and there is very little motion (mean absolute frame difference 0.004). Clearer prompts are needed to control the number of people, the framing and the action.

**Notes**

- pip printed a dependency-resolver warning about Kaggle's preinstalled packages. It did not affect the run.
- Generation time grows with the number of frames and the resolution. The values above apply only to 33 frames at 832x480. Longer clips have not been measured yet.

## Offline verification

- Unit tests cover input validation, safe prompt embedding, report and video validation, and blank/NaN frame rejection.
- The kernel's `load_pipeline` and `render` functions were also run on a CPU with the pinned package set. They used a tiny, randomly initialised Wan pipeline (real diffusers code, fake weights).

## Limits

- Wan 2.1 1.3B produces 480p clips without sound. Quality is clearly below large models, and text and faces are often poor.
- Every run consumes free Kaggle GPU quota that is shared with Model Lab. The workflow has no schedule and only runs manually.
- Artifacts are temporary. Download clips you want to keep and do not commit MP4 files to this repository.
