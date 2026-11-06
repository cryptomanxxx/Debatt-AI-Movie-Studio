# CL-48 — larger model: Wan 2.2 TI2V 5B on Kaggle

CL-47 showed that a Kaggle T4 has headroom with Wan 2.1 T2V 1.3B: 11.7 of 15.6 GB peak VRAM, 33.7 GB RAM and about 1 TB of disk. CL-48 tries the next size up, **Wan 2.2 TI2V 5B** (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`, Apache 2.0). It uses the same verified GitHub → Kaggle → FFprobe flow.

## Why 5B, and why it may fit

- The transformer is about 10 GB in fp16. Model CPU offload keeps only one component on the GPU at a time: the text encoder (~11 GB in bf16, already measured in CL-47), then the transformer, then the VAE.
- The Wan 2.2 VAE compresses 16× spatially, compared with 8× for Wan 2.1. At 832x480 the transformer therefore sees a quarter as many tokens per frame. A 2-second clip at 832x480 is 5,070 tokens, compared with 14,040 for the CL-47 run.
- Based on CL-47's measured ~15 s per step, the estimate is roughly 20–60 s per step for 5B at the default settings. This is **not measured**; the first run will show the real figure.
- Larger models such as Wan 14B or LTX-2.5 22B do not fit a 16 GB card without heavy quantization, and would take more than an hour per short clip.

## Run it

**Actions → Kaggle CL-48 Wan 2.2 TI2V 5B Scene (manual) → Run workflow**, then type `YES` and `YES_DOWNLOAD_WAN22_TI2V_5B`.

| Input | Default | Notes |
| --- | --- | --- |
| `num_frames` | 49 | 24 fps: 25 = 1 s, 49 = 2 s, 73 = 3 s, 121 = 5 s |
| `resolution` | 832x480 | 1280x704 and 704x1280 are the model's native 720p and much slower |
| `steps` | 30 | 10–50 (the model authors use 50) |
| `transformer_dtype` | float16 | Use bfloat16 if fp16 gives black or broken frames. bf16 runs correctly on a T4 (CL-47's text encoder did), but slowly |

A guard rejects combinations above **15,000 latent tokens** before any GPU time is used. At 720p this means at most 49 frames (11,440 tokens), and 73 or 121 frames are refused. At 832x480 all frame counts are allowed; 121 frames is 12,090 tokens.

The artifact `movie-studio-cl48-wan22-scene-<run id>` (7 days) contains `wan22-scene.mp4`, a copy with a silent audio track for `assemble`, the JSON report and the Kaggle logs. The Kaggle time limit is 2 hours.

## Implementation

The CL-47 module is now generic. A `Profile` holds everything that differs between models: model ID, kernel slug, fps, allowed resolutions and frame counts, dtypes, step limit, scheduler shift, token budget and timeouts. CL-47 (`WAN21_1_3B`) behaves exactly as before. CL-48 (`scripts/kaggle_cl48_wan22_scene.py`) only defines the `WAN22_TI2V_5B` profile. It keeps the scheduler shift from the model repository instead of overriding it. `WanPipeline` handles Wan 2.2 TI2V through its `expand_timesteps` setting, which is read from the repository's `model_index.json`.

## Verified, and not yet verified

- Offline unit tests cover profile isolation, geometry (multiples of 32, valid frame counts), the token budget, input bounds, and that CL-47's configuration is unchanged.
- The kernel's `load_pipeline` and `render` ran on a CPU against a tiny, randomly initialised Wan 2.2 TI2V-shaped pipeline (48-channel residual VAE with 16× compression, `expand_timesteps=True`). The run produced a 25-frame H.264 MP4 that passed `probe_video`. The same CPU test of the CL-47 profile still passes after the refactor.
- **Not yet verified:** a real Kaggle run. Download size and time, load time, seconds per step, peak VRAM, RAM headroom and fp16 output quality are unknown until the first run.

## Possible next optimisation

The second T4 is still unused. If VRAM becomes the limit, the text encoder can run on GPU 1 so the transformer has GPU 0 to itself. This is not implemented, because model CPU offload is expected to be enough.
