# CL-30 — actual LTX video latent patchification and projection

CL-30 introduces the **real model input-preparation path** using `LTXVModel._process_input` on the loaded LTX-2.5 audio-video model. The input is intentionally tiny and **synthetic**: `[batch=1, in_channels=128, time=1, height=2, width=2]`. Model-specific patchification, video pixel coordinates, GGUF-backed `patchify_proj`, and optional model positional handling produce four projected transformer video tokens and model-generated middle-grid start/end coordinates. The real LTX video-only RoPE creator consumes those coordinates; block 0 then performs its real-weight video branch with synthetic text and AdaLN conditioning.

This checks the transition from **synthetic latent data to authentic LTX token preparation**. It is **not** pixel/video encoding, an actual VAE output, real text encoding, a denoising loop, decoding, audio inference or video generation. The real latent geometry is highly simplified and too small for any claim about useful video generation.

Run manually from GitHub Actions → **Kaggle CL-30 LTX Latent Patchify (manual)**. Three exact opt-ins:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_latent_patchify`: `YES_LATENT_PATCHIFY`

Inherited safeguards remain in place: pinned SHA-verified 8.46 GB GGUF download and cleanup, max 80M logical elements per GGUF projection, 4 GiB PyTorch GPU cap, 12 GB sampled child RSS cap, 180-second child watchdog, 35-minute CI job timeout, and diagnostics-only artifacts kept for 7 days. The workflow has no automatic trigger. A successful Actions run alone is not sufficient: inspect the `real_patchify_projection`, `coordinate_source` and all block/guard fields.
