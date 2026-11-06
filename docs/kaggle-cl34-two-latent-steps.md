# CL-34 — two synthetic latent updates with a fresh model pass

CL-34 builds on successful CL-33. Step 1 runs eight real-weight video transformer blocks, native output projection/unpatchify, then a synthetic Euler update from sigma `1.0` to `0.96875`. **Step 2 takes the updated latent**, converts it to the model's float16 input type, and executes a *fresh* native patchify, RoPE and eight-block video model evaluation, native output/unpatchify, followed by a second update from `0.96875` to `0.9375`.

This is two separate model evaluations and two linked updates, not a replay of a cached velocity. Each pass processes only four latent video tokens and eight of the model's blocks. Text, timestep/prompt modulation and sigma schedule are synthetic; no true learned diffusion timestep, full LTX diffusion solver, full model, VAE decode, audio or visible video is involved. The check establishes shape, finite values, a nonzero update and a bounded numerical residual, **not** physically meaningful denoising or video generation.

Run manually from GitHub Actions → **Kaggle CL-34 Two Latent Steps (manual)** with:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_two_steps`: `YES_TWO_LATENT_STEPS`

Inherited guardrails: pinned SHA-verified 8.46 GB Q2_K GGUF and cleanup, 80M maximum logical elements per GGUF projection, 4 GiB PyTorch CUDA allocation cap, 12 GB sampled child RSS cap, 180-second child watchdog, 35-minute Actions timeout and seven-day diagnostic retention. No automated GPU execution.
