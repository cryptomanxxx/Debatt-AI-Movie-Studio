# CL-32 — native LTX video latent output after eight blocks

CL-32 extends the successful CL-31 pathway: tiny *synthetic* latent `[1, in_channels, 1, 2, 2]` → real LTX video patchify and GGUF projection → real video rotary positions → eight consecutive real-weight video transformer blocks (0–7) → **native video output modulation, projection and unpatchify**.

The output path uses the actual `LTXVModel._process_output` method on the audio-video checkpoint, bypassing the AV override. `embedded_timestep` is a small synthetic zero tensor with per-token hidden width; this is **not** the model's diffusion scheduler or learned timestep embedding. The native output LayerNorm, GGUF-backed `proj_out` and patchifier `unpatchify` are executed. A successful result must return a finite latent with the *same shape* as the tiny input latent.

This test does **not** run VAE encoding/decoding, text encoding, realistic token counts, a denoising loop, true image/video input, audio inference or video generation. Matching the latent shape is evidence for interface compatibility, not that a visually meaningful video can be made.

Manually run GitHub Actions → **Kaggle CL-32 LTX Latent Output (manual)** with these explicit inputs:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_latent_output`: `YES_LATENT_OUTPUT`

Inherited safety controls: the 8.46 GB SHA-verified Q2_K GGUF download + cleanup, maximum 80M elements per GGUF projection, PyTorch GPU peak capped at 4 GiB, sampled child RSS capped at 12 GB, 180-second child watchdog, 35-minute Actions job timeout and 7-day diagnostic-artifact retention. No automatic execution is configured.
