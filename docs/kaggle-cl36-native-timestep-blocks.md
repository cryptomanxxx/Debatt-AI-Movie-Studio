# CL-36 — native LTXAV AdaLN inside eight GGUF transformer blocks

CL-36 extends the verified CL-35 pathway with **one additional bounded video model evaluation** that uses checkpoint-backed `_prepare_timestep` output as actual transformer modulation. The same synthetic one-frame, four-token latent is patchified again, with native video RoPE. Video blocks 0–7 receive the native compressed per-frame modulation and native prompt timestep; the native output embedding is broadcast over four tokens and used by `LTXVModel._process_output`.

The earlier CL-34/35 reference calculations remain in the child, so this is a **third independent model pass**, not a replacement of the two earlier synthetic-conditioning passes. The only newly tested pass uses native timestep conditioning. Its scalar timestep `1.0`, text context and latent are still **synthetic**. There is no full model, real sampler/scheduler, VAE or generated video. Success proves that the native embedding interface and eight-block video branch can be connected, not that a real diffusion update works.

Preflight from CL-35 explicitly checks the user-authorized sole `adaln_single.linear` exception `(36864, 4096)` = 150,994,944 elements. All other GGUF projections remain limited to 80M elements. Other limits: 8.46 GB SHA-pinned GGUF with cleanup, 4 GiB PyTorch GPU allocation, 12 GB child RSS, 180-second watchdog, 35-minute GitHub Actions timeout and seven-day diagnostic retention.

Run manually in GitHub Actions → **Kaggle CL-36 Native Timestep Blocks (manual)**:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_native_blocks`: `YES_NATIVE_TIMESTEP_BLOCKS`

No automatic GPU execution.
