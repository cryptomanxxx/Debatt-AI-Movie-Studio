# CL-29 — eight video blocks with real rotary positions

This bounded experiment combines the *validated* CL-27 path through eight consecutive LTX-2.5 video transformer blocks (0–7) with CL-28's *validated* video-only rotary positional embedding (RoPE). All blocks process the same four synthetic latent video tokens in sequence, using one shared RoPE tensor computed from four synthetic start/end coordinate pairs, 25 fps and the checkpoint's `use_middle_indices_grid=True` setting.

The real model weights are loaded from the SHA-verified LTX-2.5 Q2_K GGUF. Attention (including rotary Q/K), cross-attention and FF are executed with synthetic text context, timestep and prompt AdaLN conditioning. The audio branch stays disabled. This does **not** execute text encoding, VAE decoding, diffusion steps, real video conditioning or video generation.

Manual GitHub Actions workflow: **Kaggle CL-29 Eight Video Blocks with RoPE (manual)**. Three required opt-in fields:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_eight_rope_blocks`: `YES_EIGHT_ROPE_BLOCKS`

The inherited safeguards include an approximately 8.46 GB weight download with pinned SHA-256 verification and cleanup, a 180-second child watchdog, sampled child RSS capped at 12 GB, PyTorch CUDA allocation capped at 4 GiB, and each GGUF logical projection limited to 80 million elements. The workflow uses a 35-minute maximum and stores only diagnostics for 7 days. No workflow starts automatically. Inspect each block's finite output and memory stats before considering CL-29 successful.

Passing CL-29 would show **eight successive transformer video branches using RoPE** on tiny synthetic inputs, not feasibility of a practical full-video pipeline.
