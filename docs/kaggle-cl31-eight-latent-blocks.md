# CL-31 — real latent patchify → RoPE → eight video transformer blocks

This experiment combines CL-30's successful model-native latent patchification and CL-29's successful eight consecutive rotary-attention video transformer blocks. We start from a **synthetic tiny LTX latent** tensor, with shape `[1, in_channels, 1, 2, 2]`, not decoded video frames or VAE-encoded pixels. The model's genuine `LTXVModel._process_input` video-only path generates four projected tokens and their middle-grid video positions. The model's base video RoPE routine then prepares the rotary embeddings. These are passed through the real-weight video branches of **blocks 0–7**, with synthetic text and AdaLN conditioning. No audio path is exercised.

This is **not** a full inference, sampling, VAE encode/decode, or video generation run. A positive result only shows the tiny prepared latent can flow through those eight real GGUF-backed transformer blocks within resource bounds; practical video token counts and diffusion loops remain untested.

Run manually in GitHub Actions → **Kaggle CL-31 Eight Latent Video Blocks (manual)** with:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_eight_latent_blocks`: `YES_EIGHT_LATENT_BLOCKS`

The existing safeguards remain: pinned 8.46 GB Q2_K GGUF download + SHA validation and cleanup, 80M logical element bound per GGUF projection, 4GiB PyTorch GPU peak, 12GB sampled child RSS, 180s child watchdog and 35-minute CI job. The workflow has no automatic trigger and stores diagnostic artifacts for 7 days. Validate all eight block outputs and memory samples before considering CL-31 passed.
