# CL-24: bounded first LTX-2.5 video transformer block branch

After the successful CL-22 feed-forward and CL-23 video self-attention probes, CL-24 invokes the real `BasicAVTransformerBlock.forward` method on the **video path** of block zero. It exercises AdaLN normalisation and residual gates, self-attention, text cross-attention and feed-forward, using real Q2_K GGUF model weights. Exactly two artificial video tokens and two artificial text context tokens are supplied, alongside zero synthetic AdaLN timestep features. The model's `cross_attention_adaln` variant is rejected rather than silently approximated.

**Not tested:** real text encoder embeddings, real denoising timestep embeddings, RoPE video positions (`v_pe=None`), audio path, video/audio cross attention, multiple transformer blocks, diffusion sampler, VAE or actual video output. The full LTX model is **not** run. Even success with 2 tokens says little about memory for a full video sequence.

Safety limits: Kaggle dispatch only after typing `YES`, `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`, `YES_VIDEO_BLOCK`; SHA-256 verified 8.46 GB model and cleanup; GGUF matrices limited to 80 million logical elements each; child watchdog 180s, sampled RSS 12 GB; PyTorch-reported GPU allocations at most 4 GiB (not a measurement of all GPU memory). Existing concurrency lock prevents overlapping GPU probes. No automatic Kaggle execution on PR or merge.

After passing Codex review and merging, manually run **Kaggle CL-24 LTX Video Block (manual)** from `main` and share its GitHub Actions URL. Failures should be diagnosed without automatically raising resource limits.
