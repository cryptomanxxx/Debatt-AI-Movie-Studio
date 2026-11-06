# CL-27: eight consecutive GGUF-backed video transformer branches

CL-26 ran blocks 0–3 successfully with two synthetic video tokens and stable PyTorch GPU allocation per block. CL-27 extends exactly the same bounded experiment to blocks 0–7, passing each block's video result into the next. Report fields include each block's index, output finiteness, PyTorch allocations after the block, and cumulative GPU-allocation peak. These are **not** comprehensive measurements of physical VRAM or proof that real video sequences fit.

Real LTX-2.5 Q2_K GGUF model weights are used, including AdaLN-normalized attention, text cross-attention and feed-forward. However all tokens and prompt/timestep conditioning are synthetic, video RoPE is disabled (`v_pe=None`), the audio branch is disabled, and no full model forward, sampler, VAE or video generation is attempted.

Safety limits remain at 4 GiB PyTorch GPU peak, 12 GB sampled child RSS, 180-second watchdog, verified SHA-256 for the 8.46 GB GGUF file, source integrity and cleanup, with individual matrices capped at 80 million logical elements. GPU use and model download require explicit manual dispatch with `YES`, `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`, `YES_EIGHT_BLOCKS`. No automatic Kaggle job on PR or merge.

After Codex approval and merge, run **Kaggle CL-27 LTX Eight Video Blocks (manual)** from `main`. Share the resulting run URL.
