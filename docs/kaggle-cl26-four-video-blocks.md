# CL-26: four consecutive LTX-2.5 video transformer blocks

CL-25 successfully ran real GGUF-backed video branches of blocks 0–1 with two synthetic tokens. CL-26 runs blocks 0–3 consecutively, passing each video output directly to the next block. It records the per-block PyTorch GPU allocation **after** the block and the cumulative PyTorch allocation peak, in addition to existing sampled process RSS and global GPU peak. These metrics do not measure total device VRAM or allow reliable runtime extrapolation to a full video.

Each block receives matching synthetic text context and zero AdaLN/prompt timestep features, validated against its actual 9-group and 2-group scale-shift tables. No real RoPE, audio, text encoder, denoising timestep, diffusion sampler, VAE or video generation occurs. The test uses only two video tokens and is not a full model forward.

Limits inherited unchanged: per GGUF linear matrix <=80 million logical elements, SHA-256 verified 8.46 GB model plus cleanup, 12 GB sampled child RSS, 180-second child watchdog, 4 GiB PyTorch-reported GPU allocation. Kaggle is dispatched manually only after `YES`, `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`, and `YES_FOUR_BLOCKS`. No PR/merge-triggered GPU job.

After Codex review and merge, launch **Kaggle CL-26 LTX Four Video Blocks (manual)** on `main` and send the run URL for analysis.
