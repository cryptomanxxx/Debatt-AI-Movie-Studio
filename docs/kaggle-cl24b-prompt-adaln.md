# CL-24B: first video block with cross-attention prompt AdaLN

CL-24 failed safely on run [42173011113](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42173011113): the real model enables `cross_attention_adaln`, and the CL-24 probe intentionally rejected this unsupported variant before invoking the block. This is not a demonstrated VRAM issue.

CL-24B retries that block's **video branch** using the real `BasicAVTransformerBlock.forward` implementation, with both required synthetic AdaLN inputs:

- `v_timestep`: `[1, 2, 9 × video_width]` (six base AdaLN groups plus three cross-attention query groups).
- `v_prompt_timestep`: `[1, 2, 2 × context_width]` (cross-attention context scale and shift), validated against the real `prompt_scale_shift_table` shape `[2, context_width]`.

These dimensions follow ComfyUI's `apply_cross_attention_adaln` source. Block zero executes video self-attention, text cross-attention, feed-forward, residual gates and AdaLN with genuine GGUF model weights. All inputs remain **synthetic**; `v_pe=None` disables rotary positions, audio is disabled and no full diffusion inference/video generation is attempted. A passing result does not prove real context or full-length video fits in GPU memory.

Safety controls are unchanged from CL-24: SHA-256 verified 8.46 GB GGUF and cleanup, bounded weights, 180-second child watchdog, 12 GB sampled RAM, 4 GiB PyTorch-reported GPU allocation and no automatic Kaggle execution. To run after Codex review/merge, select **Kaggle CL-24B LTX Prompt AdaLN (manual)** on `main` and enter `YES`, `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`, `YES_PROMPT_ADALN`.
