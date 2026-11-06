# CL-17 — one bounded complete Q2_K tensor

CL-16 successfully dequantized one **84-byte Q2_K block** to 256 float32 values and copied the resulting 1,024 bytes to both Tesla T4 GPUs. CL-17 extends this to **one complete, size-bounded tensor** (never the entire model or layer stack).

## Run manually

GitHub Actions → **Kaggle CL-17 Bounded Whole Q2 Tensor (manual)**, branch `main`, and enter:

- `confirm_gpu_use=YES`
- `confirm_single_weight_download=YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_whole_tensor=YES_WHOLE_TENSOR`

This repeats the 8.46 GB model-file download and uses Kaggle free GPU quota. It is not scheduled or auto-triggered.

## Guardrails

- Reuses CL-13–16 model integrity checks, 13 GB cap, 12-minute independent download deadline, staging outside Kaggle outputs and `finally` cleanup.
- Chooses the smallest *complete* Q2_K tensor with packed size at most **4 MiB**, decoded size at most **64 MiB**, and a valid 84-byte block multiple. Refuses to continue if no such tensor exists.
- Decodes the entire selected tensor using GGUF's actual Q2_K decoder to `float32`; validates the exact expected expansion and finite values.
- Transfers the decoded tensor to the two Tesla T4 GPUs sequentially. Before each copy, requires at least 8 GB free VRAM; bounds added PyTorch allocation to **128 MiB** and observed free-VRAM drop to **512 MiB**. Releases GPU and mmap references before file cleanup, including on failure.
- Writes only diagnostic JSON; no encoded video, text encoder/VAE download, ComfyUI graph, full model placement, or inference.

## Interpretation

Success would establish that a complete real Q2_K tensor of limited size can be decoded and placed in either GPU's memory. It cannot establish that all 22B model parameters, activations, intermediate weights or runtime caches fit. Nor does it show that ComfyUI-GGUF's tensor placement and offloading work end-to-end.
