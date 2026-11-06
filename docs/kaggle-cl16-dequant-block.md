# CL-16 — one real Q2_K quantization block

CL-15 successfully moved 8,192 raw GGUF bytes to each Kaggle Tesla T4. CL-16 tests **actual dequantization** of precisely one 84-byte Q2_K block from a real transformer tensor, which should decode to 256 float32 elements (1,024 bytes). Each GPU receives only that small decoded array. This is **not** neural-network layer execution or a full model load.

## Manual approvals

Run GitHub Actions → **Kaggle CL-16 One Q2 Block Dequantization (manual)** on `main` with:

- `confirm_gpu_use=YES`
- `confirm_single_weight_download=YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_gpu_dequant_test=YES_Q2_BLOCK`

Like earlier probes, this repeats the **8.46 GB** checkpoint download and uses free Kaggle GPU quota. It does not trigger automatically.

## Scope and limits

- Inherits CL-13–15 SHA-256 verification, 13 GB single-file cap, 12-minute hard network deadline, temporary storage outside retrievable outputs and final file cleanup.
- Locates a real tensor with GGUF `Q2_K` storage and reads **only the first 84-byte compressed block**; checks the decompressor returns exactly 256 finite `float32` values.
- Measures PyTorch memory allocation when copying 1,024 decoded bytes to each of the two Tesla T4 devices, sequentially. Requires at least 8 GB free VRAM per device; limits allocated memory delta to 16 MB and observed free-memory drop to 1 GB.
- Releases tensor-backed mmap references even on error. Does not dequantize an entire tensor or layer; no text encoder/VAE, ComfyUI inference or video is attempted.

## Interpretation

Success establishes that **one real Q2_K block** can be decoded correctly by the GGUF library and transferred as floating-point values to both GPUs. It cannot establish whole-model VRAM requirements: activations, caches, runtime offloading and full model compatibility are still untested.
