# CL-18 — one real LTX weight matrix in bounded arithmetic

CL-17 successfully decoded a complete 4,194,304-parameter Q2_K matrix from 1.38 MB packed to 16.78 MB float32 and copied it to both Tesla T4 GPUs. CL-18 runs a **single `torch.mv` matrix-vector multiplication** using that matrix and a deterministic **synthetic** input vector. CPU and each GPU perform the same operation; results are compared numerically. This proves a limited mathematical computation on genuine model weights, **not** a valid LTX neural-layer forward pass, an LTX model load, or video inference.

## Manual run

GitHub Actions → **Kaggle CL-18 Real Weight Matrix-Vector (manual)**, branch `main`, with:

- `confirm_gpu_use=YES`
- `confirm_single_weight_download=YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_matrix_arithmetic=YES_MATVEC`

This uses Kaggle GPU quota and repeats one 8.46 GB weight file download; no scheduled run.

## Bounded checks

- Inherits the CL-13–17 integrity checks, 12-minute independent download deadline, temporary staging outside outputs, SHA-256 and cleanup.
- The whole Q2_K tensor is selected *before* decoding: compressed <=4 MiB, decoded <=64 MiB. A 2-D shape is required, with matrix dimensions between 2 and 8,192.
- Create a fixed input with `torch.linspace`, compute the CPU `torch.mv`, then compute GPU `torch.mv` on each T4 in turn. Compare with `torch.allclose(rtol=1e-3, atol=1e-3)`, report maximum absolute discrepancy.
- Require at least 8 GB free VRAM per GPU, <=128 MiB increased PyTorch allocation and <=512 MiB free-VRAM drop; clear GPU and mmap references on exceptions before file cleanup.
- Does not install or run an LTX model architecture, use tokenizer, text encoder, VAE, neural-layer activation/bias or ComfyUI inference graph. No videos are generated.

## Interpretation

If CL-18 passes, the test demonstrates a real matrix-vector operation with quantized LTX weights after GGUF decoding. It does not establish correctness of an LTX layer (the synthetic input and lack of bias/normalization are deliberately not authentic inference), actual model offloading, or end-to-end memory fit.
