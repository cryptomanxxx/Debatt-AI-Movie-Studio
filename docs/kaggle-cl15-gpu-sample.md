# CL-15: bounded GPU memory sample of real Q2 GGUF data

CL-13 verified the 8.46 GB LTX-2.5 Q2 transformer. CL-14 parsed all 4,349 tensor descriptions and ten metadata fields. CL-15 exercises CUDA transfer on both Tesla T4 cards using only **one tiny raw packed tensor's bytes** from the GGUF file. This is a hardware plumbing experiment, **not** neural-network model loading, ComfyUI-GGUF layer execution, dequantization, or video generation.

## Manual run

Select GitHub Actions → **Kaggle CL-15 Bounded Q2 GPU Tensor Sample (manual)** on `main`, enter three confirmations:

- `confirm_gpu_use=YES`
- `confirm_single_weight_download=YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_gpu_tensor_copy=YES_GPU_SAMPLE`

CL-15 repeats a full **8.46 GB download** to verify integrity, parses the GGUF index, then copies one raw tensor representation of **1 KiB–4 MiB** to GPU 0 and GPU 1 *sequentially*. The file is deleted after the run. This consumes Kaggle free GPU quota. There is no scheduled trigger.

## Guards

- Inherits the ungated-repository, fixed file, pinned HF revision, SHA-256, 13 GB size cap, 12-minute download deadline and `/kaggle/temp` staging from CL-13/14.
- Requires exactly two CUDA GPUs and at least 8 GB free memory on each before transfer.
- Caps PyTorch's added allocated memory at 16 MB and observed device-free memory drop at 1 GB per sample; records before/during measurements and releases the GPU sample in `finally`.
- No torch model, decoder, text encoder, neural-network forward pass or video file is created. GGUF data are copied as packed bytes, **not converted into meaningful inference weights**.
- The diagnostics verifier rejects a missing successful sample, incomplete download, failed checksum or cleanup, or changed Torch/CUDA version.

## Interpretation

Passing CL-15 establishes that PyTorch can move a small, real piece of the quantized checkpoint's raw storage onto both GPUs within resource guards. It **does not show** that ComfyUI-GGUF can load and dequantize the model, distribute layers between GPUs, or infer frames. Those require separately reviewed, much more demanding tests.
