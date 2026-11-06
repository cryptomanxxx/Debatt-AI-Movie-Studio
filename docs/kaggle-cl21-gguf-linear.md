# CL-21 — One real ComfyUI-GGUF linear layer on Kaggle T4

CL-20 established that ComfyUI could construct a CPU-offloaded LTX-2.5 ModelPatcher from the 8.46 GB Q2_K GGUF without a large resident-RAM or GPU allocation. CL-21 attempts one actual `GGMLOps.Linear` forward pass with a small synthetic input tensor and a real quantized layer selected from that constructed model.

**This is not a whole diffusion-model forward step, a valid LTX-2.5 inference, or video generation.** Such an experiment would additionally require valid conditioning, positional embeddings and a complete sequence of transformer blocks. We deliberately establish compatibility of the custom quantized layer's real `forward()` before attempting those operations.

The test selects the smallest eligible quantized layer with two dimensions, up to 8 million logical elements and both dimensions <=8192. Its input is synthetic, batch size one, on `cuda:0`. Output shape and finiteness are checked. The subprocess watchdog retains the 12 GB observed-RSS limit and 180-second deadline; PyTorch GPU peak allocations must remain <=1 GiB per GPU. GPU peak checks occur after the forward, so extremely short OOM spikes cannot be prevented. The downloaded checkpoint is verified and deleted in cleanup; no model files are uploaded.

After Codex approval and merge, run **Kaggle CL-21 Real GGUF Linear Forward (manual)** on `main` with:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_single_layer_forward`: `YES_SINGLE_GGUF_LAYER`

No automatic triggers. This consumes Kaggle T4 quota and repeats the model download. A failure should be analyzed before any full-model inference attempt.
