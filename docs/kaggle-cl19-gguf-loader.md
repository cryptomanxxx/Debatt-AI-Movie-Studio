# CL-19 — ComfyUI-GGUF state-dict reader, without full model loading

CL-18 proved one actual GGUF Q2_K weight matrix can be dequantized and multiplied by a synthetic vector on both Kaggle T4 GPUs. CL-19 tests the next distinct integration boundary: whether the real `city96/ComfyUI-GGUF` `gguf_sd_loader` parses the 8.46 GB LTX-2.5 Q2_K checkpoint into an mmap-backed tensor dictionary.

This **does not** call `comfy.sd.load_diffusion_model_state_dict`, construct a model patcher, perform inference, or produce video. Those operations are deliberately out of scope, because full-model memory requirements have not been established. The loader runs in a separate process watched every 50 ms by its parent, which kills it after 120 seconds or when sampled resident RAM exceeds 3 GB. The child reports PyTorch's peak allocated GPU memory, which must remain below 16 MiB per device. Polling cannot guarantee detection of arbitrarily brief RAM spikes or prevent every possible OOM, but the separate process permits termination on observed violations. The verified checkpoint is staged under `/kaggle/temp` and deleted before Kaggle publishes outputs. A failed probe fails validation rather than being marked successful.

The report records tensor count, architecture, memory deltas, and raw shapes/types of three BF16 connector/position parameters reported in upstream issue `city96/ComfyUI-GGUF#477`. The shape audit is diagnostic: it does not claim to fix upstream LTX-2.5 model loading.

After merging the PR, manually open **Actions → Kaggle CL-19 ComfyUI-GGUF State Dict (manual) → Run workflow** and enter exactly:

- `confirm_gpu_use`: `YES`
- `confirm_single_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_loader_probe`: `YES_GGUF_LOADER`

This consumes another Kaggle T4 quota allocation and downloads the full checkpoint again. No automatic trigger or background rerun is configured.
