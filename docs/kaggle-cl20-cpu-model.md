# CL-20: Attempt LTX-2.5 GGUF model construction on CPU

CL-19 succeeded: the actual GGUF loader indexed 4,349 LTX-2.5 Q2_K tensors in 9.4 seconds, with 0.93 GB sampled peak resident RAM and zero PyTorch allocated GPU memory. CL-20 attempts the next step: `comfy.sd.load_diffusion_model_state_dict` with `GGMLOps` and both `load_device` and `offload_device` explicitly set to CPU. The goal is to obtain a model patcher, **not** to load weights to GPU, execute a forward pass, or render video.

Because this is a genuine 22B model construction attempt, memory usage and runtime may be significantly higher. A separate child process is watched at 50 ms intervals and killed when *observed* RSS exceeds **12 GB** or after **180 seconds**. The child reports peak PyTorch CUDA allocations (max **128 MiB per GPU**). RSS polling is not a hard cgroup limit: a very fast allocation spike or operating-system OOM remains possible. Therefore this experiment is explicitly opt-in, may fail, and should not be run automatically. The verified 8.46 GB checkpoint is deleted from `/kaggle/temp` in cleanup. Outputs contain diagnostics only.

An upstream LTX-2.5 / ComfyUI-GGUF compatibility issue can also cause model construction to fail before memory limits are reached. Such a failure is evidence about the integration boundary, not evidence of successful inference.

After Codex approval and merge, run **Actions → Kaggle CL-20 CPU LTX Model Construction (manual)** on `main` with:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_cpu_construction`: `YES_CONSTRUCT_CPU_MODEL`

No workflow will start without all three explicit approvals. Wait for Codex review before running.
