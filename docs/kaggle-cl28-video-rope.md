# CL-28 — bounded real video RoPE

CL-28 tests the actual ComfyUI LTX-2.5 positional embedding constructor and its rotary attention path in **one real-weight video transformer block (block 0)**. The coordinates are four synthetic `(frame, y, x)` positions expressed as start/end pairs (`[batch, 3, tokens, 2]`) for the checkpoint's `use_middle_indices_grid=True` mode and use an explicit 25 fps scaling. The four latent video tokens, text context and prompt AdaLN/timestep tensors are synthetic. This is **not** real text encoding, video tokenization, video generation, diffusion sampling or an audio forward pass.

The probe uses the existing vetted LTX-2.5 Q2_K GGUF weight download and bounded GGUF matrix conversion. It explicitly calls `LTXBaseModel._prepare_positional_embeddings(root, ...)` (the single-video implementation inherited by `LTXVModel`), bypassing the incompatible `LTXAVModel` override that expects video/audio coordinate pairs. It passes the resulting `(rotation_matrix, split_mode)` to `BasicAVTransformerBlock(..., v_pe=pe)` with the video branch enabled. It checks finite output, correct tensor shapes and the existing attention/FF weight evidence. The RoPE flag only records that the real model path was used; it is **not** proof of video quality or full inference feasibility.

Run manually from GitHub Actions → **Kaggle CL-28 LTX Video RoPE (manual)** → Run workflow, with these explicit values:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_video_rope`: `YES_VIDEO_ROPE`

No workflow auto-triggers. Expected budget guards inherited from CL-24B include the 8.46 GB artifact verification and cleanup, a 180-second child watchdog, 12 GB sampled child RSS maximum, 4 GiB PyTorch GPU peak allocation maximum, and bounded matrix allocations at 80 million logical entries. The GitHub job has a 35-minute upper bound and persists only diagnostic artifacts for seven days. Abort/failure must not be interpreted as a passed experiment.

If CL-28 passes, a later test can carry the positional embeddings through multiple blocks. Real video generation still requires a real latent/video conditioning path, denoising loop, decoder and feasible total GPU/CPU memory.
