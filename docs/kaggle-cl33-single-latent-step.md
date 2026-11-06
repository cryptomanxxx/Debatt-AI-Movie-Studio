# CL-33 — one synthetic Euler-style latent update

CL-33 extends the verified CL-32 pathway: a **tiny synthetic** 1×128×1×2×2 latent → native LTX video patchification and GGUF projection → eight real-weight transformer blocks (0–7) with native video RoPE → native output projection and unpatchify → **one bounded illustrative Euler-style latent update**.

The update is `x_next = x + (sigma_next - sigma_current) * model_output`, with a fixed *synthetic* sigma pair `1.0 → 0.96875`. Its arithmetic uses float32 and verifies finite values, unchanged latent shape, a nonzero difference and an analytical residual at most `1e-5`. The model output is produced by only eight of the full network's blocks with a **synthetic zero timestep embedding** and is **not established to be a valid diffusion velocity prediction**. This is an integration/arithmetic test only, **not** a validated LTX sampling algorithm. It does not invoke the real diffusion scheduler, use learned timestep embeddings, execute the full model, complete multiple denoising steps, run a VAE, generate frames or make a video.

Manually select GitHub Actions → **Kaggle CL-33 Single Latent Step (manual)** and enter:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_single_step`: `YES_SINGLE_LATENT_STEP`

Inherited safeguards: explicitly approved 8.46 GB SHA-verified Q2_K GGUF download and cleanup; maximum 80M logical elements per GGUF projection; 4 GiB sampled PyTorch CUDA-allocated peak limit; 12 GB child RSS limit; 180-second child watchdog; 35-minute Actions job timeout; diagnostics retained seven days. No automatic GPU execution.
