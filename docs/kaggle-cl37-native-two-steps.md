# CL-37 — two linked native-AdaLN-conditioned latent updates

CL-37 combines CL-34's latent feedback with CL-36's native model timestep embedding. A tiny synthetic `[1,128,1,2,2]` latent is run through eight actual GGUF-backed video blocks (0–7) with LTXAV `_prepare_timestep` conditioning at synthetic sigma `1.0`. The native model output is used for one bounded float32 Euler-style update to `0.96875`. That updated latent is cast to model input dtype, passed through native patchify and **another eight-block evaluation**, now with a freshly computed native timestep embedding at `0.96875`, then native output and a second update to `0.9375`.

The new linked pair reuses the successful CL-36 first native-conditioned pass and appends its second pass. The child still executes the inherited earlier *synthetic-conditioning* CL-34 reference passes, so **four video eight-block evaluations total** occur. Two of these form the new native-conditioned linked pair.

The sigma sequence is intentionally synthetic. **This is not the true LTX scheduler or a validated denoising algorithm.** Text and latent inputs are synthetic; only eight transformer blocks are used; no full model, VAE, audio or video frames are generated. This is a compatibility and numerical stability experiment.

Inherited constraints: SHA-pinned 8.46 GB Q2_K GGUF with cleanup; user-approved single-matrix exception `(36864,4096)` for `adaln_single.linear`, preflight all new GGUF AdaLN projections and 80M cap for the rest; 4 GiB PyTorch CUDA allocation cap, 12 GB child RSS cap, 180-second watchdog, 35-minute Actions timeout, seven-day diagnostic retention.

Run manually from GitHub Actions → **Kaggle CL-37 Native Two Steps (manual)**:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_native_two_steps`: `YES_NATIVE_TWO_STEPS`

No automatic GPU execution.
