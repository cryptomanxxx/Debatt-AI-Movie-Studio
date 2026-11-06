# CL-35 — checkpoint-backed LTXAV timestep embedding probe

CL-35 retains the verified CL-34 two-pass, tiny-latent experiment and adds a **separate native timestep embedding probe**. ComfyUI's LTXAVModel `_prepare_timestep` scales the supplied timestep, runs the checkpoint-backed `adaln_single` and `prompt_adaln_single` modules, and can compress video modulation per frame. The probe calls this method for two **synthetic** sigma-like values, `1.0` and `0.96875`, on one frame with four spatial patches. It checks the video modulation, output embedding and prompt modulation dimensions, finite values, and a nonzero response to the changed timestep.

Importantly, the newly computed embeddings are **not yet wired into the two transformer passes**. The CL-34 passes retain synthetic zero timestep tensors. The sigma sequence is **not validated as the real LTX scheduler**, the full denoising procedure is not implemented and no video frames are generated. CL-35 establishes a narrow checkpoint-backed embedding interface so that a later CL can replace synthetic modulation with native tensors safely.

Reference implementation: [ComfyUI LTXAVModel `_prepare_timestep`](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/lightricks/av_model.py), including the `CompressedTimestep` video embedding wrapper. This workflow uses the existing pinned ComfyUI package from preceding CLs; results depend on that revision, not an unpinned source download.

Manually run GitHub Actions → **Kaggle CL-35 Native Timestep Probe (manual)** with:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_native_timestep`: `YES_NATIVE_TIMESTEP_PROBE`

**Explicit, narrowly approved exception (after Codex P2):** The model-native `adaln_single.linear` has a fixed GGUF matrix shape `(36864, 4096)`, exactly **150,994,944 logical elements**, exceeding the previous 80M element projection cap. The user authorized investigating this necessary architecture-specific exception. Before calling the embedding API, CL-35 now inventories both newly executed native AdaLN modules, verifies all linear weights are GGUF-backed, checks the sole exception by exact module path and shape, and enforces **80M elements on all their other projections**. Unexpected shapes or modules fail closed. This does not increase the projection limit elsewhere or relax memory bounds.

Other inherited safety: 8.46 GB pinned SHA-verified GGUF and cleanup, maximum 80M logical elements for all non-exempt GGUF projections, 4 GiB PyTorch CUDA allocation cap, 12 GB child RSS cap, 180-second child watchdog, 35-minute Actions timeout, seven-day diagnostics retention. No automatic GPU execution.
