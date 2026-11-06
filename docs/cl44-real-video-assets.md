# CL-44 — real LTX-2.5 asset install gate (PR 2/4)

The goal is one actual playable video clip by CL-46, subject to suitable hardware and access to model weights. CL-44 **does not render video**, reserve a GPU, download models on CI, or bypass runtime safeguards.

## Official weights, no GGUF mixing

Install only the pinned official `Lightricks/LTX-2.5` ComfyUI ConvRot stack: `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`, `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`, `vae/ltx-2.5-video-vae-conv-bf16.safetensors`, and `vae/ltx-2.5-audio-vae-bf16.safetensors`.

Source: https://huggingface.co/Lightricks/LTX-2.5/blob/main/README.md . This stack is **not** the experimental Q2_K GGUF stack and must not be passed to native LTX pipelines.

## Prepare a real machine

1. Install a matching ComfyUI runtime and LTXVideo nodes, and ensure `ComfyUI/main.py` exists. Provide `ffmpeg` on `PATH`. This script intentionally does not provision a cloud machine or install node plugins.
2. Run the safe plan: `python -m scripts.cl44_ltx_assets_installer --comfy-root /path/to/ComfyUI`.
3. Check the official model repository for an exact **40-character commit SHA**. The installer consults Hugging Face metadata and verifies LFS SHA-256 and declared sizes. It refuses unknown metadata, revisions that move, stacks above 80 GB, or insufficient disk (the remaining unverified assets plus 10 GB safety margin).
4. On the selected machine, install `huggingface_hub` separately, authenticate if the repository requires it, and only then explicitly approve downloads:

   `python -m scripts.cl44_ltx_assets_installer --comfy-root /path/to/ComfyUI --revision <40-char-commit-sha> --apply --confirm DOWNLOAD_LTX25_MODELS`

The script downloads into the ComfyUI `models/` tree, verifies size and SHA-256, and reuses previously verified files. Secrets and Hugging Face tokens are not logged to the JSON report. It does **not** certify the model will fit Kaggle GPUs or run within their storage quotas.

## Next two PRs

- CL-45 (PR 3/4): prepare the real first-frame / last-frame ComfyUI workflow and a single preflighted launch path, including explicit replacement of the workflow's video VAE with `ltx-2.5-video-vae-conv-bf16.safetensors`. Do not silently use the default DiffVAE or pretend a file exists.
- CL-46 (PR 4/4): attempt a minimal, actual MP4 render with the available supported hardware and report either the playable artifact or the exact blocking error. Do not add further research PRs before this attempt.

Official guidance: https://docs.ltx.io/open-source-model/integration-tools/comfy-ui/ and https://docs.ltx.io/open-source-model/reference/workflow-asset-compatibility .
