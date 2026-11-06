# CL-43 — pivot to a real LTX-2.5 video workflow

## Four-PR delivery budget

The project owner has set a limit of **four more PRs** before making an
actual video-generation attempt. CL-43 is **PR 1/4**. We will not spend
these PRs adding transformer blocks individually. CL-44 should prepare a
concrete runtime/asset setup, CL-45 should implement a bounded complete
workflow launch, and CL-46 should **attempt** to generate and export a tiny
real clip. The attempt is not a guarantee that Kaggle T4 has sufficient
VRAM/RAM/disk for the complete 22B model.

## Why switch paths?

The project's 48-block Q2_K GGUF experiments (CL-1 to CL-42) proved useful
pieces of native LTX computation. They are not a full text-conditioned video
generator: we have not integrated a compatible Gemma text encoder, video
VAE, full denoising path or MP4 export. Adding blocks alone does not solve
these missing components.

The *official* LTX-2.5 ComfyUI integration documents a lower-VRAM
`INT8 ConvRot` path that includes the transformer, compatible Gemma text
encoder and lighter convolutional video VAE. The first/last-frame workflow
`video_ltx2_5_flf2v` is single-stage and avoids the two-stage spatial
upscaler; it requires start/end images and a compatible prompt. We also
include the LTX audio VAE because official LTX-2.5 workflows can generate
audio. We will separately establish if it can be omitted in a video-only
configuration. Do **not** substitute the existing Q2_K GGUF into an official
INT8 ConvRot graph without a compatibility test.

Authoritative docs:
- https://docs.ltx.io/open-source-model/integration-tools/comfy-ui/
- https://docs.ltx.io/open-source-model/reference/workflow-asset-compatibility

## CL-43 deliverable

`scripts/cl43_video_runtime_preflight.py` performs **read-only, CPU-only**
checks for ComfyUI, ComfyUI-LTXVideo nodes, FFmpeg, minimum free disk and
the expected model filenames. It does not download assets or validate their
hashes, actual size, execution compatibility or memory fit. An empty GitHub
runner should correctly report **not ready**, not pretend that a clip was
generated. The check is also runnable in a prepared machine:

```bash
python -m scripts.cl43_video_runtime_preflight \
  --comfy-root /path/to/ComfyUI --output cl43-results/report.json
```

The true clip target is a short, low-resolution MP4 with a verifiable
nonempty output file; intermediate synthetic tensors are not sufficient.
No Kaggle job or model download is triggered by this PR.
