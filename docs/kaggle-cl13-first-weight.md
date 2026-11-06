# CL-13 — first real Q2 weight, one-file verification

CL-13 is a **manual, explicitly approved** Kaggle run. It tests that Kaggle can retrieve and SHA-256 verify one LTX-2.5 Q2 transformer checkpoint. It does **not** download the full four-file pack, start model inference, load LTX into GPU memory, or generate a video. The file is deleted before the Kaggle output is collected.

## Run

Select GitHub Actions → **Kaggle CL-13 First Q2 Weight Verification (manual)** → Run workflow, and enter both:

- `confirm_gpu_use`: `YES`
- `confirm_single_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`

The workflow requires the existing `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` GitHub secrets. It may use free Kaggle GPU quota and remote bandwidth. Do not schedule it automatically.

## Safety and reproducibility

- Exactly one file is targeted: `ltx-2.5-22b-distilled-Q2_K.gguf` from `unsloth/LTX-2.5-GGUF`. Model revision is resolved to a commit SHA. Repository must be ungated; no authentication workaround is used.
- The download needs Git-hosted metadata including file size and SHA-256. If unavailable, it stops without downloading.
- No file larger than **13 GB** is accepted. The local disk must have room for the file plus **4 GB reserve**; free space is checked during streaming. Transfer stops after **12 minutes**.
- A temporary `.part` file is checked against the declared byte length and SHA-256, then removed in `finally`. The downloaded model file never becomes a GitHub artifact.
- A missing checksum, timeout, network error, insufficient space, failed checksum, or failed cleanup makes the experiment fail. The JSON diagnostics are preserved where available.
- The existing CL-8 small dependency installation and source inventory run first. Torch and CUDA versions must remain unchanged and the source checkouts are removed before reporting.

## Interpretation

Passing CL-13 proves only that **one** real quantized checkpoint can be transferred and checked in Kaggle within the enforced limits. It does not prove the remaining model files fit, that two T4 GPUs can run LTX-2.5, or that a video can be generated. A future, separately reviewed experiment must test model loading and memory behavior.
