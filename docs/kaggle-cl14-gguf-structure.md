# CL-14: actual Q2 GGUF structure, without neural-network inference

CL-13 proved that Kaggle can retrieve and SHA-256 verify the 8,459,002,240-byte LTX-2.5 Q2 transformer. CL-14 repeats that **single-file** download, then uses the installed `gguf` reader to parse the real checkpoint's tensor index and metadata with read-only memory mapping. This is a necessary format compatibility probe, **not** a successful ComfyUI model load, GPU placement, or video generation.

## Manual opt-in

Run GitHub Actions → **Kaggle CL-14 Q2 GGUF Structure (manual)** with branch `main` and both text confirmations:

- `confirm_gpu_use=YES`
- `confirm_single_weight_download=YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`

This uses Kaggle free GPU quota and downloads 8.46 GB again. No automatic trigger, additional model files, paid inference API, or video generation is included.

## What it checks

- Reuses CL-13's ungated-repository, pinned revision, SHA-256 and file-size controls.
- Uses the same 13 GB file ceiling, 4 GB temporary disk reserve and independent 12-minute hard download deadline.
- Stages under `/kaggle/temp`, outside retrievable outputs; removes the checkpoint after reading.
- Parses the GGUF metadata/tensor index with `gguf.GGUFReader` in read-only mode and records the tensor count. It does **not** instantiate an LTX torch model, move tensors to VRAM, or run inference.
- Validates final diagnostics and fails the workflow if download, checksum, parse, cleanup or invariant checks fail.

## What follows

If CL-14 succeeds, a separately reviewed test can explore ComfyUI-GGUF's real tensor loading and bounded CPU/GPU offloading. That will require explicit resource monitoring and likely the remaining encoder/VAE files. File size alone does not predict VRAM use, and the two T4 GPUs are not a pooled 32 GB device.
