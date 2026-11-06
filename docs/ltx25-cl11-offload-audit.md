# CL-11: CPU-offload and GGUF source audit

CL-9 proved that ComfyUI's HTTP server and six GGUF loader nodes start on free Kaggle T4 GPUs. CL-10 showed the candidate Q2 pack's file bytes alone exceed either single GPU's measured free VRAM. This **does not** establish that CPU offload or multi-GPU model splitting is functional.

CL-11 uses **GitHub Actions CPU only**, not Kaggle. It shallow-clones the upstream ComfyUI and ComfyUI-GGUF repositories and records source revisions, source-code locations and snippets for the relevant CLI flags and memory-management methods. It does not execute ComfyUI, load weights, or infer actual LTX compatibility.

Run **Actions → LTX 2.5 CL-11 Offload Source Audit (CPU only)** manually. Download `movie-studio-cl11-offload-<run-id>` and inspect `offload-source-audit.json`.

Interpretation rules:

- Presence of `--lowvram`, `--cpu`, `--reserve-vram`, `--cpu-vae`, offload methods, or GGUF loader names is **source evidence only**.
- A missing pattern may mean code moved; it does not prove a feature is unsupported.
- No verified 2×T4 memory pooling, LTX 2.5 GGUF loader compatibility, cross-publisher component combination, GPU residency, CPU/RAM capacity or generation speed follows from this audit.
- Any future use of real LTX 2.5 model weights requires separately approved downloads and strict free-Kaggle resource checks.
