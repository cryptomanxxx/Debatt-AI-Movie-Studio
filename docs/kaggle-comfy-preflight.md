# CL-4: ComfyUI and ComfyUI-GGUF source/dependency inventory

The preceding CL-3 Kaggle run [42134862341](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42134862341)
successfully installed `gguf` version 0.19.0. This does **not** mean ComfyUI
or LTX 2.5 can run on the two Kaggle Tesla T4 cards.

**Actions → Kaggle ComfyUI Source Preflight (manual)** requires typing `YES`
to use a small amount of **free Kaggle GPU quota**. This uses the existing
Kaggle secrets and a separate private notebook whose ID matches its title.

The notebook:

- Shallow-clones only the official ComfyUI and `city96/ComfyUI-GGUF` code.
- Records exact repository commit SHAs for reproducibility.
- Reads their `requirements.txt` files and inventories already-installed
  dependencies versus required versions; it **does not install or resolve**
  missing requirements.
- Measures free disk before/after source clones and aborts additional clones
  when free space drops below 19 GB.
- Records GPUs and CUDA/PyTorch versions, writing a JSON report to a short-lived
  GitHub Actions artifact (7-day retention).

The probe intentionally **does not** install ComfyUI's potentially enormous
PyTorch and related dependencies, start a ComfyUI server, download LTX weights,
run inference, or generate video. Internet access inside Kaggle is used solely
for the two public Git repositories; no credentials are placed in the notebook.
This is a source/dependency **inventory**, not a working ComfyUI installation.

The CL-2 weight-only Q2 candidate is ~16.23 GB. With a 3 GB reserve on the
previous measured 20.94 GB Kaggle free disk, only ~1.71 GB was left for *all*
runtime dependencies, caches, downloads, output and swap. Source inventory
helps quantify part of that overhead. GPU memory compatibility and true
free-inference feasibility remain untested.
