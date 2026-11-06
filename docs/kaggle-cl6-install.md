# CL-6: bounded installation of directly missing Python packages on Kaggle

CL-5 established a **compressed direct-wheel estimate** of 95,836,305 bytes.
This is not an installed-size estimate or a complete dependency-resolution
result. CL-6 measures actual free disk after installing the 12 direct missing
packages identified in the earlier CL-4 notebook snapshot.

From **Actions → Kaggle CL-6 Direct Dependency Install (manual)**, select
`main` and type **YES** to use a portion of *free Kaggle GPU quota*. This is
a separate **private** notebook, not a paid compute service.

The probe installs the listed packages using pip flags `--no-deps`,
`--no-cache-dir`, `--only-binary=:all:` and short network/process timeouts.
It refuses to proceed to the next package if free disk falls below 19.3 GB,
preserving approximately 16.23 GB for candidate Q2 weights and 3 GB reserve.
It records PyTorch and CUDA versions both before and after and fails if either
changes. It records the installed package step results and checks disk usage.

Important limits:

- It **does not** install transitive dependencies, ComfyUI source code, or
  ComfyUI-GGUF custom nodes. It does **not** launch ComfyUI or generate video.
- It does **not** download LTX weights or accept any Hugging Face license.
- Because the package environment is incomplete, even a fully green probe
  cannot establish that ComfyUI can start or LTX can run on two Tesla T4 cards.
- Kaggle may use some of the free weekly GPU quota during notebook execution.
- The notebook environment is ephemeral; installed packages are not a durable
  shared installation between Kaggle notebooks.

Only a later, separately reviewed experiment may attempt a full ComfyUI
runtime test. Never infer a safe model download from this probe alone.
