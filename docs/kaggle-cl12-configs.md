# CL-12: Four no-model ComfyUI memory configurations

CL-11 confirmed that the upstream ComfyUI source defines `--lowvram`, `--cpu-vae`, dynamic VRAM and memory offload features, and that GGUF loaders reference offload devices. These source indicators do **not** prove LTX 2.5 Q2 compatibility.

CL-12 measures actual **empty-server** startup under four configurations on Kaggle's free T4 GPU quota:

1. Default dynamic-VRAM behavior (no overrides).
2. `--disable-dynamic-vram --lowvram` (avoids ComfyUI's documented no-op interaction).
3. `--cpu-vae`.
4. `--disable-smart-memory`.

Each variant is started locally on `127.0.0.1` on a distinct port, queried at `/object_info`, and stopped before the next variant. The report records HTTP readiness, GGUF node registration, the server process's RSS, GPU free memory and system available RAM before/after. The probe enforces a 75-second startup timeout per variant, 19.3 GB minimum disk reserve and 3 GB available RAM guard. The GitHub Actions workflow is manually dispatched with `YES` and uses free Kaggle GPU quota.

The report artifact is `movie-studio-cl12-configs-<run-id>`, including `cl12-config-probe.json`. A green GitHub job indicates only valid diagnostics, not that every configuration starts successfully. No LTX weights, video generation or CPU-offload effectiveness with an actual model are tested. Startup-only memory deltas cannot predict weight loading or inference peaks. No paid GPU is used.
