# CL-9: ComfyUI + GGUF startup diagnostics

CL-8 [run 42146255719](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42146255719) passed `import nodes` after installing `trampoline`. CL-9 tests the next layer: whether the free Kaggle GPU host can launch ComfyUI's HTTP server locally, with ComfyUI-GGUF registered as an installed custom node.

- Manually opt in with `YES` in **Kaggle CL-9 ComfyUI GGUF Startup (manual)** to spend free Kaggle GPU quota. No paid GPU service is involved.
- Reuse the CL-8 guarded dependency installs and source checkouts. Do not install eight template media packages or download LTX model weights.
- Symlink the cloned ComfyUI-GGUF tree under ComfyUI's `custom_nodes`. Start `main.py` bound only to `127.0.0.1:8188` for up to 90 seconds, without browser launch.
- Poll `/object_info`, report HTTP readiness, available GGUF-labelled node names and GPU memory before/after. Any failure and a bounded log tail are recorded in `cl9-startup-probe.json`.
- Stop the server and remove checked-out sources before Kaggle outputs are retrieved. Check disk reserve >=19.3 GB and unchanged PyTorch/CUDA.

A green GitHub Actions job **does not imply** that startup or GGUF registration succeeded: inspect `cl9_checks.startup_probe`. Even a successful startup does **not** establish that LTX 2.5 fits VRAM or can generate video. A separate reviewed opt-in is required before any model download or inference.
