# CL-7 — Kaggle ComfyUI import diagnostics

This is a manually dispatched, opt-in diagnostic notebook using **free Kaggle GPU quota**. It is not LTX inference or a complete ComfyUI setup.

1. Re-run CL-6's 12 direct binary-only, `--no-deps`, `--no-cache-dir` wheel installs. Abort if its disk / PyTorch / CUDA guards fail.
2. Run `pip check` and record unresolved transitive dependencies as diagnostics (not a reason to hide the report).
3. Shallow-clone ComfyUI and ComfyUI-GGUF source into Kaggle working storage; monitor each checkout during cloning, enforce a 90-second/150 MB ceiling and 19.3 GB minimum free-space reserve, and remove the generated checkout before retrieving Kaggle outputs.
4. Try `import nodes` in a subprocess with a 45-second timeout. Missing dependencies, import errors and captured partial output on timeout are recorded in the JSON report for later investigation.
5. Recheck disk guard (>=19.3 GB free) and unchanged PyTorch/CUDA. The notebook does not start the ComfyUI HTTP service, download model weights, load LTX or generate video.

Run GitHub Actions → **Kaggle CL-7 ComfyUI Import Diagnostics (manual)** → `Run workflow` → enter `YES` to authorize spending your free Kaggle GPU quota. Output artifact `movie-studio-cl7-import-<run-id>` contains `cl7-import-probe.json` and diagnostic logs.

The source checkouts use public GitHub network access from Kaggle. Failures are observations, not green evidence that ComfyUI can start. We need a separate, reviewed follow-up before installing transitive packages or attempting model weights.
