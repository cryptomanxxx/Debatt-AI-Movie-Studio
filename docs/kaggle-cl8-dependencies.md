# CL-8 — Resolve first missing ComfyUI import dependency

CL-7's GPU probe showed `ModuleNotFoundError: No module named 'trampoline'` while importing `nodes`, through `torchsde`. CL-8 adds only a bounded wheel-only, `--no-deps`, `--no-cache-dir` installation of `trampoline` after the same CL-6 direct dependencies. It then retries the bounded ComfyUI import probe and records the next failure, if any.

CL-8 also requests **PyPI metadata only** for the eight missing `comfyui-workflow-templates-*` packages, capturing minimum and maximum wheel sizes for the latest releases. These are compressed download-size observations, *not* installed disk footprints, and no template wheels are installed.

Launch manually from GitHub Actions, entering `YES` to use free Kaggle GPU quota. The output artifact is `movie-studio-cl8-dependencies-<run-id>`, with `cl8-dependency-probe.json` and logs. No paid GPU, full ComfyUI server, model weights, inference or videos. The GitHub Actions job can succeed while the ComfyUI import remains blocked: inspect the JSON `cl8_checks` result, not merely the green CI status.
