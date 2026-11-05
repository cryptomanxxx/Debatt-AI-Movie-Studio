# CL-3: bounded Kaggle GGUF environment probe

CL-2 proved only that the smallest candidate **model weights** fit the prior
disk measurement. CL-3 checks whether Kaggle can import the small GGUF format
package and records its actual GPU/disk conditions **without fetching any
model weights**. It does not install or test ComfyUI, ComfyUI-GGUF, Gemma 4,
audio/video decoders, model offloading or inference. These remain blockers to
claiming that LTX 2.5 runs on T4.

Open **Actions → Kaggle GGUF Environment Probe (manual)**, select `main`, and
type `YES` only if you agree to consume a small portion of **free Kaggle GPU
quota**. No paid compute service is configured. The workflow is manual-only,
uses the existing Kaggle secrets and submits a **private**, separate notebook
named `debatt-ai-movie-gguf-probe`. The probe enables Kaggle internet solely
to install the small `gguf` Python dependency from PyPI. The installation
uses `--no-deps`, a 180-second process timeout, and limited network retries;
it never downloads LTX checkpoint weights. GitHub polls for a bounded period,
validates commit/run identity, and uploads a short JSON report and diagnostic
logs. Kaggle notebook execution may still be subject to Kaggle's own quotas,
availability and notebook timeout; a GitHub polling timeout does not guarantee
that Kaggle immediately terminates its worker.

**Do not use the result as evidence of model readiness.** The next decision
is whether a compatible full ComfyUI + ComfyUI-GGUF environment can be built
within the remaining ~1.71 GB Q2 weight-only disk margin and T4 VRAM. This
probe does not authorize a large model download, paid compute, or generation.
