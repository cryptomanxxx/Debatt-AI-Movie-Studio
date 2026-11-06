# CL-10: No-weight GGUF memory feasibility scenarios

CL-9 [run 42147482064](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42147482064) successfully started ComfyUI and registered six GGUF nodes. The empty server showed free VRAM of **15,419,113,472 bytes on GPU 0** and **15,526,068,224 bytes on GPU 1**. This is not evidence that any LTX checkpoint can be loaded.

Use **Actions → LTX 2.5 CL-10 GGUF Memory Budget (metadata only) → Run workflow**. It runs on a GitHub-hosted CPU, does not launch Kaggle, and fetches only Hugging Face repository metadata (file paths, sizes, pinned revisions and gating).

For the two existing GGUF candidate packs (`distilled_q2` and `distilled_ud_q2`), CL-10 calculates:

- Complete checkpoint file sizes and remaining *weights-only* disk space after the 3 GB reserve.
- A one-GPU file-size proxy compared with each measured free VRAM budget.
- An **illustrative, unverified** two-GPU component split: transformer+VAEs on GPU 0, text encoder on GPU 1; it reports the apparent margins but **does not assert that ComfyUI supports the split**.
- Explicit `unknown` runtime VRAM feasibility and CPU offload feasibility.

Checkpoint file sizes are not the same as GPU memory allocations. Inference adds activations, caches, dequantization and overhead, and separate T4 cards are **not** one 32 GB pool. Disk calculations exclude temporary downloads, dependencies, runtime caches and outputs. Cross-publisher compatibility, license access, model parallelism and inference speed are untested. This workflow never downloads weights, consumes Kaggle quota, or generates a video. Any actual model loading requires its own separate approval.
