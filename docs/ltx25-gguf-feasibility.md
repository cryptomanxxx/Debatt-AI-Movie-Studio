# CL-2: LTX 2.5 GGUF storage feasibility

Use **Actions → LTX 2.5 GGUF Feasibility (metadata only) → Run workflow**.
The job runs on a regular GitHub-hosted CPU runner; it **never starts Kaggle**,
downloads model weights, generates video, or spends GPU credits.

The script checks two *distilled transformer* GGUF candidates from
`unsloth/LTX-2.5-GGUF` (Q2_K, UD-Q2_K_XL), a separate projected Gemma 4 Q2_K
text encoder from `elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF`, and official
convolutional video/audio VAEs from `Lightricks/LTX-2.5`. It reads file-size
metadata from pinned Hugging Face repository revisions and fails closed if any
specified file is absent or ambiguous.

The disk reference is the measured Kaggle free disk (20,940,562,432 bytes) in
[hardware probe 42128365700](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42128365700).
A 3 GB reserve is subtracted. `weights_fit_disk_only` counts the four model
files **only**. A positive result does **not** budget ComfyUI, Python packages,
swap, temporary download files, cache, model patches or output artifacts.
It also does not show that two Tesla T4 cards (15.6 GB VRAM each) can execute
the model. Their VRAM cannot simply be added together unless the inference
backend actually supports model parallelism.

This is a *candidate sizing experiment*, **not a validated inference recipe**.
Cross-publisher GGUF combinations require compatibility testing. Some Hugging
Face repositories are gated and require accepting their license terms before
any downloads. The workflow neither accepts terms nor requests gated access.
Any future runtime test must have its own explicitly approved, cost-free,
limited workflow, with a dry-run and termination safeguards.
