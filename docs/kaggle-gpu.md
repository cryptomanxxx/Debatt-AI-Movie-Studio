# Kaggle GPU probe (GitHub Actions only)

This workflow follows the existing Debatt-AI Model Lab pattern: GitHub Actions controls Kaggle through the Kaggle CLI, with no Kaggle website interaction required.

## One-time setup (Movie Studio repository)

Add these two **repository secrets** under Settings → Secrets and variables → Actions:

- `KAGGLE_USERNAME` — the username of the **existing** Kaggle account already used by Debatt-AI Model Lab.
- `KAGGLE_API_TOKEN` — a Kaggle API token for the same account.

Secrets do not automatically transfer from Model Lab to Movie Studio. Never paste API tokens into workflows, commits, public issues or logs.

## Start from GitHub, not Kaggle

1. After this PR is merged, open **Actions → Kaggle GPU Feasibility (manual)**.
2. Select **Run workflow** and enter `YES` for `confirm_gpu_use`.
3. The workflow creates and submits a **private** Kaggle T4 ×2 script, polls for completion and downloads the hardware report.
4. Read the GitHub Actions summary or download the `movie-studio-kaggle-probe-...` artifact (7-day retention).

The first test measures the exact GPU devices, VRAM per device, CUDA version, system memory and free disk space. It **does not download LTX 2.5, generate video, or run a model benchmark**.

## Limits and safety

- **Manual trigger only:** there is no schedule and no automatic GPU consumption.
- **Free GPU quota is shared** with Model Lab because both repositories use the same Kaggle account.
- The Kaggle job has a 900-second requested timeout; GitHub polling is also bounded.
- A new, private Kaggle kernel named `debatt-ai-movie-gpu-probe` is created or updated in the user's existing Kaggle account.
- The script refuses stale output by checking the source commit SHA and unique GitHub run ID.
- Results are temporary GitHub Actions artifacts, not permanent media storage.
- A future LTX generation step requires a separately reviewed PR after hardware feasibility is known.
