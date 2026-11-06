# CL-23: bounded real video self-attention

CL-22 evaluated the first LTX-2.5 video transformer's feed-forward submodule with genuine Q2_K GGUF weights on Kaggle T4. CL-23 isolates its **self-attention submodule** (`transformer_blocks[0].attn1`) and sends **two synthetic tokens** through real GGUF-backed Q/K/V projections, attention and output projection. This tests cross-token attention arithmetic, not the whole transformer block.

**Scope and limitations:** `pe=None` deliberately **skips rotary positional embeddings**. There is no video tokenization, conditioning, full transformer block/model forward, diffusion sampling or generated video. A passing result cannot establish that a complete LTX-2.5 video fits on Kaggle T4, or that attention with real positions works. The experiment uses the existing CL-20 ComfyUI-GGUF model construction and the exact SHA-256 checked 8.46 GB model download.

Safety controls inherited from earlier probes:

- Explicit manual `workflow_dispatch` confirmations, with exact values `YES`, `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD` and `YES_SELF_ATTENTION`.
- Exactly two synthetic tokens; weight dimensions checked before execution, each linear bounded to 80 million logical elements and dimension at most 32,768.
- Dedicated child process limited by a 180-second watchdog and sampled 12 GB RSS ceiling; kill on timeout or overage. PyTorch-reported peak GPU allocation ceiling is 3 GiB across two T4s (this is not a whole-device memory measurement).
- Q2_K GGUF download/verification and removal; result JSON reports only diagnostics. No automatic Kaggle submission on PR or merge.

After review and merge, choose GitHub Actions → **Kaggle CL-23 LTX Self Attention (manual)** → **Run workflow** on `main`, provide the three confirmation strings above, and send back its run link. A failed probe is a meaningful feasibility result; do not increase bounds automatically.
