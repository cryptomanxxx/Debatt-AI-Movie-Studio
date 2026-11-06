# CL-25: two consecutive video transformer blocks

CL-24B verified that the first video branch runs with real LTX-2.5 Q2_K GGUF weights and correct synthetic prompt-AdaLN inputs. CL-25 passes the resulting two-token video tensor directly from block 0 into block 1, exercising the actual AdaLN-normalized self-attention, cross-attention and feed-forward branches in order. Both blocks use the real model weights, verified dimensions, separate synthetic text and timestep features, and disabled audio.

The experiment has deliberately **only two synthetic video tokens**. `v_pe=None` skips positional rotation, text embeddings and AdaLN inputs are not produced by real model encoders, and the full 48-block model, diffusion sampler and VAE are not invoked. Passing demonstrates numerical compatibility across two adjacent blocks, not that an actual video fits on T4.

Bounded operation: manual `workflow_dispatch` requires `YES`, `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`, `YES_TWO_BLOCKS`; SHA-256 verified 8.46 GB weights and deletion after probe; 80-million-logical-element cap per GGUF linear matrix; 12 GB sampled RSS and 180s child watchdog; 4 GiB PyTorch-reported GPU allocation ceiling. No automatic Kaggle run on PR or merge.

After Codex approval and merge, run **Kaggle CL-25 LTX Two Video Blocks (manual)** on `main`. Share the run link for analysis.
