# CL-22 — Actual LTX-2.5 transformer feed-forward sub-block

CL-21 successfully executed `GGMLOps.Linear.forward` on a real quantized 32×2048 LTX-2.5 layer using a tiny synthetic input on Kaggle T4. CL-22 moves from one linear layer to the **complete feed-forward submodule (`transformer_blocks[0].ff`)** of the first video/audio transformer block, exercising multiple sequential layers through ComfyUI-GGUF.

The full LTX-2.5 attention block has several additional inputs for text/audio/video context, positional embeddings and time modulation. **CL-22 does not execute attention, a complete transformer block, diffusion-model forward, sampling, or video generation.** It only tests the genuine loaded feed-forward submodule, with one synthetic hidden-state token `[1,1,hidden_width]`. This isolates a meaningful intermediate integration step.

Safety: the two-to-eight feed-forward linears must all have logical dimensions no larger than 32768 and at most 80 million logical matrix elements each. The child process is watched every 50 ms and killed when observed RSS exceeds 12 GB or after 180 seconds; reported peak PyTorch GPU allocations must not exceed 2 GiB per GPU. As in previous experiments, sampling memory is not a cgroup hard limit and cannot prevent every extremely short spike/OOM. The 8.46 GB GGUF checkpoint is SHA-256-verified, staged outside notebook output, and deleted in cleanup. No automatic runs.

After Codex approval and merge, manually choose **Kaggle CL-22 LTX Transformer Feed Forward (manual)** on `main`, with:

- `confirm_gpu_use`: `YES`
- `confirm_weight_download`: `YES_I_APPROVE_Q2_WEIGHT_DOWNLOAD`
- `confirm_transformer_ff`: `YES_TRANSFORMER_FF`

Kaggle T4 quota is consumed and the entire model file is downloaded again. Please inspect any failure before advancing to attention.
