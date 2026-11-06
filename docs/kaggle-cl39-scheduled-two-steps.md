# CL-39 — two native model updates with the LTX sigma formula

CL-37 verified two bounded, linked native-AdaLN eight-video-block model
evaluations with synthetic Euler sigmas `1, 0.96875, 0.9375`. CL-38 separately
reproduced the ComfyUI `LTXVScheduler.execute` mathematical formula on CPU.
CL-39 joins them: it computes a stretched **20-step / four-token** schedule
on the submission host, embeds its first three values into CL-37's proven
Kaggle child and performs exactly two native-conditioned Euler updates.

Expected first sigmas: approximately `1.0, 0.9719346224, 0.9425690441`.
This is a formula reproduction, **not a direct execution of ComfyUI's
LTXVScheduler**, and Python float values may differ slightly from `torch.float32`.
The four-token test **extrapolates below** the scheduler's 1024–4096 token
shift anchors; this does not validate the schedule for real video latents.

The child retains CL-37's constraints: eight native video transformer blocks,
real pinned Q2_K GGUF, bounded projection probes, two linked model evaluations,
4 GiB PyTorch peak allocation bound, sampled child RSS cap of 12 GB,
180-second watchdog, 35-minute workflow timeout and cleanup. No text encoder,
VAE, full model forward, rendered frames, or video generation. The report
explicitly records `official_scheduler_executed=false`,
`scheduler_formula_reproduction=true` and `video_generated=false`.

The workflow `Kaggle CL-39 Scheduled Two Steps (manual)` is **never triggered
automatically**. The user must separately approve Kaggle GPU consumption,
the 8.46 GB weight download, and this specific experiment through three
manual `workflow_dispatch` confirmations. Results are retained seven days.

Offline preflight (no Kaggle and no GPU):

```bash
python -m unittest tests.test_kaggle_cl39_scheduled_two_steps -v
python -m unittest discover -s tests -q
```
