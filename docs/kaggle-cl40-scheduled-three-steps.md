# CL-40 — three linked native-AdaLN steps with bounded LTX-formula sigmas

CL-39 succeeded with two native model evaluations (8 video blocks per pass),
two Euler updates and the first three sigma values from a CPU reproduction
of ComfyUI's `LTXVScheduler.execute`. CL-40 adds **one more native model
evaluation and Euler update**, using the fourth sigma from the same 20-step
formula for a four-token synthetic latent grid.

The third pass consumes the second pass's updated latent, then patchifies,
applies native timestep/AdaLN and the same eight loaded video transformer
blocks, and computes another native conditioned output and Euler update.
The JSON evidence separately records three evaluations, three updates,
third-step numerical residual, output shape, finiteness and GPU peak.

This remains a deliberately small synthetic probe. Four tokens extrapolate
below the scheduler's 1024–4096 reference anchors; native ComfyUI scheduler
execution, authentic text conditioning, full model, VAE, frames and video
generation are **not claimed**.

The same limits remain: pinned Q2_K GGUF, 4 GiB PyTorch allocation cap,
12 GB sampled child RSS bound, 180-second watchdog, 35-minute workflow
timeout, cleanup and seven-day report retention. The Kaggle workflow must
be started **manually**, with explicit confirmations for GPU use, GGUF weight
download and this specific third-step experiment. It never runs on push or PR.

Offline verification:

```bash
python -m unittest tests.test_kaggle_cl40_scheduled_three_steps -v
python -m unittest discover -s tests -q
```
