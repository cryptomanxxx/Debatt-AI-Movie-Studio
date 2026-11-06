# CL-38 — bounded LTX sigma scheduler reference (CPU only)

CL-37 used a synthetic fixed-step Euler schedule for two bounded LTX model
evaluations. CL-38 isolates the **sigma selection formula** used by ComfyUI's
`LTXVScheduler.execute`, independently of model inference or Kaggle. It does
**not** execute ComfyUI's scheduler: it reproduces its published mathematics in
plain Python double-precision floats. Minor rounding differences versus
ComfyUI's `torch.float32` implementation are expected.

Primary implementation reference:
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_lt.py
(`LTXVScheduler.execute`).

For a requested number of latent tokens `n`, the scheduler linearly interpolates
the shift `s` between `(1024, base_shift=0.95)` and
`(4096, max_shift=2.05)`. For `steps=4`, the initial sigma grid is
`[1, 0.75, 0.5, 0.25, 0]`. Nonzero sigmas are transformed as
`exp(s) / (exp(s) + (1/sigma - 1))`. When stretching is enabled, it
rescales nonzero entries with
`factor = (1 - last_nonzero_sigma) / (1 - terminal)` and
`sigma_stretched = 1 - (1 - sigma) / factor`, making the penultimate
sigma `terminal=0.1`; the trailing zero remains zero.

The report compares a four-token synthetic grid to a 4096-token reference,
and records the 4096-token unstretched curve. **Four tokens fall below the
1024–4096 token calibration anchors**, so this is an extrapolation to inspect
the formula, **not** evidence that the four-token schedule represents
production video latents. It is also not proof of native LTX video generation.

## Safe execution

```bash
python -m unittest tests.test_cl38_ltx_sigma_schedule -v
python -m scripts.cl38_ltx_sigma_schedule --output cl38-sigma-results/report.json
```

The manual GitHub Actions workflow `CL-38 LTX Sigma Schedule (manual CPU)`
executes the standard unit test suite and writes a seven-day JSON artifact.
No Kaggle secrets, model downloads, GPU, model forward pass, denoising loop,
VAE decoder, image, or video are involved. This step only verifies sigma
schedule properties and produces a reproducible reference for a **separately
authorized** future bounded model-inference experiment.
