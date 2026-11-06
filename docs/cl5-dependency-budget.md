# CL-5: approximate Python wheel download budget (metadata only)

**GitHub Actions → CL-5 LTX Dependency Size Estimate (metadata only) → Run workflow**.
This job is manually triggered on a normal GitHub CPU runner. It uses the
official PyPI JSON API for the 12 missing or version-incompatible packages
reported by successful [CL-4 run 42137390963](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42137390963).

CL-4 identified **11 ComfyUI** requirements and **1 GGUF plug-in** requirement
not installed in that specific Kaggle notebook. GGUF 0.19.0 was installed in
a *different* earlier notebook; notebook environments should not be assumed to
share installed packages.

For each package, CL-5 selects a non-yanked wheel for a compatible version
using the **GitHub Ubuntu Python 3.11** runtime as a proxy. It reports the
wheel's compressed download size, including packages that are missing or have
no compatible wheel. It compares this **incomplete direct-wheel estimate** to
the approximate disk margin remaining after Q2 weights and a 3 GB reserve.

**This is not a pip resolution, not Kaggle compatibility validation, and not
an installed-size estimate.** It excludes transitive dependencies, file
expansion, caches, model patches, ComfyUI extensions, swap and outputs. The
Q2 model size is rounded from the CL-2 report. Even a positive fit result
must not be used to authorize weight downloads. A missing or unresolved wheel
causes the result to remain *unknown*, not "fits".

The workflow does not contact Kaggle, start any GPUs, download wheels or
model weights, install ComfyUI, or generate video. It requires no API tokens.
