# CL-42 — real pinned Q2_K GGUF block-8 metadata, without inference

CL-41 identified block 8 as the first untested candidate after the proven
video blocks 0–7. CL-42 inspects the **read-only GGUF tensor index**, matching
tensor names containing `transformer_blocks.8.`. It records shapes, element
counts, quantization kinds and whether each tensor is within the existing
80-million-logical-element projection guard. Other observed transformer block
indices are reported without asserting the checkpoint's full architectural
depth or whether all required modules were present.

This is **not** a video-block forward pass or proof of memory fitness. An
empty matched list indicates that the expected naming could not be confirmed,
not necessarily that the underlying architecture lacks block 8. The GGUF
reader uses read-only memory mapping; this code never accesses tensor data,
dequantizes weights, constructs ComfyUI's model, or executes GPU operations.
It reuses CL-14's pinned 8.46 GB GGUF download, verified SHA, cleanup and
bounded environment report, so a Kaggle session and weight download are
still necessary.

The workflow must be started **manually** with independent confirmations for
Kaggle use, large GGUF download and metadata-only inspection. It has a
35-minute timeout and 7-day diagnostics retention. No automatic job on PR or
merge. A later single-block-8 computation would require another separately
reviewed PR and manual GPU authorization.

```bash
python -m unittest tests.test_kaggle_cl42_block8_metadata -v
python -m unittest discover -s tests -q
```
