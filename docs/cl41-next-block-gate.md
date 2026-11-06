# CL-41 — CPU-only gate before expanding beyond eight video blocks

The successful CL-40 run (GitHub Actions `42215353262`, commit
`3b76f5c667998e008bdcd7a21be5be8cd1e3ec39`) verified three linked
native video evaluations using **blocks 0–7**, with recorded PyTorch peak
allocation 1,521,324,032 bytes and sampled child peak RSS 2,791,383,040
bytes. This is evidence for the already-tested workload, **not** a bound for
the next block or a guarantee that the entire model fits.

CL-41 records a candidate for a *single* additional block (index 8),
but deliberately emits `METADATA_PREFLIGHT_REQUIRED` and explicitly marks
the next block's existence, weight shapes, memory fit and total model depth
as **unverified**. The script is a deterministic CPU-only decision record;
it does not fetch the pinned model manifest or execute an expanded network.

Before any later experiment executes block 8, independently confirm its
existence and inspect all required quantized projection shapes against the
existing 80-million-logical-element guard. Preserve the 4 GiB PyTorch peak,
12 GB sampled child RSS, 180-second child watchdog, 35-minute workflow
timeout and triple manual confirmation. Any later increased-block GPU probe
requires a separately reviewed PR and separate user-triggered Kaggle job.

```bash
python -m unittest tests.test_cl41_next_block_gate -v
python -m scripts.cl41_next_block_gate --output cl41-results/report.json
```

Neither a full model pass nor rendered video is claimed.
