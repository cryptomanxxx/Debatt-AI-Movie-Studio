# CL-50 — close-ups and better eyes

## Problem

In the 720p CL-48 clip ([42247737170](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42247737170)), both faces had weak eyes:

- The woman's eyes were dark smudges, with no clear iris or pupil.
- In the last ~0.25 s the man turned his head and his face morphed. His eyes became slits, and the face-region frame difference jumped from ~1–3 to 12.9.

The main cause is scale. Wan 2.2 TI2V compresses 16× spatially, so in a medium shot an eye of ~10–15 px sits inside a single 16×16 latent cell. The model has almost no capacity to draw it. Fast head turns and 30 rather than 50 steps make it worse. The tiled VAE decode is not the cause, because the artifacts follow motion, not tile boundaries.

## Changes

- The **default negative prompt** gains face terms: `deformed eyes, closed eyes, blurry eyes, asymmetric eyes, distorted face, morphing face, extra fingers, deformed hands`. The original terms are kept. This applies to CL-47 and CL-48 runs.
- **Close-up prompts and 50 steps** are tested with the existing workflow inputs, without code changes. A close-up makes the face 3–4× larger, so each eye spans several latent cells.

## Test run

The test is a 720p close-up of one speaker, with 49 frames, **50 steps**, seed 42 and a 1080p upscale. The prompt asks for a subtle head movement rather than a fast turn. At the measured ~22.9 s per step, 50 steps should add about 7–8 minutes compared with 30 steps. That is an estimate. Results are recorded below once the run finishes.
