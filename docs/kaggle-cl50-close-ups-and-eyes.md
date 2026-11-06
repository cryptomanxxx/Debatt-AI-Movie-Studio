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

The test is a 720p close-up of one speaker, with 49 frames, **50 steps**, seed 42 and a 1080p upscale. The prompt asks for a subtle head movement rather than a fast turn. 
## Result: run 42270674814 — eyes fixed in close-up

[Run 42270674814](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42270674814) passed end to end. The prompt was *"Close-up shot of a female politician speaking passionately in a bright TV studio, face filling the frame, sharp detailed eyes looking slightly off camera, natural blinking, subtle head movement, soft studio lighting, shallow depth of field"*. Settings were 1280x704, 49 frames, **50 steps**, seed 42, the new negative prompt and the 1080p upscale. FFprobe verified the 1280x704 and 1920x1080 MP4s.

| Phase | Time |
| --- | --- |
| 50 denoising steps | 1,115 s (~22.3 s per step) |
| VAE decode (512 px tiles) | 382 s |
| Real-ESRGAN x2 + fit to 1080p | 72 s |
| **Total Kaggle script** | **1,939 s (~32 min)**, about 7 minutes more than 30 steps |

Peak VRAM was 12.4 GB.

**Visual check:** the eyes are now the strongest part of the image. Both have clear blue irises, round pupils, catchlights, lashes, eyeliner and lid creases. They stay consistent in every sampled frame (9 frames, every 6th, inspected on the 1080p output). There is no smearing, no slits and no identity drift. Skin texture, wrinkles, an earring and hair strands are sharp at 1080p. Expression changes clearly between frames: open mouth, frown, closed mouth.

For comparison, the medium-shot clip ([42247737170](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42247737170)) had eyes that were dark smudges or slits, and a face that morphed during a head turn.

**Remaining limits:**
- No blink was seen in the sampled frames, although the prompt asked for natural blinking.
- The near-side eye's lashes are slightly heavy and irregular.
- The face-region frame difference ranges from 1.5 to 14.5, with a median of 4.9. That is higher than in the medium shot only because the face covers far more pixels. The two highest values (frames 1 and 25) are expression changes, not morphing.
- The close-up wins on detail but shows one person. Two-person debate shots still need wider framing, where eyes stay small. Cutting between single close-ups (shot / reverse shot) is the practical way to build a debate scene.
