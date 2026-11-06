# CL-49 — 1080p upscale and faster 720p decode

Two follow-ups to the working 720p CL-48 run ([42247737170](https://github.com/cryptomanxxx/Debatt-AI-Movie-Studio/actions/runs/42247737170)).

## 1. Upscale to 1920x1080 (Real-ESRGAN)

Rendering 1080p directly on a T4 would mean about 2.3× more tokens than 720p, which is over the token budget. Instead, the CL-48 workflow now has an `upscale` input (default `1080p`). After the video pipeline finishes and is freed from the GPU, the kernel:

1. Installs `spandrel==0.4.2`, a model loader used by ComfyUI that loads `.pth` weights with a restricted unpickler.
2. Downloads **Real-ESRGAN x2plus** from the official GitHub release and refuses it unless its SHA-256 matches `49fafd45…a266abb`.
3. Upscales every frame 2× in fp16 on GPU 0, then scales and centre-crops each frame to exactly 1920x1080. A landscape 1280x704 frame becomes 2560x1408, is scaled to 1964x1080 and has 22 px cropped from each side. Portrait sources become 1080x1920, and 832x480 sources are scaled up slightly after the 2× pass.
4. Writes `wan22-scene-1080p.mp4` next to the original clip. GitHub verifies it separately with FFprobe for exact frame count and 1920x1080 size, and adds a silent-audio copy for `assemble`.

The original 720p MP4 is always kept and verified too.

**Offline check:** the kernel's `upscale_frames` ran the real Real-ESRGAN x2plus weights through spandrel on a CPU, on frame 24 of the CL-48 720p clip. It produced a 1920x1080 frame in 48 s on the CPU. Compared with plain bicubic scaling, edges are cleaner and compression noise is reduced. Real-ESRGAN cannot invent detail the source lacks, so the small, blurry eyes discussed for the 720p clip remain blurry. Face restoration (for example CodeFormer or GFPGAN) is a possible later step. **GPU speed and memory on Kaggle are not yet measured.**

## 2. Faster tiled decode at 720p

CL-48b decoded 720p with diffusers' default 256 px tiles (stride 192). That is 4 × 7 = 28 overlapping tiles per frame, decoded frame by frame, and took ~9.2 min — longer than the 30 denoising steps. CL-49 uses **512 px tiles with stride 448**, which keeps the same 64 px overlap and is aligned to Wan 2.2's 32 px grid. That is 2 × 3 = 6 tiles per frame, with less recomputed overlap.

Memory reasoning: the full-frame 720p decode needed more than ~12.6 GiB. A 512x512 tile is about 29 % of a 1280x704 frame, so tiled decode should stay far below the limit. **The new decode time is not yet measured.** The report now records `denoise_seconds` and `vae_decode_seconds` separately, so the next run shows the real speed-up. 832x480 and smaller still use the proven full-frame decode.

## First Kaggle runs (2026-11-06)

All runs used the same prompt and seed as CL-48b's 720p run: 1280x704, 49 frames, 30 steps, `upscale: 1080p`.

### Run 42255103054 (`68e489c`): everything ran on Kaggle; the 1080p size check failed

| Phase | CL-48b (256 px tiles) | CL-49 (512 px tiles) |
| --- | --- | --- |
| Prompt encoding + 30 denoising steps | ~723 s | 764 s |
| VAE decode + export | ~553 s | **411 s** |
| Real-ESRGAN x2 + fit to 1080p, 49 frames | – | **73 s** (about 1.5 s per frame, 2.0 GB peak VRAM) |
| Total Kaggle script | 1,564 s | 1,549 s |

The larger tiles made decode **about 25 % faster**, not the several-fold gain hoped for. Most of the decode cost appears to be the frame-by-frame causal decoder rather than tile overlap. The upscale is cheap. The 1080p file came out **1920x1088**, imageio's `macro_block_size=16` padding that Codex also flagged, and the FFprobe check rightly rejected it. `cc8dfd3` exports with `macro_block_size=1`.

### Run 42259470579 (`cc8dfd3`): killed by the Linux OOM killer

The Kaggle log ends in `Killed` at t≈1,430 s, right at the end of the VAE decode (decode began at 1,022 s) and **before the upscale started**. The process was killed by the operating system for running out of system RAM, so no JSON report was written. The 1080p export change was never reached, so it is not the cause. The previous run survived the same phase, so system RAM was simply marginal at the end of decode. With model CPU offload, the idle UMT5 text encoder (~11 GB in bf16) and the transformer (~10 GB) both sit in the 33.7 GB of CPU RAM, and the decoded frames and their copies are added on top.

**Fix:** prompts are now encoded **first**, with only the tokenizer and UMT5 loaded, using diffusers' own `WanPipeline.encode_prompt` (`max_sequence_length=512`, the `__call__` default). The encoder is then deleted, and the video pipeline is loaded with `text_encoder=None, tokenizer=None` and given `prompt_embeds` / `negative_prompt_embeds`. This removes about 11 GB from system RAM for the whole denoise, decode and upscale phase.

**Equivalence check:** on a tiny, randomly initialised Wan 2.2-shaped pipeline with the same seed, the old path (prompt strings with the encoder inside the pipeline) and the new path produced **bit-identical frames** (max abs diff 0.0).
