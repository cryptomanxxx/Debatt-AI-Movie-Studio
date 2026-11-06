# CL-46 — attempt first real playable MP4 (PR 4/4)

This is the **last PR** under the agreed four-PR deadline. It is executable infrastructure for actual GPU rendering, not another synthetic computation experiment. The presence of this script or passing its mocked tests is **not evidence that a movie was generated**.

## First real environment probe (2026-11-06)

An actual read-only probe was executed in the available workspace using `python -m scripts.cl46_real_mp4_attempt --comfy-root ./ComfyUI`. It returned `comfyui_online=false`, `comfy_main_exists=false`, `ltx_video_custom_nodes_exist=false`, all four model weights absent, `devices=[]`, and connection error `URLError`. FFmpeg was present. **No MP4 was produced and no GPU inference occurred.** This is an actual environment limitation, not a successful rendering benchmark. A complete render requires access to a provisioned machine with model weights and a running ComfyUI server.

## Actual execution

1. Choose a real GPU environment that can run LTX-2.5. Install ComfyUI, LTXVideo custom nodes and FFmpeg. Kaggle T4 viability is unverified and the full model stack needs substantial RAM/VRAM and disk. Do not run GPU jobs just to test this setup.
2. Verify/install the four pinned LTX-2.5 assets via CL-44. Use the official `video_ltx2_5_flf2v` template from https://docs.ltx.io/open-source-model/integration-tools/comfy-ui/ . Export it in ComfyUI API format and supply real input images.
3. Use CL-45 to prepare `cl45-results/ready-api-workflow.json`, explicitly overriding the template's video VAE with the Conv VAE.
4. Start ComfyUI on the local machine (`127.0.0.1:8188`). Read-only connection probe:

   `python -m scripts.cl46_real_mp4_attempt --comfy-root /path/to/ComfyUI`

5. **Only when ready and after deciding to spend GPU time**, explicitly run:

```bash
python -m scripts.cl46_real_mp4_attempt \
  --comfy-root /path/to/ComfyUI \
  --workflow cl45-results/ready-api-workflow.json \
  --output cl46-results/first-real-clip.mp4 \
  --submit --confirm QUEUE_REAL_LTX25_VIDEO
```

The script sends the actual graph to the official ComfyUI `/prompt` endpoint, polls `/history/{prompt_id}` for an explicit successful completion, finds an MP4 output, downloads it from `/view`, and uses FFprobe to verify a readable video stream with frames before publishing it. If ComfyUI cannot run the model, it preserves the server's HTTP rejection diagnostics and execution-error messages rather than manufacturing a fake video. A pending prompt may take up to 30 minutes before timing out; a timed-out prompt can still be running in ComfyUI and should be inspected before resubmitting. The report is overwritten at the start of each attempt and records the prompt ID as soon as it is queued, so a previous success cannot mask the current failure.

Only localhost ComfyUI is supported; HTTP redirects are rejected, and the script does not upload credentials or model weights. GitHub Actions validates Python code only and **never** runs the real GPU job on a GitHub runner.

Official ComfyUI queue API: https://github.com/Comfy-Org/docs/blob/main/development/comfyui-server/comms_routes.mdx .
