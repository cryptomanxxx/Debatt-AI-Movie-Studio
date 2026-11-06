# CL-45 — first real LTX-2.5 video workflow preparation (PR 3/4)

This is the last setup PR before CL-46's **real MP4 generation attempt**. CL-45 does not queue a GPU job. It prepares an actual ComfyUI API-format FLF2V graph from an official template exported by the user, and fails explicitly if prerequisites are missing. No new synthetic LTX benchmarks are introduced.

1. Install the real ComfyUI runtime with the official LTX nodes. Use CL-44 to download and SHA-verify the official LTX-2.5 INT8 ConvRot transformer and Gemma 4 text encoder, BF16 Conv video VAE, and BF16 audio VAE. Use substantial disk space and appropriate GPU hardware; compatibility with Kaggle's T4 is **not** established.
2. In ComfyUI, select the `video_ltx2_5_flf2v` official First/Last Frame template. Export **API-format** JSON (not the regular UI graph). This requires a real installed workflow; CL-45 does not claim to embed or reconstruct its topology.
3. Copy two real image files to `ComfyUI/input/start.png` and `ComfyUI/input/end.png` and inspect the exported JSON for node IDs. IDs such as `251:229` from flattened ComfyUI subgraphs are supported. Identify the four model loader nodes and their input selectors (`unet_name`, `clip_name`, `vae_name`, `vae_name`), two `LoadImage` nodes, and the prompt `text` node. The API graph must contain `SaveVideo` with a nonempty `filename_prefix` or `VHS_VideoCombine` with a nonempty `filename_prefix` and an MP4 `format`; saving a still image is not a video result.
4. From this repo run:

```bash
python -m scripts.cl45_prepare_real_video \
  --comfy-root /path/to/ComfyUI \
  --workflow /path/to/official-flf2v-api.json \
  --start-image start.png --end-image end.png \
  --prompt "A short gentle camera move from the first frame to the last" \
  --model-node MODEL_ID --text-node TEXT_ID \
  --video-vae-node VIDEO_VAE_ID --audio-vae-node AUDIO_VAE_ID \
  --start-node START_ID --end-node END_ID --prompt-node PROMPT_ID
```

CL-45 confirms the specified files, selector inputs and a save node, then writes `cl45-results/ready-api-workflow.json`. It forces the four model filenames, especially the Conv video VAE override. This is **not** full ComfyUI graph validation; only ComfyUI itself can validate the graph and actual runtime dependencies. CL-46 will submit the prepared graph to ComfyUI's `/prompt` API, monitor `/history/{prompt_id}`, and retrieve a playable MP4 or document the actual error. The server is not contacted in CL-45.

Sources: https://docs.ltx.io/open-source-model/integration-tools/comfy-ui/ ; https://docs.ltx.io/open-source-model/reference/workflow-asset-compatibility ; https://github.com/Comfy-Org/docs/blob/main/development/comfyui-server/comms_routes.mdx .
