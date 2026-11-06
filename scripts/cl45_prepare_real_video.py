"""CL-45: validate and prepare a real exported ComfyUI API workflow.

No GPU work, downloads or queue submissions. CL-46 uses the prepared JSON.
"""
import argparse
import json
import re
from pathlib import Path

from scripts.cl43_video_runtime_preflight import ASSETS, preflight

ROLE_INPUTS = {
    "transformer": ("model_node", "unet_name"),
    "text_encoder": ("text_node", "clip_name"),
    "video_vae": ("video_vae_node", "vae_name"),
    "audio_vae": ("audio_vae_node", "vae_name"),
}


def prepare(comfy_root, workflow_path, start_image, end_image, prompt,
            *, model_node, text_node, video_vae_node, audio_vae_node,
            start_node, end_node, prompt_node):
    root = Path(comfy_root).expanduser().resolve()
    readiness = preflight(root)
    if not readiness["ready_for_manual_workflow_import"]:
        raise ValueError("ComfyUI runtime/assets/ffmpeg/disk preflight is not ready")
    data = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data or not all(
        isinstance(k, str) and re.fullmatch(r"[0-9]+(?::[0-9]+)*", k) and isinstance(v, dict)
        and isinstance(v.get("class_type"), str) and isinstance(v.get("inputs"), dict)
        for k, v in data.items()
    ):
        raise ValueError("Export the official FLF2V workflow in ComfyUI API format")

    node_ids = {"model_node": model_node, "text_node": text_node,
                "video_vae_node": video_vae_node, "audio_vae_node": audio_vae_node,
                "start_node": start_node, "end_node": end_node,
                "prompt_node": prompt_node}
    if len(set(node_ids.values())) != len(node_ids):
        raise ValueError("Each selected API workflow node must be distinct")
    for role, (arg, input_name) in ROLE_INPUTS.items():
        node = data.get(node_ids[arg])
        if not node or input_name not in node["inputs"]:
            raise ValueError(f"Missing explicit {role} selector on node {node_ids[arg]}")
        filename = next(name for r, _, name in ASSETS if r == role)
        # Refuse to silently use the default BF16 DiffVAE template selector.
        node["inputs"][input_name] = filename

    for label, node_id, filename in (("start", start_node, start_image),
                                     ("end", end_node, end_image)):
        node = data.get(node_id)
        if not node or node.get("class_type") != "LoadImage" or "image" not in node["inputs"]:
            raise ValueError(f"{label} image must use a LoadImage node")
        # ComfyUI uses filenames in its input directory, not absolute paths.
        if Path(filename).name != filename or filename in ("", ".", ".."):
            raise ValueError("Image input must be a filename without directories")
        if not (root / "input" / filename).is_file():
            raise ValueError(f"Missing ComfyUI/input/{filename}")
        node["inputs"]["image"] = filename

    node = data.get(prompt_node)
    if not node or "text" not in node["inputs"] or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt node must expose a text input and prompt must be nonempty")
    node["inputs"]["text"] = prompt
    video_outputs = []
    for output in data.values():
        kind = output["class_type"]
        settings = output["inputs"]
        if kind == "SaveVideo" and isinstance(settings.get("filename_prefix"), str) \
                and settings["filename_prefix"].strip():
            video_outputs.append(kind)
        elif kind == "VHS_VideoCombine" and isinstance(settings.get("filename_prefix"), str) \
                and settings["filename_prefix"].strip() \
                and isinstance(settings.get("format"), str) \
                and "mp4" in settings["format"].lower():
            video_outputs.append(kind)
    if not video_outputs:
        raise ValueError("API graph must contain a supported video-saving output node")

    return {"graph": data,
            "report": {"source": "official_FLF2V_exported_API_graph",
                       "nodes": len(data), "video_output_nodes": video_outputs,
                       "vae_override": next(
                           name for role, _, name in ASSETS if role == "video_vae"),
                       "start_image": start_image, "end_image": end_image,
                       "runtime_preflight_passed": True,
                       "runtime_model_hashes_verified": False,
                       "comfyui_graph_validated_by_server": False,
                       "video_generated": False, "gpu_used": False,
                       "next_action": "CL-46 explicit queue submission and MP4 retrieval"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--start-image", required=True)
    parser.add_argument("--end-image", required=True)
    parser.add_argument("--prompt", required=True)
    for name in ("model-node", "text-node", "video-vae-node", "audio-vae-node",
                 "start-node", "end-node", "prompt-node"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, default=Path("cl45-results/ready-api-workflow.json"))
    args = parser.parse_args(argv)
    kwargs = {k: getattr(args, k) for k in ("model_node", "text_node", "video_vae_node",
                                           "audio_vae_node", "start_node", "end_node", "prompt_node")}
    prepared = prepare(args.comfy_root, args.workflow, args.start_image,
                       args.end_image, args.prompt, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prepared["graph"], indent=2) + "\n", encoding="utf-8")
    report = args.output.with_suffix(".report.json")
    report.write_text(json.dumps(prepared["report"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"workflow_file": str(args.output), "report_file": str(report),
                      **prepared["report"]}, indent=2))


if __name__ == "__main__":
    main()
