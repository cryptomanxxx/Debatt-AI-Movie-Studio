"""CL-30: genuine model latent-video patchify/projection with tiny synthetic latents."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl28_video_rope import KERNEL_SOURCE as CL28_SOURCE, _ROPE_BLOCK as CL28_BLOCK

SLUG = "debatt-ai-movie-cl30-latent-patchify"
TITLE = "Debatt AI Movie CL30 Latent Patchify"
REPORT_NAME = "cl30-latent-patchify.json"
RESULTS = Path("kaggle-cl30-results")
SUBMISSION = Path("kaggle-cl30-submission")

_PATCHIFY_BLOCK = CL28_BLOCK.replace(
    '    vx = torch.linspace(-1, 1, 4 * v_width, device="cuda:0", dtype=torch.float16).reshape(1, 4, v_width)',
    '    # Synthetic *latent* input, not a video, encoded pixels or VAE output.\n'
    '    from comfy.ldm.lightricks.model import LTXVModel\n'
    '    channels = int(root.in_channels)\n'
    '    if not (2 <= channels <= 512):\n'
    '        raise RuntimeError("Unexpected LTX latent input channel count")\n'
    '    proj = root.patchify_proj\n'
    '    w = getattr(proj, "weight", None)\n'
    '    if (not isinstance(proj, GGMLOps.Linear) or w is None\n'
    '            or not hasattr(w, "tensor_type") or len(w.shape) != 2\n'
    '            or tuple(w.shape) != (v_width, channels)\n'
    '            or w.shape[0] * w.shape[1] > 80_000_000):\n'
    '        raise RuntimeError("Unsafe or non-GGUF patchify projection")\n'
    '    latent = torch.linspace(-1, 1, channels * 4, device="cuda:0",\n'
    '                            dtype=torch.float16).reshape(1, channels, 1, 2, 2)\n'
    '    # Explicit video-only path avoids AV audio patchifier and returns genuine\n'
    '    # patchified tokens, projected embeddings, and video pixel positions.\n'
    '    vx, coords, _ = LTXVModel._process_input(root, latent, None, None)\n'
    '    if tuple(vx.shape) != (1, 4, v_width) or not torch.isfinite(vx).all().item():\n'
    '        raise RuntimeError("Real latent-video patchify/projection invalid")\n'
    '    if tuple(coords.shape) != (1, 3, 4, 2):\n'
    '        raise RuntimeError("Real latent coordinates invalid")'
).replace(
    '    # Four distinct synthetic (frame, y, x) coordinates; 25 fps for video time.\n'
    '    coords = torch.tensor([[[[0, 1], [0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [1, 2], [1, 2]], [[0, 1], [1, 2], [0, 1], [1, 2]]]],\n'
    '                          device="cuda:0", dtype=torch.float32)\n',
    '    # Coordinates come from LTXVModel._process_input, not a hardcoded grid.\n'
).replace(
    '"rotary_positions_applied": True, "coordinate_source": "synthetic_grid",',
    '"rotary_positions_applied": True, "coordinate_source": "latent_patchifier",\n'
    '              "real_patchify_projection": True, "synthetic_latent": True,\n'
    '              "vae_encoded": False, "real_video_input": False,'
)
assert _PATCHIFY_BLOCK != CL28_BLOCK
assert 'LTXVModel._process_input(root, latent, None, None)' in _PATCHIFY_BLOCK
assert 'coordinate_source": "latent_patchifier"' in _PATCHIFY_BLOCK
assert 'coords = torch.tensor(' not in _PATCHIFY_BLOCK


def make_kernel():
    source = CL28_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL28_BLOCK) == 1
    child = child.replace(CL28_BLOCK, _PATCHIFY_BLOCK)
    source = source.replace(lines[0], marker + repr(child))
    for old, new in (
        ('"comfy_gguf_video_rope"', '"comfy_gguf_latent_patchify"'),
        ('"cl28_checks"', '"cl30_checks"'),
        ('"cl28_disk_free_after_bytes"', '"cl30_disk_free_after_bytes"'),
        ('"cl28_torch_after"', '"cl30_torch_after"'),
        ('"cl28_cuda_after"', '"cl30_cuda_after"'),
        ('cl28-sources', 'cl30-sources'),
        ('cl28-video-rope.json', REPORT_NAME),
        ('cl28_video_rope_complete', 'cl30_latent_patchify_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl30_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_latent_patchify", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl30_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl30_" + suffix)
    validate_cl14(baseline, sha, run_id)
    block = model.get("video_block", {})
    counts = block.get("layer_counts", {})
    if (model.get("ok") is not True or model.get("comfy_model_constructed") is not True
            or model.get("full_model_forward") is not False or model.get("video_generated") is not False
            or block.get("block_index") != 0 or block.get("video_branch_complete") is not True
            or block.get("submodules") != ["attn1", "attn2", "ff"]
            or not all(2 <= counts.get(n, 0) <= 10 for n in ("attn1", "attn2", "ff"))
            or block.get("tokens") != 4 or block.get("finite") is not True
            or block.get("input_shape") != block.get("output_shape")
            or block.get("input_shape") != [1, 4, 4096]
            or block.get("synthetic_input") is not True
            or block.get("synthetic_text_context") is not True
            or block.get("synthetic_timestep") is not True
            or block.get("synthetic_prompt_timestep") is not True
            or block.get("cross_attention_adaln") is not True
            or block.get("rotary_positions_applied") is not True
            or block.get("coordinate_source") != "latent_patchifier"
            or block.get("real_patchify_projection") is not True
            or block.get("synthetic_latent") is not True
            or block.get("vae_encoded") is not False
            or block.get("real_video_input") is not False
            or block.get("frame_rate") != 25
            or block.get("audio_branch_run") is not False
            or block.get("full_model_forward") is not False
            or len(model.get("gpu_peak_allocated_bytes", [])) != 2
            or any(v > 4 * 1024**3 for v in model["gpu_peak_allocated_bytes"])
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-30 latent patchify or video block incomplete/unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
