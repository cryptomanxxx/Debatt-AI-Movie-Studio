"""CL-31: eight LTX video blocks fed by real latent patchify/projection and RoPE."""
import ast
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl29_eight_blocks_rope import (
    KERNEL_SOURCE as CL29_SOURCE,
    _EIGHT_ROPE_BLOCK as CL29_BLOCK,
    _COORDINATE_SETUP as CL29_SETUP,
)

SLUG = "debatt-ai-movie-cl31-eight-latent-blocks"
TITLE = "Debatt AI Movie CL31 Eight Latent Blocks"
REPORT_NAME = "cl31-eight-latent-blocks.json"
RESULTS = Path("kaggle-cl31-results")
SUBMISSION = Path("kaggle-cl31-submission")

_LATENT_SETUP = (
    '    # Synthetic latent tensor only; no video pixels or VAE encoding.\n'
    '    from comfy.ldm.lightricks.model import LTXVModel, LTXBaseModel\n'
    '    channels = int(root.in_channels)\n'
    '    if not (2 <= channels <= 512 and isinstance(root, LTXVModel)\n'
    '            and root.use_middle_indices_grid):\n'
    '        raise RuntimeError("Unexpected video latent model configuration")\n'
    '    proj = root.patchify_proj\n'
    '    w = getattr(proj, "weight", None)\n'
    '    if (not isinstance(proj, GGMLOps.Linear) or w is None\n'
    '            or not hasattr(w, "tensor_type") or len(w.shape) != 2\n'
    '            or tuple(w.shape) != (width, channels)\n'
    '            or w.shape[0] * w.shape[1] > 80_000_000):\n'
    '        raise RuntimeError("Unsafe latent patchify projection")\n'
    '    latent = torch.linspace(-1, 1, channels * 4, device="cuda:0",\n'
    '                            dtype=torch.float16).reshape(1, channels, 1, 2, 2)\n'
    '    vx, coords, _ = LTXVModel._process_input(root, latent, None, None)\n'
    '    if (tuple(vx.shape) != (1, 4, width)\n'
    '            or tuple(coords.shape) != (1, 3, 4, 2)\n'
    '            or not torch.isfinite(vx).all().item()):\n'
    '        raise RuntimeError("Model latent patchify output invalid")\n'
    '    # Bypass LTXAVModel AV override; use genuine video positions from patchifier.\n'
    '    pe = LTXBaseModel._prepare_positional_embeddings(\n'
    '        root, coords, frame_rate=25, x_dtype=torch.float16)\n'
    '    if not (isinstance(pe, tuple) and len(pe) == 2\n'
    '            and torch.is_tensor(pe[0])\n'
    '            and pe[0].shape[0:3] == (1, 4, blocks[0].attn1.heads)\n'
    '            and torch.isfinite(pe[0]).all().item()):\n'
    '        raise RuntimeError("Invalid model video RoPE from latent positions")\n'
)

_EIGHT_LATENT_BLOCK = CL29_BLOCK.replace(
    '    vx = torch.linspace(-1, 1, 4 * width, device="cuda:0", dtype=torch.float16).reshape(1, 4, width)\n',
    ''
).replace(
    CL29_SETUP,
    _LATENT_SETUP
).replace(
    '"coordinate_source": "synthetic_middle_grid", "frame_rate": 25,',
    '"coordinate_source": "latent_patchifier", "frame_rate": 25,\n'
    '              "real_patchify_projection": True, "synthetic_latent": True,\n'
    '              "vae_encoded": False, "real_video_input": False,'
)
assert _EIGHT_LATENT_BLOCK != CL29_BLOCK
assert 'LTXVModel._process_input(root, latent, None, None)' in _EIGHT_LATENT_BLOCK
assert 'coords = torch.tensor(' not in _EIGHT_LATENT_BLOCK
assert _EIGHT_LATENT_BLOCK.count('LTXBaseModel._prepare_positional_embeddings(') == 1
assert 'for index in range(8):' in _EIGHT_LATENT_BLOCK


def make_kernel():
    source = CL29_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL29_BLOCK) == 1
    child = child.replace(CL29_BLOCK, _EIGHT_LATENT_BLOCK)
    source = source.replace(lines[0], marker + repr(child))
    for old, new in (
        ('"comfy_gguf_eight_rope_blocks"', '"comfy_gguf_eight_latent_blocks"'),
        ('"cl29_checks"', '"cl31_checks"'),
        ('"cl29_disk_free_after_bytes"', '"cl31_disk_free_after_bytes"'),
        ('"cl29_torch_after"', '"cl31_torch_after"'),
        ('"cl29_cuda_after"', '"cl31_cuda_after"'),
        ('cl29-sources', 'cl31-sources'),
        ('cl29-eight-blocks-rope.json', REPORT_NAME),
        ('cl29_eight_rope_blocks_complete', 'cl31_eight_latent_blocks_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14
    first = report.get("cl31_checks", {}).get("first_weight", {})
    model = first.get("comfy_gguf_eight_latent_blocks", {})
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl31_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl31_" + suffix)
    validate_cl14(baseline, sha, run_id)
    result = model.get("eight_video_blocks", {})
    blocks = result.get("blocks", [])
    if (model.get("ok") is not True or model.get("comfy_model_constructed") is not True
            or model.get("full_model_forward") is not False or model.get("video_generated") is not False
            or result.get("block_indices") != list(range(8))
            or result.get("video_branches_complete") is not True
            or not isinstance(blocks, list) or len(blocks) != 8
            or any(b.get("block_index") != i or b.get("output_finite") is not True
                   or not all(2 <= b.get("layer_counts", {}).get(n, 0) <= 10
                              for n in ("attn1", "attn2", "ff"))
                   or not (0 <= b.get("gpu_allocated_after_bytes", -1) <= 4 * 1024**3)
                   or not (0 < b.get("gpu_peak_allocated_bytes", 0) <= 4 * 1024**3)
                   for i, b in enumerate(blocks))
            or result.get("tokens") != 4 or result.get("finite") is not True
            or result.get("input_shape") != result.get("output_shape")
            or result.get("input_shape") != [1, 4, 4096]
            or result.get("synthetic_input") is not True
            or result.get("synthetic_text_context") is not True
            or result.get("synthetic_timestep") is not True
            or result.get("synthetic_prompt_timestep") is not True
            or result.get("cross_attention_adaln") is not True
            or result.get("rotary_positions_applied") is not True
            or result.get("coordinate_source") != "latent_patchifier"
            or result.get("real_patchify_projection") is not True
            or result.get("synthetic_latent") is not True
            or result.get("vae_encoded") is not False
            or result.get("real_video_input") is not False
            or result.get("frame_rate") != 25
            or result.get("audio_branch_run") is not False
            or result.get("full_model_forward") is not False
            or len(model.get("gpu_peak_allocated_bytes", [])) != 2
            or any(v > 4 * 1024**3 for v in model["gpu_peak_allocated_bytes"])
            or not (0 < model.get("peak_child_rss_bytes", 0) <= 12_000_000_000)):
        raise ValueError("CL-31 eight latent-fed video blocks incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
