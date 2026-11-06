"""CL-35: probe LTXAV native timestep embeddings without changing diffusion loop."""
import ast
import math
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl34_two_latent_steps import (
    KERNEL_SOURCE as CL34_SOURCE, _TWO_STEP_BLOCK as CL34_BLOCK,
    validate_report as validate_cl34,
)

SLUG = "debatt-ai-movie-cl35-native-timestep-probe"
TITLE = "Debatt AI Movie CL35 Native Timestep Probe"
REPORT_NAME = "cl35-native-timestep-probe.json"
RESULTS = Path("kaggle-cl35-results")
SUBMISSION = Path("kaggle-cl35-submission")

_TIMESTEP_PROBE = (
    '    # Actual checkpoint-backed LTXAV AdaLN timestep embedding, but the\n'
    '    # timestep values themselves are synthetic and NOT a real scheduler.\n'
    '    from comfy.ldm.lightricks.av_model import LTXAVModel, CompressedTimestep\n'
    '    if not isinstance(root, LTXAVModel):\n'
    '        raise RuntimeError("Expected native LTXAV timestep model")\n'
    '    native_ts = []\n'
    '    for ts_value in (1.0, 0.96875):\n'
    '        ts = torch.full((1, 4), ts_value, device="cuda:0", dtype=torch.float32)\n'
    '        prepared, embedded, _ = root._prepare_timestep(\n'
    '            ts, batch_size=1, hidden_dtype=torch.float16,\n'
    '            orig_shape=tuple(latent.shape), has_spatial_mask=False)\n'
    '        if (not isinstance(prepared, (list, tuple)) or len(prepared) != 5\n'
    '                or not isinstance(embedded, (list, tuple)) or len(embedded) != 2\n'
    '                or not isinstance(prepared[0], CompressedTimestep)\n'
    '                or not isinstance(embedded[0], CompressedTimestep)\n'
    '                or prepared[3] is None):\n'
    '            raise RuntimeError("Unexpected native AV timestep outputs")\n'
    '        v_mod = prepared[0].data\n'
    '        v_out = embedded[0].data\n'
    '        v_prompt = prepared[3]\n'
    '        if (tuple(v_mod.shape) != (1, 1, 9 * width)\n'
    '                or tuple(v_out.shape) != (1, 1, width)\n'
    '                or tuple(v_prompt.shape) != (1, 1, 2 * width)\n'
    '                or any(not torch.isfinite(x).all().item()\n'
    '                       for x in (v_mod, v_out, v_prompt))):\n'
    '            raise RuntimeError("Invalid native AV timestep dimensions/values")\n'
    '        native_ts.append((v_mod.detach().float(), v_out.detach().float(),\n'
    '                          v_prompt.detach().float()))\n'
    '    delta_mod = float((native_ts[0][0] - native_ts[1][0]).abs().max().item())\n'
    '    delta_out = float((native_ts[0][1] - native_ts[1][1]).abs().max().item())\n'
    '    delta_prompt = float((native_ts[0][2] - native_ts[1][2]).abs().max().item())\n'
    '    if not (0 < delta_mod < 1_000_000 and 0 < delta_out < 1_000_000\n'
    '            and 0 < delta_prompt < 1_000_000):\n'
    '        raise RuntimeError("Timestep embedding did not respond to changed sigma")\n'
    '    native_spec = {"native_av_prepare_timestep": True,\n'
    '                   "checkpoint_adaln_weights": True,\n'
    '                   "per_frame_compressed": True,\n'
    '                   "native_embedding_used_in_model_forward": False,\n'
    '                   "real_ltx_scheduler": False,\n'
    '                   "synthetic_sigma_values": [1.0, 0.96875],\n'
    '                   "modulation_shape": [1, 1, 9 * width],\n'
    '                   "embedded_shape": [1, 1, width],\n'
    '                   "prompt_shape": [1, 1, 2 * width],\n'
    '                   "delta_mod": delta_mod, "delta_out": delta_out,\n'
    '                   "delta_prompt": delta_prompt, "finite": True,\n'
    '                   "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0))}\n'
)

_NATIVE_TIMESTEP_BLOCK = CL34_BLOCK.replace(
    '    result = {"block_indices": list(range(8)),',
    _TIMESTEP_PROBE + '    result = {"block_indices": list(range(8)),'
).replace(
    '"two_step": two_step_spec,',
    '"two_step": two_step_spec, "native_timestep": native_spec,'
)
assert _NATIVE_TIMESTEP_BLOCK != CL34_BLOCK
assert '"native_timestep": native_spec' in _NATIVE_TIMESTEP_BLOCK


def make_kernel():
    source = CL34_SOURCE
    marker = '        child_code = '
    lines = [line for line in source.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    child = ast.literal_eval(lines[0][len(marker):])
    assert child.count(CL34_BLOCK) == 1
    source = source.replace(lines[0], marker + repr(child.replace(CL34_BLOCK, _NATIVE_TIMESTEP_BLOCK)))
    for old, new in (
        ('"comfy_gguf_two_latent_steps"', '"comfy_gguf_native_timestep"'),
        ('"cl34_checks"', '"cl35_checks"'),
        ('"cl34_disk_free_after_bytes"', '"cl35_disk_free_after_bytes"'),
        ('"cl34_torch_after"', '"cl35_torch_after"'),
        ('"cl34_cuda_after"', '"cl35_cuda_after"'),
        ('cl34-sources', 'cl35-sources'),
        ('cl34-two-latent-steps.json', REPORT_NAME),
        ('cl34_two_latent_steps_complete', 'cl35_native_timestep_complete'),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    checks = report.get("cl35_checks", {})
    first = checks.get("first_weight", {})
    model = first.get("comfy_gguf_native_timestep", {})
    prior = dict(report)
    old_checks = dict(checks)
    old_first = dict(first)
    old_first["comfy_gguf_two_latent_steps"] = model
    old_checks["first_weight"] = old_first
    prior["cl34_checks"] = old_checks
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        prior["cl34_" + suffix] = report.get("cl35_" + suffix)
    validate_cl34(prior, sha, run_id)
    native = model.get("eight_video_blocks", {}).get("native_timestep", {})
    if (native.get("native_av_prepare_timestep") is not True
            or native.get("checkpoint_adaln_weights") is not True
            or native.get("per_frame_compressed") is not True
            or native.get("native_embedding_used_in_model_forward") is not False
            or native.get("real_ltx_scheduler") is not False
            or native.get("synthetic_sigma_values") != [1.0, 0.96875]
            or native.get("modulation_shape") != [1, 1, 9 * 4096]
            or native.get("embedded_shape") != [1, 1, 4096]
            or native.get("prompt_shape") != [1, 1, 2 * 4096]
            or native.get("finite") is not True
            or any(not isinstance(native.get(k), (float, int))
                   or not math.isfinite(native[k]) or not (0 < native[k] < 1_000_000)
                   for k in ("delta_mod", "delta_out", "delta_prompt"))
            or not (0 < native.get("gpu_peak_allocated_bytes", 0) <= 4 * 1024**3)):
        raise ValueError("CL-35 real AdaLN timestep probe incomplete or unsafe")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
