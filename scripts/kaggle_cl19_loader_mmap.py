"""CL-19: exercise ComfyUI-GGUF's actual state-dict reader, without model load."""
from pathlib import Path

from scripts import kaggle_cl7_import_probe as cl7
from scripts.kaggle_cl14_gguf_structure import KERNEL_SOURCE as CL14_SOURCE

SLUG = "debatt-ai-movie-cl19-gguf-loader-mmap"
TITLE = "Debatt AI Movie CL19 GGUF Loader Mmap"
REPORT_NAME = "cl19-gguf-loader-mmap.json"
RESULTS = Path("kaggle-cl19-results")
SUBMISSION = Path("kaggle-cl19-submission")

_MARKER = '''    except Exception as exc:
        first["error"] = (type(exc).__name__ + ": " + str(exc))[:500]
    finally:
        # Never let Kaggle output retrieval archive giant checkpoints.'''

_PROBE = '''        # Invoke the *actual* ComfyUI-GGUF GGUF-to-state-dict loader. Never
        # invoke comfy.sd.load_diffusion_model_state_dict or ModelPatcher:
        # those may materialize the full 22B model and exceed the T4 memory.
        import importlib
        import sys
        import types
        import time
        import psutil
        import torch
        pkg_name = "cl19_gguf_package"
        package = types.ModuleType(pkg_name)
        package.__path__ = [str(sources / "ComfyUI-GGUF")]
        sys.modules[pkg_name] = package
        sys.path.insert(0, str(sources / "ComfyUI"))
        state_dict = None
        extra = None
        try:
            module = importlib.import_module(pkg_name + ".loader")
            process = psutil.Process()
            rss_before = process.memory_info().rss
            gpu_before = [torch.cuda.memory_allocated(i) for i in range(2)]
            started = time.monotonic()
            def loader_deadline(signum, frame):
                raise TimeoutError("ComfyUI-GGUF state-dict reader exceeded 120 seconds")
            saved_handler = signal.signal(signal.SIGALRM, loader_deadline)
            signal.setitimer(signal.ITIMER_REAL, 120)
            try:
                state_dict, extra = module.gguf_sd_loader(str(stage))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, saved_handler)
            elapsed = round(time.monotonic() - started, 3)
            rss_after = process.memory_info().rss
            gpu_after = [torch.cuda.memory_allocated(i) for i in range(2)]
            count = len(state_dict)
            if not (1000 <= count <= 100_000):
                raise ValueError("Unexpected GGUF loader state-dict tensor count")
            if rss_after - rss_before > 3_000_000_000:
                raise ValueError("GGUF loader exceeded 3 GB additional resident RAM")
            if any(after - before > 16 * 1024 * 1024
                   for before, after in zip(gpu_before, gpu_after)):
                raise ValueError("GGUF loader unexpectedly allocated GPU weights")
            # Some BF16 non-linear parameters may have GGML packed shapes.
            # Inspect metadata only; never dequantize the entire checkpoint.
            special = (
                "audio_embeddings_connector.learnable_registers",
                "video_embeddings_connector.learnable_registers",
                "keyframes_abs_pos_embedding",
            )
            watched = []
            for suffix in special:
                matches = [(key, value) for key, value in state_dict.items()
                           if key.endswith(suffix)]
                watched.append({
                    "suffix": suffix, "matches": len(matches),
                    "tensors": [{"name": key[:160],
                                 "python_type": type(value).__name__,
                                 "shape": [int(n) for n in value.shape][:8],
                                 "tensor_type": str(getattr(value, "tensor_type", ""))[:80]}
                                for key, value in matches[:2]],
                })
            first["comfy_gguf_loader"] = {
                "ok": True,
                "entrypoint": "gguf_sd_loader",
                "tensor_count": count,
                "elapsed_seconds": elapsed,
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
                "gpu_allocated_delta_bytes": [a - b for a, b in zip(gpu_after, gpu_before)],
                "architecture": str(extra.get("arch_str", ""))[:100],
                "special_bf16_tensors": watched,
                "state_dict_parsed_only": True,
                "comfy_model_constructed": False,
                "full_model_forward": False,
                "video_generated": False,
            }
        finally:
            state_dict = None
            extra = None
            gc.collect()
            sys.modules.pop(pkg_name, None)
            if str(sources / "ComfyUI") in sys.path:
                sys.path.remove(str(sources / "ComfyUI"))
'''


def make_kernel():
    source = CL14_SOURCE
    assert source.count(_MARKER) == 1
    source = source.replace(_MARKER, _PROBE + _MARKER)
    for old, new in (
        ("cl14-sources", "cl19-sources"),
        ('"cl14_checks"', '"cl19_checks"'),
        ('"cl14_disk_free_after_bytes"', '"cl19_disk_free_after_bytes"'),
        ('"cl14_torch_after"', '"cl19_torch_after"'),
        ('"cl14_cuda_after"', '"cl19_cuda_after"'),
        ("cl14-gguf-structure.json", REPORT_NAME),
        ("cl14_gguf_structure_complete", "cl19_gguf_loader_mmap_complete"),
    ):
        assert old in source, old
        source = source.replace(old, new)
    return source


KERNEL_SOURCE = make_kernel()


def validate_report(report, sha, run_id):
    first = report.get("cl19_checks", {}).get("first_weight", {})
    loader = first.get("comfy_gguf_loader", {})
    from scripts.kaggle_cl14_gguf_structure import validate_report as validate_cl14

    # Reuse the established CL-14 safety requirements with CL-19 field aliases.
    baseline = dict(report)
    baseline["cl14_checks"] = report.get("cl19_checks", {})
    for suffix in ("torch_after", "cuda_after", "disk_free_after_bytes"):
        baseline["cl14_" + suffix] = report.get("cl19_" + suffix)
    validate_cl14(baseline, sha, run_id)
    if (loader.get("ok") is not True
            or loader.get("entrypoint") != "gguf_sd_loader"
            or not (1000 <= loader.get("tensor_count", 0) <= 100_000)
            or loader.get("state_dict_parsed_only") is not True
            or loader.get("comfy_model_constructed") is not False
            or loader.get("full_model_forward") is not False
            or loader.get("video_generated") is not False
            or len(loader.get("gpu_allocated_delta_bytes", [])) != 2
            or any(n > 16 * 1024 * 1024 for n in loader["gpu_allocated_delta_bytes"])
            or loader.get("rss_after_bytes", 0) - loader.get("rss_before_bytes", 0)
            > 3_000_000_000
            or len(loader.get("special_bf16_tensors", [])) != 3):
        raise ValueError("Incomplete or unsafe CL-19 ComfyUI-GGUF mmap report")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
