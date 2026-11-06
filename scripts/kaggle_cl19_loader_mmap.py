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
        import subprocess
        import sys
        import time
        import psutil
        # A parent watchdog can terminate a child even if the loader is stuck.
        # The child reports PyTorch peak allocated bytes, not just its final
        # allocation. The parent's RSS measurements include transient peaks.
        child_code = __CL19_CHILD_CODE__
        child = subprocess.Popen(
            [sys.executable, "-c", child_code, str(sources), str(stage)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        process = psutil.Process(child.pid)
        peak_rss = 0
        started = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic() - started > 120:
                    raise TimeoutError("ComfyUI-GGUF loader exceeded 120 seconds")
                try:
                    rss = process.memory_info().rss
                except psutil.NoSuchProcess:
                    break
                peak_rss = max(peak_rss, rss)
                if peak_rss > 3_000_000_000:
                    raise MemoryError("Loader child exceeded 3 GB resident RAM")
                time.sleep(0.05)
            output, errors = child.communicate(timeout=5)
            if child.returncode:
                raise RuntimeError("Loader child failed: " + errors[-400:])
            # Upstream loader may print progress before the final JSON record.
            data = json.loads(output.strip().splitlines()[-1])
            if any(n > 16 * 1024 * 1024 for n in data["gpu_peak_allocated_bytes"]):
                raise MemoryError("Loader GPU peak allocation exceeded 16 MiB")
            first["comfy_gguf_loader"] = dict(data, ok=True,
                peak_child_rss_bytes=peak_rss,
                elapsed_seconds=round(time.monotonic() - started, 3))
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
'''

_CHILD_CODE = '''import importlib
import json
import sys
import types
from pathlib import Path
import torch

sources, stage = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(sources / "ComfyUI"))
pkg_name = "cl19_gguf_package"
package = types.ModuleType(pkg_name)
package.__path__ = [str(sources / "ComfyUI-GGUF")]
sys.modules[pkg_name] = package
module = importlib.import_module(pkg_name + ".loader")
before = [torch.cuda.memory_allocated(i) for i in range(2)]
for i in range(2):
    torch.cuda.reset_peak_memory_stats(i)
state_dict, extra = module.gguf_sd_loader(stage)
count = len(state_dict)
if not (1000 <= count <= 100_000):
    raise ValueError("Unexpected GGUF loader tensor count")
special = (
    "audio_embeddings_connector.learnable_registers",
    "video_embeddings_connector.learnable_registers",
    "keyframes_abs_pos_embedding",
)
watched = []
for suffix in special:
    matches = [(key, value) for key, value in state_dict.items()
               if key.endswith(suffix)]
    watched.append({"suffix": suffix, "matches": len(matches),
        "tensors": [{"name": key[:160], "python_type": type(value).__name__,
            "shape": [int(n) for n in value.shape][:8],
            "tensor_type": str(getattr(value, "tensor_type", ""))[:80]}
            for key, value in matches[:2]]})
print(json.dumps({
    "entrypoint": "gguf_sd_loader", "tensor_count": count,
    "gpu_peak_allocated_bytes": [max(0, torch.cuda.max_memory_allocated(i) - before[i])
                                 for i in range(2)],
    "architecture": str(extra.get("arch_str", ""))[:100],
    "special_bf16_tensors": watched,
    "state_dict_parsed_only": True, "comfy_model_constructed": False,
    "full_model_forward": False, "video_generated": False,
}))
'''


def make_kernel():
    source = CL14_SOURCE
    assert source.count(_MARKER) == 1
    source = source.replace(_MARKER, _PROBE.replace("__CL19_CHILD_CODE__", repr(_CHILD_CODE)) + _MARKER)
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
            or len(loader.get("gpu_peak_allocated_bytes", [])) != 2
            or any(n > 16 * 1024 * 1024 for n in loader["gpu_peak_allocated_bytes"])
            or not (0 < loader.get("peak_child_rss_bytes", 0) <= 3_000_000_000)
            or len(loader.get("special_bf16_tensors", [])) != 3):
        raise ValueError("Incomplete or unsafe CL-19 ComfyUI-GGUF mmap report")


def main():
    cl7.SLUG, cl7.TITLE, cl7.REPORT_NAME = SLUG, TITLE, REPORT_NAME
    cl7.RESULTS, cl7.SUBMISSION = RESULTS, SUBMISSION
    cl7.KERNEL_SOURCE, cl7.validate_report = KERNEL_SOURCE, validate_report
    cl7.main()


if __name__ == "__main__":
    main()
