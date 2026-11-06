"""CL-48: render a short scene with Wan 2.2 TI2V 5B on a Kaggle T4.

Same verified GitHub -> Kaggle -> FFprobe flow as CL-47, with a model about
four times larger. Wan 2.2 TI2V uses a VAE that compresses 16x spatially, so
the transformer sees far fewer tokens per frame than Wan 2.1 at the same
resolution. That is what makes a 5B model plausible on a 16 GB T4.
"""
import sys
from pathlib import Path

from scripts.kaggle_cl47_wan21_scene import Profile, main as render_main

WAN22_TI2V_5B = Profile(
    name="Wan 2.2 TI2V 5B", slug="debatt-ai-movie-wan22-ti2v-5b-scene",
    title="Debatt AI Movie Wan22 TI2V 5B Scene", model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    stem="wan22-scene", results=Path("kaggle-cl48-results"),
    submission=Path("kaggle-cl48-submission"), fps=24,
    # Height and width must be multiples of 32 (16x VAE times 2x2 patches).
    resolutions={"832x480": (480, 832), "480x832": (832, 480),
                 "1280x704": (704, 1280), "704x1280": (1280, 704)},
    frame_choices=(25, 49, 73, 121), dtypes=("float16", "bfloat16"), max_steps=50,
    flow_shift=None,  # keep the shift from the model repository's scheduler config
    # CL-47 measured ~15 s/step for Wan 2.1 1.3B at 14,040 tokens. Scaling by
    # parameters, 15,000 tokens should stay near one minute per step for 5B.
    max_latent_tokens=15_000, kaggle_timeout_seconds=7200, max_polls=140)

DEFAULTS = {"resolution": "832x480", "num_frames": 49, "steps": 30}


def main(argv=None):
    return render_main(argv, profile=WAN22_TI2V_5B, defaults=DEFAULTS)


if __name__ == "__main__":
    sys.exit(main())
