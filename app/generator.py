"""Text -> keyframe images using a locally-run diffusion model (lazy-loaded)."""
import threading

import torch

# config must be imported before diffusers so IMAGE_MODEL points at the
# local model dir (models/sd-turbo) when it exists.
from .config import IMAGE_MODEL, DEFAULT_SIZE
from .backends import detect_device
from diffusers import AutoPipelineForText2Image

_pipe = None
_pipe_lock = threading.Lock()


def get_pipeline():
    """Load the diffusion pipeline once, on first use."""
    global _pipe
    with _pipe_lock:
        if _pipe is None:
            pipe = AutoPipelineForText2Image.from_pretrained(
                IMAGE_MODEL,
                torch_dtype=torch.float32,   # CPU-friendly; use torch.float16 on GPU
                safety_checker=None,
            )
            pipe.set_progress_bar_config(disable=True)
            pipe.to(detect_device())
            _pipe = pipe
    return _pipe


def generate_image(prompt: str, seed: int, steps: int = 1, size: int = DEFAULT_SIZE):
    """Generate a single keyframe image. Returns a PIL.Image."""
    pipe = get_pipeline()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,          # turbo models are unguided
        width=size,
        height=size,
        generator=gen,
    )
    return result.images[0]


def warmup():
    """Pre-download and load the model so the first request is fast."""
    get_pipeline()
