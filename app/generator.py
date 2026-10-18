"""Text -> keyframe images using a locally-run diffusion model (lazy-loaded)."""
import threading

import torch

# config must be imported before diffusers so IMAGE_MODEL points at the
# local model dir (models/sd-turbo) when it exists.
from .config import IMAGE_MODEL, DEFAULT_SIZE
from .backends import detect_device
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

_pipe = None
_i2i_pipe = None
_pipe_lock = threading.RLock()   # reentrant: img2img derives from txt2img under the same lock


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


def get_img2img_pipeline():
    """Derive an img2img pipeline from the loaded txt2img pipe.

    from_pipe() shares all weights -- no second copy in RAM, no reload.
    """
    global _i2i_pipe
    with _pipe_lock:
        if _i2i_pipe is None:
            _i2i_pipe = AutoPipelineForImage2Image.from_pipe(get_pipeline())
    return _i2i_pipe


def generate_variation(prompt: str, init_image, seed: int,
                       strength: float = 0.55, steps: int = 4,
                       size: int = DEFAULT_SIZE):
    """Coherent-scene step: evolve the previous keyframe into the next one.

    The init image anchors composition (so the scene stays recognisable)
    while the noise/strength injects controlled change -- which is exactly
    what the morph transitions then animate frame-to-frame.
    """
    pipe = get_img2img_pipeline()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=init_image.resize((size, size)),
        num_inference_steps=steps,
        strength=strength,           # fraction of the denoising path re-run
        guidance_scale=0.0,
        generator=gen,
    )
    return result.images[0]


def warmup():
    """Pre-download and load the model so the first request is fast."""
    get_pipeline()
