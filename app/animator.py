"""Turn a handful of keyframe images into a short MP4.

Technique (fully local, no extra models):
  - Each keyframe gets a Ken Burns move (slow zoom + slight pan).
  - Consecutive segments are blended with a cross-dissolve.
The result is a smooth ~10s "AI video" built from diffusion keyframes.
"""
import imageio
import numpy as np
from PIL import Image


def _prep(img: Image.Image, size: int) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize((size, size), Image.LANCZOS))


def _kenburns(img: np.ndarray, n: int, zoom: float):
    """Render n frames with a smooth zoom-in and subtle diagonal pan."""
    h, w = img.shape[:2]
    pil = Image.fromarray(img)
    frames = []
    for t in range(n):
        p = t / max(n - 1, 1)
        scale = 1.0 + zoom * p
        cw, ch = int(w / scale), int(h / scale)
        # pan drifts from slightly up-left to slightly down-right
        dx = int((p - 0.5) * w * zoom * 0.25)
        dy = int((p - 0.5) * h * zoom * 0.25)
        x0 = min(max((w - cw) // 2 + dx, 0), w - cw)
        y0 = min(max((h - ch) // 2 + dy, 0), h - ch)
        crop = pil.crop((x0, y0, x0 + cw, y0 + ch))
        frames.append(np.asarray(crop.resize((w, h), Image.BICUBIC)))
    return frames


def animate(keyframes, out_path, size=512, fps=24,
            segment_seconds=2.0, fade_seconds=0.5, zoom=0.10):
    """Write keyframes -> MP4 at out_path. Returns the output path."""
    frames_per_seg = max(int(fps * segment_seconds), 2)
    fade_frames = min(int(fps * fade_seconds), frames_per_seg // 2)

    segments = [_kenburns(_prep(k, size), frames_per_seg, zoom) for k in keyframes]

    # Stitch with cross-dissolves
    out = []
    for i, seg in enumerate(segments):
        if i == 0:
            out.extend(seg)
            continue
        for j in range(fade_frames):
            a = out[len(out) - fade_frames + j].astype(np.float32)
            b = seg[j].astype(np.float32)
            w = (j + 1) / (fade_frames + 1)
            out[len(out) - fade_frames + j] = (a * (1 - w) + b * w).astype(np.uint8)
        out.extend(seg[fade_frames:])

    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
    )
    try:
        for f in out:
            writer.append_data(f)
    finally:
        writer.close()
    return out_path
