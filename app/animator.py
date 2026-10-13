"""Turn keyframe images into a short MP4 with real camera movement.

Features:
  - Per-segment camera presets: zoom in/out, pans, tilts, orbit sweeps,
    drift, pulse, rise, handheld shake ("auto" picks varied motions)
  - Eased motion (smooth accelerate / decelerate)
  - Optical-flow morph transitions: consecutive frames genuinely warp into
    each other (Farneback flow) instead of a flat cross-dissolve
  - Subtle film polish: vignette + fine grain

All pixel buffers are RGB throughout (PIL -> numpy -> imageio).
"""
import random

import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Camera presets: fn(p, size, z) -> (scale, tx_px, ty_px, angle_deg)
#   p    eased progress within the segment, 0..1
#   size output edge length in px
#   z    zoom/motion strength (e.g. 0.10 = 10%)
# --------------------------------------------------------------------------

def _zoom_in(p, size, z):
    return (1 + z * p, 0.0, size * z * 0.10 * p, 0.0)


def _zoom_out(p, size, z):
    return (1 + z * (1 - p), 0.0, -size * z * 0.10 * (1 - p), 0.0)


def _pan_right(p, size, z):
    a = size * z * 0.35
    return (1 + z * 0.5, a * (2 * p - 1), 0.0, 0.0)


def _pan_left(p, size, z):
    a = size * z * 0.35
    return (1 + z * 0.5, a * (1 - 2 * p), 0.0, 0.0)


def _tilt_up(p, size, z):
    a = size * z * 0.35
    return (1 + z * 0.5, 0.0, a * (1 - 2 * p), 0.0)


def _tilt_down(p, size, z):
    a = size * z * 0.35
    return (1 + z * 0.5, 0.0, a * (2 * p - 1), 0.0)


def _orbit(p, size, z):
    a = size * z * 0.22
    ang = (2 + 8 * z) * (2 * p - 1)
    return (1 + z, a * np.sin(2 * np.pi * p), -a * (1 - np.cos(2 * np.pi * p)), ang)


def _drift(p, size, z):
    a = size * z * 0.30
    return (1 + z * p, a * (2 * p - 1), a * (p - 0.5) * 0.6, 0.0)


def _pulse(p, size, z):
    return (1 + z * 0.8 * np.sin(np.pi * p), 0.0, 0.0, 0.0)


def _rise(p, size, z):
    a = size * z * 0.30
    return (1 + z * 0.7 * (1 - p), 0.0, a * (1 - 2 * p) * 0.6, 0.0)


def _handheld(p, size, z, phase=0.0):
    a = size * z * 0.16
    tx = a * np.sin(6.0 * p + phase)
    ty = a * np.sin(4.7 * p + phase * 1.7 + 1.1)
    ang = (3 + 6 * z) * 0.25 * np.sin(3.3 * p + phase * 0.6)
    return (1 + z * 0.5, tx, ty, ang)


MOTIONS = {
    "zoom_in": _zoom_in, "zoom_out": _zoom_out,
    "pan_right": _pan_right, "pan_left": _pan_left,
    "tilt_up": _tilt_up, "tilt_down": _tilt_down,
    "orbit": _orbit, "drift": _drift, "pulse": _pulse,
    "rise": _rise, "handheld": _handheld,
}


def _ease(p):
    """Smoothstep: gentle acceleration and deceleration."""
    return p * p * (3 - 2 * p)


def _choose_motions(n, motion, rng):
    """Pick one preset per segment. 'auto' varies them and avoids repeats."""
    if motion != "auto":
        return [motion] * n
    names = list(MOTIONS)
    picks = []
    for _ in range(n):
        choices = [m for m in names if not picks or m != picks[-1]]
        picks.append(rng.choice(choices))
    return picks


def _render(img, scale, tx, ty, angle):
    """Apply one camera transform (scale + pan + rotate about the center)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _segment_frames(img, preset, n, size, z, phase):
    frames = []
    for i in range(n):
        p = _ease(i / max(n - 1, 1))
        if preset == "handheld":
            scale, tx, ty, ang = _handheld(p, size, z, phase)
        else:
            scale, tx, ty, ang = MOTIONS[preset](p, size, z)
        frames.append(_render(img, scale, tx, ty, ang))
    return frames


# --------------------------------------------------------------------------
# Optical-flow morph transition
# --------------------------------------------------------------------------

def _flow(a, b):
    """Dense forward flow a->b (HxWx2), computed at half resolution."""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    h, w = ga.shape
    small = (max(w // 2, 8), max(h // 2, 8))
    f = cv2.calcOpticalFlowFarneback(
        cv2.resize(ga, small), cv2.resize(gb, small), None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=2,
        poly_n=5, poly_sigma=1.1, flags=0,
    )
    f = cv2.resize(f, (w, h))
    f[..., 0] *= w / small[0]
    f[..., 1] *= h / small[1]
    return f


def _warp(img, fx, fy):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + fx, gy + fy, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def _morph(a, b, n):
    """n intermediate frames where a warps toward b while b warps back."""
    try:
        f = _flow(a, b)
        fx, fy = f[..., 0].astype(np.float32), f[..., 1].astype(np.float32)
        out = []
        for i in range(1, n + 1):
            t = i / (n + 1)
            at = _warp(a, fx * t, fy * t)
            bt = _warp(b, -fx * (1 - t), -fy * (1 - t))
            out.append(np.clip(at * (1 - t) + bt * t, 0, 255).astype(np.uint8))
        return out
    except cv2.error:
        # flow failed -> plain dissolve fallback
        return [np.clip(a * (1 - (i + 1) / (n + 1)) + b * ((i + 1) / (n + 1)),
                        0, 255).astype(np.uint8) for i in range(n)]


# --------------------------------------------------------------------------
# Film polish
# --------------------------------------------------------------------------

def _vignette_mask(size, strength=0.22):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    r = np.sqrt(((x - c) / c) ** 2 + ((y - c) / c) ** 2) / np.sqrt(2)
    return (1 - strength * r ** 2)[..., None].astype(np.float32)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def animate(keyframes, out_path, size=512, fps=24,
            segment_seconds=2.0, fade_seconds=0.5, zoom=0.10,
            motion="auto", morph=True, effects=True, seed=None):
    """Write keyframes -> MP4 at out_path. Returns the output path."""
    import imageio

    rng = random.Random(seed)
    grain = np.random.default_rng(seed)
    frames_per_seg = max(int(fps * segment_seconds), 2)
    trans_frames = min(int(fps * fade_seconds), frames_per_seg // 2)
    picks = _choose_motions(len(keyframes), motion, rng)

    segments = []
    for kf, preset in zip(keyframes, picks):
        img = np.asarray(kf.convert("RGB").resize((size, size), Image.LANCZOS))
        phase = rng.uniform(0, 6.28)
        segments.append(_segment_frames(img, preset, frames_per_seg, size, zoom, phase))

    # Stitch: morph (or dissolve) transitions between moving segments
    out = []
    for i, seg in enumerate(segments):
        if i == 0 or trans_frames == 0:
            out.extend(seg)
            continue
        prev = out[len(out) - 1]
        if morph:
            out.extend(_morph(prev, seg[0], trans_frames))
        else:
            for j in range(trans_frames):
                t = (j + 1) / (trans_frames + 1)
                out.append(np.clip(prev * (1 - t) + seg[0] * t,
                                   0, 255).astype(np.uint8))
        out.extend(seg[1:])

    # Polish + encode
    vig = _vignette_mask(size) if effects else None
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
    )
    try:
        for f in out:
            if vig is not None:
                f = f.astype(np.float32) * vig
                f += grain.normal(0, 2.0, f.shape)
                f = np.clip(f, 0, 255).astype(np.uint8)
            writer.append_data(f)
    finally:
        writer.close()
    return out_path
