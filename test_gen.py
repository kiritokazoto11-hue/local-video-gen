"""End-to-end smoke test: prompt -> keyframes -> MP4."""
import time

from app.generator import generate_image
from app.animator import animate

prompt = "a misty mountain lake at sunrise, golden light, calm water reflections"

t0 = time.time()
print("Generating keyframe 1/2 ...", flush=True)
a = generate_image(prompt, seed=42, steps=1, size=512)
print(f"  keyframe 1 done in {time.time() - t0:.1f}s", flush=True)

t1 = time.time()
b = generate_image(prompt, seed=43, steps=1, size=512)
print(f"  keyframe 2 done in {time.time() - t1:.1f}s", flush=True)

out = animate([a, b], "outputs/smoke_test.mp4", size=512, fps=24)
print("SMOKE TEST OK ->", out, flush=True)
