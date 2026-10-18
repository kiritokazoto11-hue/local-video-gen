"""FastAPI app: web UI + generation API."""
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, jobs
from .backends import device_info

app = FastAPI(title="VideoAI")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    style: str = "none"          # none | cinematic | anime | claymation | cyberpunk
    num_keyframes: int = Field(default=config.DEFAULT_KEYFRAMES, ge=3, le=10)
    steps: int = Field(default=config.DEFAULT_STEPS, ge=1, le=8)
    size: int = Field(default=config.DEFAULT_SIZE, ge=384, le=768)
    fps: int = Field(default=config.DEFAULT_FPS, ge=12, le=30)
    motion: str = Field(
        default="auto",
        pattern="^(auto|zoom_in|zoom_out|pan_left|pan_right|tilt_up|tilt_down|orbit|drift|pulse|rise|handheld)$",
    )              # camera preset; 'auto' varies them across segments
    morph: bool = True           # optical-flow morph between segments
    effects: bool = True         # vignette + grain polish
    coherent: bool = True        # evolve each keyframe from the previous one (one continuous scene)
    flow_strength: float = Field(default=0.55, ge=0.2, le=0.9)  # how much each chained step changes
    seed: int | None = None


STYLE_SUFFIX = {
    "none": "",
    "cinematic": ", cinematic lighting, film still, dramatic, 35mm",
    "anime": ", anime style, vibrant colors, detailed illustration",
    "claymation": ", claymation style, stop motion, soft lighting",
    "cyberpunk": ", cyberpunk, neon lights, futuristic, rain",
}


def _run_generation(job_id: str, req: GenerateRequest):
    from .generator import generate_image, generate_variation
    from .animator import animate

    try:
        jobs.update(job_id, status="running", stage="loading model")
        base_seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "big")
        prompt = req.prompt + STYLE_SUFFIX.get(req.style, "")
        frames = []
        if req.coherent:
            # Frame 1 from scratch, then each frame evolved from the last ->
            # one continuous scene that the morph transitions animate.
            jobs.update(job_id, stage=f"generating keyframe 1/{req.num_keyframes}",
                        progress=0 / (req.num_keyframes + 1))
            frames.append(generate_image(prompt, seed=base_seed, steps=req.steps, size=req.size))
            for i in range(1, req.num_keyframes):
                jobs.update(
                    job_id,
                    stage=f"evolving scene {i + 1}/{req.num_keyframes}",
                    progress=i / (req.num_keyframes + 1),
                )
                frames.append(generate_variation(
                    prompt, frames[-1], seed=base_seed + i,
                    strength=req.flow_strength, size=req.size,
                ))
        else:
            for i in range(req.num_keyframes):
                jobs.update(
                    job_id,
                    stage=f"generating keyframe {i + 1}/{req.num_keyframes}",
                    progress=i / (req.num_keyframes + 1),
                )
                frames.append(generate_image(prompt, seed=base_seed + i, steps=req.steps, size=req.size))

        jobs.update(job_id, stage="rendering video", progress=req.num_keyframes / (req.num_keyframes + 1))
        out_path = config.OUTPUT_DIR / f"{job_id}.mp4"
        animate(
            frames, out_path,
            size=req.size, fps=req.fps,
            segment_seconds=config.SEGMENT_SECONDS,
            fade_seconds=config.FADE_SECONDS,
            zoom=config.ZOOM_AMOUNT,
            motion=req.motion,
            morph=req.morph,
            effects=req.effects,
            seed=base_seed,
        )
        jobs.update(job_id, status="done", progress=1.0, stage="done",
                    result={"video": f"/outputs/{job_id}.mp4", "seed": base_seed})
    except Exception as e:  # surface any failure to the UI
        jobs.update(job_id, status="error", error=str(e))


@app.get("/api/health")
def health():
    return device_info()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    job = jobs.create_job("video")
    jobs.submit(_run_generation, job["id"], req)
    return {"job_id": job["id"]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/gallery")
def gallery():
    vids = sorted(config.OUTPUT_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True)
    return [{"video": f"/outputs/{p.name}", "name": p.name} for p in vids[:50]]


# Static assets
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")


@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html")
