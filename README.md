# VideoAI 🎬 — local text-to-video

Turn a text prompt into a short video, entirely on your own machine. No cloud, no API keys.

**How it works:** a local diffusion model (SD-Turbo by default) renders several
keyframe images from your prompt, then a Ken Burns + cross-dissolve animator
stitches them into a smooth 7–17 second MP4.

## Quick start

Double-click **`run.bat`**. First run installs everything (a few minutes) and
downloads the model (~5 GB, one time, from ModelScope — huggingface.co is
blocked on this network, so `scripts/fetch_model.py` pulls it from
www.modelscope.cn instead). The app opens at http://127.0.0.1:8000.

## API

```
POST /api/generate      { "prompt": "...", "style": "cinematic", "num_keyframes": 6, "seed": null }
GET  /api/jobs/{id}     -> { status, progress, stage, result: { video, seed } }
GET  /api/gallery       -> recent videos
GET  /outputs/{id}.mp4  -> the video files themselves
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `VTAI_IMAGE_MODEL` | local `models/sd-turbo` (or `AI-ModelScope/sd-turbo`) | Any diffusers text2img model |
| `VTAI_SIZE` | `512` | Output resolution (square) |
| `VTAI_FPS` | `24` | Frames per second |
| `VTAI_KEYFRAMES` | `6` | Default keyframe count |
| `VTAI_SEG_SECONDS` | `2.0` | Seconds per keyframe segment |
| `VTAI_ZOOM` | `0.10` | Ken Burns zoom strength |

## Running on an NVIDIA GPU (phase 2)

This machine has an Intel UHD 770 iGPU, so everything runs on the CPU
(a 6-keyframe video takes roughly 1–3 minutes). On any CUDA machine:

1. Install CUDA PyTorch: `.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124`
2. Set `VTAI_IMAGE_MODEL=stabilityai/sdxl-turbo` (or bigger) for higher quality.
3. Everything else works unchanged — `app/backends.py` detects CUDA automatically.

For real *video diffusion* (not keyframe animation) on a 12 GB+ NVIDIA GPU, the
natural next step is `LTX-Video` or `Wan2.1-T2V-1.3B` via the same FastAPI app.

## Project layout

```
video-ai/
├── app/
│   ├── main.py       # FastAPI server + API
│   ├── generator.py  # diffusion keyframe generation
│   ├── animator.py   # Ken Burns + crossfade -> MP4
│   ├── jobs.py       # background job queue
│   ├── backends.py   # CUDA / CPU auto-detection
│   └── config.py     # settings
├── web/index.html    # UI
├── outputs/          # generated videos
└── run.bat           # one-click start
```
