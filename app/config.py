"""Central configuration. Everything can be overridden with env vars.

Note: huggingface.co is unreachable from this network, and hf-mirror.com now
just redirects back to it. The model is instead fetched from ModelScope
(www.modelscope.cn) via scripts/fetch_model.py into models/sd-turbo.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("VTAI_OUTPUT_DIR", BASE_DIR / "outputs"))
WEB_DIR = BASE_DIR / "web"
LOCAL_MODEL_DIR = BASE_DIR / "models" / "sd-turbo"

# Local model dir if present, else the ModelScope repo id (modelscope
# snapshot_download can fetch it, or any HF-compatible source).
if LOCAL_MODEL_DIR.exists():
    IMAGE_MODEL = os.environ.get("VTAI_IMAGE_MODEL", str(LOCAL_MODEL_DIR))
else:
    IMAGE_MODEL = os.environ.get("VTAI_IMAGE_MODEL", "AI-ModelScope/sd-turbo")

# Generation defaults
DEFAULT_SIZE = int(os.environ.get("VTAI_SIZE", 512))       # square output, multiple of 8
DEFAULT_FPS = int(os.environ.get("VTAI_FPS", 24))
DEFAULT_KEYFRAMES = int(os.environ.get("VTAI_KEYFRAMES", 6))
DEFAULT_STEPS = int(os.environ.get("VTAI_STEPS", 1))       # SD-Turbo needs only 1 step
SEGMENT_SECONDS = float(os.environ.get("VTAI_SEG_SECONDS", 2.0))
FADE_SECONDS = float(os.environ.get("VTAI_FADE_SECONDS", 0.5))
ZOOM_AMOUNT = float(os.environ.get("VTAI_ZOOM", 0.10))     # 10% Ken Burns zoom per segment

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
