"""Download SD-Turbo from ModelScope (reachable on this network) into models/sd-turbo."""
from pathlib import Path

from modelscope import snapshot_download

local_dir = str(Path(__file__).resolve().parent.parent / "models" / "sd-turbo")

snapshot_download(
    "AI-ModelScope/sd-turbo",
    local_dir=local_dir,
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ],
)
print("MODEL DOWNLOAD OK ->", local_dir, flush=True)
