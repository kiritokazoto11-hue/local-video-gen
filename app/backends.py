"""Device / backend detection.

Priority:
  1. NVIDIA CUDA  (works automatically if torch with CUDA is installed)
  2. CPU          (always available; what this machine uses — Intel iGPU has no CUDA)
"""
import platform

import torch


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_info() -> dict:
    info = {
        "device": detect_device(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
