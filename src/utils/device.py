# Every team member owns different machines

# 1) Macbook (Aniket)
# 2) Windows (Madhvi and Shahbaz)

import torch
from src.utils.logger import get_logger

logger = get_logger("device")


def get_device(config: dict) -> torch.device:
    """
    Detects and returns best available device.
    Priority: MPS (Apple) → CUDA (NVIDIA) → CPU
    
    Usage:
        from src.utils.device import get_device
        device = get_device(config)
    """
    device_config = config.get("device", {})

    # Apple Silicon MPS
    if device_config.get("use_mps", True):
        if torch.backends.mps.is_available():
            logger.info("Device: Apple MPS (M-series GPU)")
            return torch.device("mps")
        else:
            logger.warning("MPS requested but not available")

    # NVIDIA CUDA
    if device_config.get("use_cuda", False):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Device: CUDA ({gpu_name})")
            return torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available")

    # CPU fallback
    logger.info("Device: CPU")
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    Returns full device information.
    Useful for logging at start of training.
    """
    info = {
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )

    return info