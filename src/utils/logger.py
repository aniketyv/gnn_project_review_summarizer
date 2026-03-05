# Made in order to track our models performance while training and while failures
# It writes these notes in two places:

# On the screen (so you can see what's happening in real time).
# In a file saved inside a folder named logs (so you can review them later).

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "results/logs") -> logging.Logger:
    """
    Single logger used across entire project.
    Writes to both console and timestamped log file.
    
    Usage:
        from src.utils.logger import get_logger
        logger = get_logger("loader")
        logger.info("Loading data...")
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Format
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console.setFormatter(fmt)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger