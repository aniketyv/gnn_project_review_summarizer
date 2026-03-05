# To ensure that we get constant results on all our machines as far as possible

import torch
import numpy as np
import random
from src.utils.logger import get_logger

logger = get_logger("seed")


def set_seed(seed: int = 42):
    """
    Sets random seed across all libraries for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")