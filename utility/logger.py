import logging
import random
import numpy as np
import torch

def setup_logger():
    """
    Configures and returns a consistent logger for the SD OOP Pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("SD_OOP")
    return logger

def set_seed(seed: int):
    """
    Pins seeds across random, numpy, and torch to ensure complete reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic execution
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
