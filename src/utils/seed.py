"""
Random seed utilities for reproducibility.
"""

import random
import numpy as np

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng(seed: int = None) -> np.random.RandomState:
    """
    Get a numpy RandomState for reproducible random operations.
    
    Args:
        seed: Optional seed (uses current time if None)
        
    Returns:
        numpy RandomState instance
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))
    return np.random.RandomState(seed)
