import numpy as np

def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus"""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)