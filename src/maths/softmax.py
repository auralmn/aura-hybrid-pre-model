import numpy as np


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Numerically stable softmax"""
    x = x / max(1e-8, temp)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum() + 1e-12)