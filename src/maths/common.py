import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid"""
    return 1 / (1 + np.exp(-x))


