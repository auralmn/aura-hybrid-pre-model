"""
PyTorch Sigmoid Implementation.
Wraps torch.sigmoid for compatibility.
"""

import torch

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid function.
    s(x) = 1 / (1 + exp(-x))
    
    Args:
        x: Input tensor on any device.
    Returns:
        Tensor on the same device.
    """
    return torch.sigmoid(x)