"""
PyTorch Softmax Implementation with Temperature.
"""

import torch
import torch.nn.functional as F

def softmax(x: torch.Tensor, temp: float = 1.0, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax with temperature scaling.
    
    Args:
        x: Input tensor.
        temp: Temperature (lower = sharper, higher = flatter).
        dim: Dimension along which to compute softmax.
        
    Returns:
        Probability distribution tensor.
    """
    # Avoid division by zero
    t = max(1e-8, temp)
    
    # Apply temperature
    if abs(t - 1.0) > 1e-6:
        x = x / t
        
    # PyTorch softmax is numerically stable (uses log-sum-exp trick internally)
    return F.softmax(x, dim=dim)