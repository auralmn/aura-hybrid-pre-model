"""
PyTorch Softplus Implementation.
Wraps torch.nn.functional.softplus for compatibility.
"""

import torch
import torch.nn.functional as F

def softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable softplus function.
    f(x) = ln(1 + exp(x))
    
    Args:
        x: Input tensor on any device.
    Returns:
        Tensor on the same device.
    """
    return F.softplus(x)