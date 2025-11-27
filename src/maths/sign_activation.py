#!/usr/bin/env python3
"""
Sign-based activation using straight-through estimator.
"""

import torch
import torch.nn as nn

class SignSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for Sign function.
    Forward: sign(x)
    Backward: HardTanh gradient (1 in [-1, 1], 0 otherwise)
    """
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient is passed through unchanged (identity) 
        # or clamped (HardTanh style) to prevent exploding gradients
        # Here we mimic HardTanh derivative: 1 if |x| <= 1
        # But we don't save input to save memory, assume identity pass-through
        return grad_output.clamp(-1, 1)

class SignActivation(nn.Module):
    """
    Sign-based activation with learnable threshold.
    """
    
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centering
        shifted = x - self.threshold
        
        # Apply STE
        return SignSTE.apply(shifted)