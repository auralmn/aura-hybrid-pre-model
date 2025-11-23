#!/usr/bin/env python3
"""
Sign-based activation using only addition/subtraction
"""

import torch
import torch.nn as nn

class SignActivation(nn.Module):
    """
    Sign-based activation using only addition/subtraction
    """
    
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sign activation with threshold
        output = sign(x - threshold)
        """
        shifted = x - self.threshold
        
        # Sign function approximation for training
        if self.training:
            # Straight-through estimator
            signs = torch.sign(shifted)
            # Add gradient approximation
            grad_approx = torch.clamp(1 - torch.abs(shifted), 0, 1)
            return signs + grad_approx - grad_approx.detach()
        else:
            return torch.sign(shifted)
