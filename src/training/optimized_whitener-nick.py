import torch
import torch.nn as nn
from typing import Dict, Any

class OptimizedWhitener(nn.Module):
    """
    GPU-Native Online Whitening.
    Computes running mean/variance on tensors without CPU sync.
    """
    def __init__(self, dim: int, eps: float = 1e-6, momentum: float = 0.01):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        # Register as buffers so they move to GPU with .to(device)
        # and save with state_dict
        self.register_buffer('mu', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for transform to support nn.Module interface"""
        return self.transform(x)
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply whitening: (x - mu) / sqrt(var + eps)
        Updates running stats if in training mode.
        """
        if self.training:
            # Calculate batch stats
            # Handle [Batch, Seq, Dim] or [Batch, Dim]
            if x.dim() > 2:
                x_flat = x.view(-1, self.dim)
            else:
                x_flat = x
                
            batch_mean = x_flat.mean(dim=0)
            batch_var = x_flat.var(dim=0, unbiased=False)
            
            # Update running stats (EMA)
            self.mu = (1 - self.momentum) * self.mu + self.momentum * batch_mean
            # Approximate running variance update
            self.var = (1 - self.momentum) * self.var + self.momentum * batch_var
            
        # Apply transformation
        # (x - mu) / sqrt(var + eps)
        return (x - self.mu) / torch.sqrt(self.var + self.eps)