#!/usr/bin/env python3
"""
Addition-only receptance gating mechanism
"""

import torch
import torch.nn as nn

class AdditiveReceptance(nn.Module):
    """
    Addition-only receptance gating mechanism
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Pattern matching for receptance
        self.receptance_patterns = nn.Parameter(torch.randn(d_ff, d_model))
        self.sigmoid_threshold = nn.Parameter(torch.zeros(d_ff))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.receptance_patterns, -0.1, 0.1)
        nn.init.zeros_(self.sigmoid_threshold)
    
    def to(self, device):
        """Override to method to ensure all parameters are moved to device"""
        super().to(device)
        self.receptance_patterns = self.receptance_patterns.to(device)
        self.sigmoid_threshold = self.sigmoid_threshold.to(device)
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Addition-only sigmoid approximation for receptance
        """
        # Ensure input is on the same device as parameters
        x = x.to(self.receptance_patterns.device)
        
        # L1 distance to patterns
        x_expanded = x.unsqueeze(1)  # (batch, 1, d_model)
        patterns_expanded = self.receptance_patterns.unsqueeze(0)  # (1, d_ff, d_model)
        
        l1_distances = torch.sum(torch.abs(x_expanded - patterns_expanded), dim=2)
        
        # Addition-only sigmoid approximation
        # sigmoid(x) ≈ 0.5 + 0.25*x for small x, clipped to [0,1]
        normalized_distances = -l1_distances + self.sigmoid_threshold
        sigmoid_approx = 0.5 + 0.25 * normalized_distances
        sigmoid_approx = torch.clamp(sigmoid_approx, 0.0, 1.0)
        
        return sigmoid_approx
