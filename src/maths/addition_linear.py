#!/usr/bin/env python3
"""
Addition-only linear transformation using L1 distance approximation
"""

import torch
import torch.nn as nn

class AdditionLinear(nn.Module):
    """
    Addition-only linear transformation using L1 distance approximation
    Instead of w·x, compute -||w-x||₁ + bias_term
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight patterns (learnable templates)
        self.weight_patterns = nn.Parameter(torch.randn(out_features, in_features))
        
        # Sign-based learning rates
        self.learning_signs = nn.Parameter(torch.ones(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters for addition-based learning"""
        # Initialize weights to small random values
        nn.init.uniform_(self.weight_patterns, -0.1, 0.1)
        nn.init.uniform_(self.learning_signs, -1.0, 1.0)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Addition-only forward pass using L1 distance
        output[i] = -sum(|weight_patterns[i] - input|) + bias[i]
        """
        # Ensure input is on the same device as parameters
        input = input.to(self.weight_patterns.device)
        
        batch_size = input.size(0)
        input_expanded = input.unsqueeze(1)  # (batch, 1, in_features)
        weight_expanded = self.weight_patterns.unsqueeze(0)  # (1, out_features, in_features)
        
        # Compute L1 distances (using only addition/subtraction)
        differences = input_expanded - weight_expanded  # (batch, out_features, in_features)
        abs_differences = torch.abs(differences)
        
        # Sum differences (pure addition)
        l1_distances = torch.sum(abs_differences, dim=2)  # (batch, out_features)
        
        # Negative distance as similarity measure
        output = -l1_distances
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
