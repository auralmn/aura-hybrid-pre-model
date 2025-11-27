#!/usr/bin/env python3
"""
Addition-only receptance gating mechanism.
Optimized for memory efficiency.
"""

import torch
import torch.nn as nn

class AdditiveReceptance(nn.Module):
    """
    Addition-only receptance gating mechanism.
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Addition-only sigmoid approximation with memory-efficient chunking.
        """
        # Handle sequence dimension
        original_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(B*T, D)
        
        # Chunking configuration
        chunk_size = 512
        output_list = []
        
        for i in range(0, self.d_ff, chunk_size):
            end = min(i + chunk_size, self.d_ff)
            
            # Weights: [Chunk, D]
            p_chunk = self.receptance_patterns[i:end]
            
            # L1 Distance: [Batch, 1, D] - [1, Chunk, D]
            dist = torch.sum(torch.abs(x.unsqueeze(1) - p_chunk.unsqueeze(0)), dim=2)
            
            # Bias
            thresh_chunk = self.sigmoid_threshold[i:end]
            norm_dist = -dist + thresh_chunk
            
            # Sigmoid approx
            sigmoid_approx = 0.5 + 0.25 * norm_dist
            output_list.append(torch.clamp(sigmoid_approx, 0.0, 1.0))
            
        output = torch.cat(output_list, dim=1)
        
        # Restore shape
        if len(original_shape) == 3:
            output = output.view(B, T, self.d_ff)
            
        return output