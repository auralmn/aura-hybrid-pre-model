#!/usr/bin/env python3
"""
Addition-only linear transformation using L1 distance approximation.
Optimized for memory efficiency (chunked execution).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditionLinear(nn.Module):
    """
    Addition-only linear transformation using L1 distance approximation.
    Computes -||w-x||₁ + bias using chunking to save memory.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight patterns (learnable templates)
        self.weight_patterns = nn.Parameter(torch.randn(out_features, in_features))
        
        # Sign-based learning rates (optional modulation)
        self.learning_signs = nn.Parameter(torch.ones(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.uniform_(self.weight_patterns, -0.1, 0.1)
        nn.init.uniform_(self.learning_signs, -1.0, 1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient forward pass.
        Splits calculation into chunks to avoid O(Batch * Out * In) memory usage.
        """
        # Ensure input is 2D [Batch, In]
        if input.dim() == 3:
            B, T, D = input.shape
            input = input.view(B*T, D)
            is_seq = True
        else:
            is_seq = False
            
        batch_size = input.size(0)
        
        # Heuristic: Chunk size to fit in L2 cache / avoid VRAM spill
        # Example: Process 1024 output neurons at a time
        chunk_size = 1024
        output_list = []
        
        # We iterate over output features (rows of weight matrix)
        for i in range(0, self.out_features, chunk_size):
            end = min(i + chunk_size, self.out_features)
            
            # Slice weights: [Chunk, In]
            w_chunk = self.weight_patterns[i:end]
            
            # Expand for broadcast: 
            # Input: [Batch, 1, In]
            # Weights: [1, Chunk, In]
            # Result: [Batch, Chunk, In] -> This is still large but manageable per chunk
            
            # Compute |x - w|
            diff = torch.abs(input.unsqueeze(1) - w_chunk.unsqueeze(0))
            
            # Sum over input dim -> [Batch, Chunk]
            dist = torch.sum(diff, dim=2)
            
            output_list.append(-dist)
            
        # Concatenate chunks: [Batch, Out]
        output = torch.cat(output_list, dim=1)
        
        if self.bias is not None:
            output = output + self.bias
            
        if is_seq:
            output = output.view(B, T, self.out_features)
            
        return output