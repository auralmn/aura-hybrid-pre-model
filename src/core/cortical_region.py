"""
Cortical Region (Neocortex Unit).

Represents a functional area of the cortex (e.g., Prefrontal, Temporal).
It receives input from the Thalamus and other regions, processes it via SNNs,
and outputs features.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from src.base.snn_brain_zones import NeuromorphicBrainZone, BrainZoneConfig

class CorticalRegion(nn.Module):
    """
    A specialized region of the Neocortex.
    Wraps the NeuromorphicBrainZone with cortical connectivity logic.
    """
    def __init__(self, config: BrainZoneConfig):
        super().__init__()
        self.name = config.name
        # The actual processing core (SNNs)
        self.functional_column = NeuromorphicBrainZone(config)
        
        # Layer Norm for stable inter-regional communication
        self.output_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, thalamic_input: torch.Tensor, context_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process inputs.
        
        Args:
            thalamic_input: Signal from Thalamus [Batch, Seq, Dim]
            context_input: Lateral input from other regions or memory [Batch, Seq, Dim]
        """
        # Integrate inputs
        if context_input is not None:
            # Simple additive integration (could be more complex later)
            combined_input = thalamic_input + context_input
        else:
            combined_input = thalamic_input
            
        # Run SNN dynamics
        output, stats = self.functional_column(combined_input)
        
        # Normalize for stable propagation
        output = self.output_norm(output)
        
        # We attach stats to the output tensor via a hook or return tuple
        # For now, we just return the tensor to keep the graph clean, 
        # but we could log stats to a global collector here.
        return output