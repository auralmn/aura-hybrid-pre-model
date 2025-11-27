"""
The Basal Ganglia: Action Selection and Integration.

Aggregates outputs from various cortical regions and produces the final system output.
Acts as a 'Gating' mechanism for thought/action.
"""

import torch
import torch.nn as nn
from typing import Dict, List

class BasalGanglia(nn.Module):
    def __init__(self, d_model: int, region_names: List[str]):
        super().__init__()
        # Learnable gating weights for each region
        self.region_gates = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(1.0)) for name in region_names
        })
        
        # Final integration layer
        self.integration = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, cortical_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Integrate cortical outputs.
        
        Args:
            cortical_outputs: Dict mapping region_name -> Output Tensor
            
        Returns:
            final_output: [Batch, Seq, Dim]
        """
        if not cortical_outputs:
            return None
            
        integrated_signal = 0.0
        total_weight = 0.0
        
        for name, output in cortical_outputs.items():
            if name in self.region_gates:
                # Apply learnable gate weight
                weight = torch.sigmoid(self.region_gates[name])
                integrated_signal += output * weight
                total_weight += weight
        
        # Normalize
        if isinstance(total_weight, torch.Tensor) or total_weight > 0:
            integrated_signal = integrated_signal / (total_weight + 1e-6)
            
        return self.integration(integrated_signal)