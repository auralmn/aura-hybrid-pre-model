"""
The Basal Ganglia: Action Selection and Integration.

Aggregates outputs from various cortical regions and produces the final system output.
Acts as a 'Gating' mechanism for thought/action.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

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
        
    def forward(self, cortical_outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Integrate cortical outputs.
        
        Args:
            cortical_outputs: Dict mapping region_name -> Output Tensor
            
        Returns:
            final_output: [Batch, Seq, Dim] or None if no outputs
        """
        if not cortical_outputs:
            return None
            
        integrated_signal = None
        total_weight = 0.0
        
        for name, output in cortical_outputs.items():
            if name in self.region_gates:
                weight = torch.sigmoid(self.region_gates[name])
                weighted_output = output * weight
                if integrated_signal is None:
                    integrated_signal = weighted_output
                else:
                    integrated_signal = integrated_signal + weighted_output
                total_weight = total_weight + weight
        
        if integrated_signal is None:
            return None
            
        # Normalize by total weight
        integrated_signal = integrated_signal / (total_weight + 1e-6)
            
        return self.integration(integrated_signal)