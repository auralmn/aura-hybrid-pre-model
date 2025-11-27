import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from src.core.language_zone.gif_neuron import GIFNeuron
from src.core.language_zone.synapsis import Synapsis

class SNNExpert(nn.Module):
    """
    GPU-Native SNN Expert (Stateless for Checkpointing).
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, L=16):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            # Synapsis
            layers.append(Synapsis(in_d, hidden_dim))
            # GIF Neuron
            layers.append(GIFNeuron(hidden_dim, hidden_dim, L=L))
            
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, states: Optional[List[Any]] = None) -> torch.Tensor:
        """
        Forward pass.
        x: [Batch, Time, Dim]
        states: List of states for each layer (optional)
        """
        h = x
        new_states = []
        
        state_idx = 0
        for layer in self.layers:
            # If state provided, use it; else None (layer will init fresh)
            s = states[state_idx] if states else None
            
            if isinstance(layer, (Synapsis, GIFNeuron)):
                h, new_s = layer(h, state=s)
                new_states.append(new_s)
                state_idx += 1
            else:
                h = layer(h)
                
        # Mean pooling over time
        h_avg = h.mean(dim=1)
        return self.readout(h_avg)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stateless prediction for MoE.
        Ensures deterministic output for Checkpointing.
        """
        # x: [Batch, Dim] -> [Batch, 1, Dim] (Add Time dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Pass states=None to force fresh initialization
        return self.forward(x, states=None)