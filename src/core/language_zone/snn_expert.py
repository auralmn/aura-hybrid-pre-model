import torch
import torch.nn as nn
from core.language_zone.gif_neuron import GIFNeuron
from core.language_zone.synapsis import Synapsis

class SNNExpert(nn.Module):
    """
    GPU-Native SNN Expert.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, L=16):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(Synapsis(in_d, hidden_dim))
            layers.append(GIFNeuron(hidden_dim, hidden_dim, L=L))
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(hidden_dim, output_dim)
        
        # State buffer (kept as None until first pass to avoid sizing issues)
        self.layer_states = [None] * len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Time, Dim]
        h = x
        for i, layer in enumerate(self.layers):
            # Pass state if available
            h, self.layer_states[i] = layer(h, state=self.layer_states[i])
            
        # Mean over time
        h_avg = h.mean(dim=1)
        return self.readout(h_avg)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batched prediction for MoE.
        x: [Batch, Dim] -> Unsqueeze time -> Forward -> Squeeze
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add time dim
        return self.forward(x)