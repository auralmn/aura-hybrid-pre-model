import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List

from src.core.language_zone.gif_neuron import GIFNeuron
from src.core.language_zone.synapsis import Synapsis


class SNNExpert(nn.Module):
    """
    SNN-based expert for Liquid MoE integration.
    
    Wraps GIF + Synapsis layers to provide expert functionality
    compatible with Liquid MoE router interface.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        num_layers: int = 2,
        L: int = 16
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build SNN layers
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            
            # Synaptic layer
            layers.append(Synapsis(in_d, hidden_dim))
            
            # Spiking neuron layer
            layers.append(GIFNeuron(hidden_dim, hidden_dim, L=L))
        
        self.layers = nn.ModuleList(layers)
        
        # Output readout (non-spiking)
        self.readout = nn.Linear(hidden_dim, output_dim)
        
        # State management
        self.reset_state()
    
    def reset_state(self):
        """Reset all internal states."""
        self.layer_states = [None] * len(self.layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SNN expert.
        
        Args:
            x: (batch, time, input_dim) spike input
        
        Returns:
            output: (batch, output_dim) continuous output
        """
        h = x
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Synapsis, GIFNeuron)):
                h, self.layer_states[i] = layer(h, state=self.layer_states[i])
        
        # Readout: average over time, then linear projection
        h_avg = h.mean(dim=1)  # (batch, hidden_dim)
        output = self.readout(h_avg)  # (batch, output_dim)
        
        return output
    
    def predict(self, x: np.ndarray) -> float:
        """
        Single-sample prediction (MoE router interface).
        
        Args:
            x: (input_dim,) numpy array
        
        Returns:
            prediction: scalar float
        """
        # Convert to torch tensor with batch and time dimensions
        x_torch = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
        
        with torch.no_grad():
            output = self.forward(x_torch)  # (1, output_dim)
        
        return float(output.squeeze().cpu().numpy())
    
    def update(
        self,
        x: np.ndarray,
        target: float,
        token_ids: List[int],
        attention_bundle: Optional[Dict[str, Any]] = None
    ):
        """
        Update expert weights (MoE router interface).
        
        This is a placeholder for online learning.
        In practice, you'd accumulate gradients here and apply them
        during a periodic training step.
        """
        # For now, just log the update request
        # In production, you'd buffer (x, target) pairs and run gradient descent
        pass
    
    def state_dict(self) -> Dict:
        """Get expert state (MoE router interface)."""
        return {
            'model': super().state_dict(),
            'layer_states': self.layer_states
        }
    
    def load_state_dict(self, state: Dict):
        """Load expert state (MoE router interface)."""
        super().load_state_dict(state['model'])
        self.layer_states = state.get('layer_states', [None] * len(self.layers))
