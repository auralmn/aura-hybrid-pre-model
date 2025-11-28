"""
SNN-Enhanced Feed-Forward Network

Replaces standard MLP with spiking neural network for biological plausibility.
Can be used as drop-in replacement for transformer FFN.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.core.language_zone.gif_neuron import GIFNeuron
from src.core.language_zone.synapsis import Synapsis


class SNNFFN(nn.Module):
    """
    Spiking Neural Network Feed-Forward Network.
    
    Replaces standard MLP (Linear -> GELU -> Linear) with:
    Synapsis -> GIFNeuron -> Synapsis -> GIFNeuron -> Readout
    
    The spike-based computation provides:
    - Sparse activation (energy efficient)
    - Temporal dynamics (multi-bit spikes over time)
    - Biological plausibility
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_timesteps: int = 4,
        L: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.num_timesteps = num_timesteps
        
        # Layer 1: Input -> Hidden (Spiking)
        self.syn1 = Synapsis(input_dim, hidden_dim)
        self.neuron1 = GIFNeuron(hidden_dim, hidden_dim, L=L)
        
        # Layer 2: Hidden -> Output (Spiking)
        self.syn2 = Synapsis(hidden_dim, self.output_dim)
        self.neuron2 = GIFNeuron(self.output_dim, self.output_dim, L=L)
        
        # Readout: Average spikes over time
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [Batch, Seq, Dim] continuous input
            
        Returns:
            output: [Batch, Seq, output_dim] continuous output
        """
        batch_size, seq_len, _ = x.shape
        
        # Expand time dimension: [B, S, D] -> [B, S, T, D]
        # We treat each position as having T timesteps
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.num_timesteps, -1)
        x_flat = x_expanded.reshape(batch_size * seq_len, self.num_timesteps, self.input_dim)
        
        # Layer 1
        h1, _ = self.syn1(x_flat, state=None)
        spikes1, _ = self.neuron1(h1, state=None)
        
        # Layer 2
        h2, _ = self.syn2(spikes1, state=None)
        spikes2, _ = self.neuron2(h2, state=None)
        
        # Average over timesteps: [B*S, T, D] -> [B*S, D]
        output_flat = spikes2.mean(dim=1)
        
        # Reshape back: [B*S, D] -> [B, S, D]
        output = output_flat.reshape(batch_size, seq_len, self.output_dim)
        
        return self.dropout(output)


class HybridFFN(nn.Module):
    """
    Hybrid FFN that combines standard MLP with SNN pathway.
    
    Uses a gating mechanism to blend continuous and spiking computations.
    This allows gradual transition from pure transformer to neuromorphic.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        snn_ratio: float = 0.5,
        num_timesteps: int = 4,
        L: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.snn_ratio = snn_ratio
        
        # Standard MLP pathway
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
        
        # SNN pathway
        self.snn = SNNFFN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_timesteps=num_timesteps,
            L=L,
            dropout=dropout
        )
        
        # Learnable gate (can be fixed or learned)
        self.gate = nn.Parameter(torch.tensor(snn_ratio))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated blend.
        
        Args:
            x: [Batch, Seq, Dim]
            
        Returns:
            output: [Batch, Seq, Dim]
        """
        mlp_out = self.mlp(x)
        snn_out = self.snn(x)
        
        # Blend with sigmoid gate
        g = torch.sigmoid(self.gate)
        return (1 - g) * mlp_out + g * snn_out

