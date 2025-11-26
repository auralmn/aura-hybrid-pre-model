import torch
import torch.nn as nn
from typing import Optional, Tuple

from .gif_neuron import GIFNeuron, MultiBitSurrogate


class ProsodyModulatedGIF(GIFNeuron):
    """
    GIF Neuron with prosody-driven threshold and gain modulation.
    
    Attention gains from prosody modulate:
    - Firing threshold (lower for high-salience tokens)
    - Input gain (amplify important tokens)
    - Adaptation rate (faster for emotional content)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        L: int = 16,
        dt: float = 1.0,
        tau: float = 10.0,
        threshold: float = 1.0,
        alpha: float = 0.01,
        attention_modulation_strength: float = 0.3
    ):
        super().__init__(input_dim, hidden_dim, L, dt, tau, threshold, alpha)
        
        self.attention_modulation_strength = attention_modulation_strength
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple] = None,
        attention_gains: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass with attention modulation.
        
        Args:
            x: Input tensor (batch, time, input_dim)
            state: Optional (v, theta) state tuple
            attention_gains: Optional (batch, time) attention gains from prosody
        
        Returns:
            spikes: (batch, time, hidden_dim)
            state: Updated (v, theta)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize state
        if state is None:
            v = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            theta = torch.full((batch_size, self.hidden_dim), self.threshold,
                              device=x.device, dtype=x.dtype)
        else:
            v, theta = state
        
        h = self.linear(x)
        spikes_list = []
        
        for t in range(seq_len):
            i_t = h[:, t, :]
            
            # Apply attention gain to input
            if attention_gains is not None:
                gain_t = attention_gains[:, t].unsqueeze(1)  # (batch, 1)
                i_t = i_t * gain_t
            
            # Update membrane potential
            v = v * self.decay + i_t
            
            # Modulate threshold based on attention
            theta_effective = theta
            if attention_gains is not None:
                # High attention → lower threshold (easier to spike)
                gain_t = attention_gains[:, t].unsqueeze(1)
                threshold_scale = 1.0 - self.attention_modulation_strength * (gain_t - 1.0)
                threshold_scale = torch.clamp(threshold_scale, 0.5, 1.5)
                theta_effective = theta * threshold_scale
            
            # Numerical stability
            clamp_limit = self.L * theta_effective * 2.0
            v = torch.clamp(v, -clamp_limit, clamp_limit)
            
            # Multi-bit spike generation
            normalized_v = v / theta_effective
            spike = MultiBitSurrogate.apply(normalized_v, self.L)
            
            # Soft reset
            v = v - spike * theta_effective
            
            # Threshold adaptation (modulated by attention)
            if self.alpha > 0:
                alpha_effective = self.alpha
                if attention_gains is not None:
                    # Higher attention → faster adaptation
                    alpha_effective = self.alpha * gain_t
                
                theta = theta + alpha_effective * spike - alpha_effective * (theta - self.threshold)
            
            spikes_list.append(spike)
        
        spikes = torch.stack(spikes_list, dim=1)
        
        return spikes, (v, theta)
