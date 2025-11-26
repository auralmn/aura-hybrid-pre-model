"""
Theta-Gamma Positional Encoding

Biological positional encoding using theta-gamma phase coupling.
Replaces sinusoidal encoding with neural oscillations observed in hippocampus.
"""

import torch
import torch.nn as nn
import numpy as np


class ThetaGammaPositionalEncoding(nn.Module):
    """
    Biological positional encoding using theta-gamma phase coupling.
    
    Instead of sinusoidal functions, this uses neural oscillations to encode position:
    - Theta oscillation (8 Hz): Slow oscillation for broad position encoding
    - Gamma oscillation (40 Hz): Fast oscillation nested within theta
    - Phase-amplitude coupling (PAC): Gamma amplitude modulated by theta phase
    
    This mimics the hippocampal theta-gamma code for spatial navigation and sequence encoding.
    
    Args:
        config: Configuration with embedding_dim, theta_frequency, gamma_frequency
        
    Example:
        >>> config = ThetaGammaConfig(embedding_dim=768, theta_frequency=8.0, gamma_frequency=40.0)
        >>> encoder = ThetaGammaPositionalEncoding(config)
        >>> positions = torch.arange(128).unsqueeze(0)  # [batch=1, seq_len=128]
        >>> pos_encoding = encoder(positions, seq_length=128)
        >>> pos_encoding.shape
        torch.Size([1, 128, 768])
    """
    
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.theta_freq = config.theta_frequency
        self.gamma_freq = config.gamma_frequency
        
        # Learnable phase offsets for each dimension
        # Allows network to learn optimal phase relationships
        self.theta_phase_offsets = nn.Parameter(
            torch.randn(config.embedding_dim) * 0.1
        )
        self.gamma_phase_offsets = nn.Parameter(
            torch.randn(config.embedding_dim) * 0.1
        )
        
        # Amplitude modulation (like place cell receptive fields)
        self.amplitude_modulation = nn.Parameter(
            torch.ones(config.embedding_dim)
        )
        
    def forward(self, positions: torch.Tensor, seq_length: int) -> torch.Tensor:
        """
        Generate theta-gamma positional encoding.
        
        Args:
            positions: [batch, seq_len] position indices (0 to seq_length-1)
            seq_length: Maximum sequence length (for normalization)
            
        Returns:
            positional_encoding: [batch, seq_len, embedding_dim]
        """
        device =positions.device
        batch_size, seq_len = positions.shape
        
        # Normalize positions to [0, 2π] to span one theta cycle
        normalized_pos = (positions.float() / max(seq_length - 1, 1)) * 2 * np.pi
        
        # Expand for broadcasting: [batch, seq_len, 1]
        normalized_pos = normalized_pos.unsqueeze(-1)
        
        # === Theta Phase Encoding (slow oscillation) ===
        # Each dimension gets different phase offset
        theta_phases = normalized_pos + self.theta_phase_offsets  # [batch, seq_len, embedding_dim]
        theta_encoding = torch.sin(theta_phases)
        
        # === Gamma Phase Encoding (fast oscillation) ===
        # Gamma frequency is ~5x theta, creating nested oscillations
        gamma_phases = (normalized_pos * self.gamma_freq / self.theta_freq
                       + self.gamma_phase_offsets)
        
        # === Phase-Amplitude Coupling (PAC) ===
        # Gamma amplitude modulated by theta phase
        # This creates the characteristic theta-gamma coupling seen in hippocampus
        gamma_amplitude = (torch.cos(theta_phases) + 1.0) / 2.0  # Range [0, 1]
        gamma_encoding = gamma_amplitude * torch.sin(gamma_phases)
        
        # === Combine Theta and Gamma ===
        # Theta provides coarse position info, gamma provides fine-grained timing
        positional_encoding = (theta_encoding + 0.5 * gamma_encoding) * self.amplitude_modulation
        
        return positional_encoding
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'embedding_dim={self.embedding_dim}, '
                f'theta_freq={self.theta_freq:.1f}Hz, '
                f'gamma_freq={self.gamma_freq:.1f}Hz, '
                f'freq_ratio={self.gamma_freq/self.theta_freq:.1f}')
