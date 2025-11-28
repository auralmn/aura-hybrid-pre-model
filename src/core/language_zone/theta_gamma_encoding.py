"""
Theta-Gamma Positional Encoding (JIT Optimized)

Biological positional encoding using theta-gamma phase coupling.
Optimized with TorchScript for kernel fusion.
"""

import torch
import torch.nn as nn
import math

class ThetaGammaPositionalEncoding(nn.Module):
    """
    Biological positional encoding using theta-gamma phase coupling.
    
    Optimizations:
    - JIT-compiled forward pass for kernel fusion.
    - Reduced memory bandwidth usage.
    """
    
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.theta_freq = float(config.theta_frequency)
        self.gamma_freq = float(config.gamma_frequency)
        
        # Learnable phase offsets
        self.theta_phase_offsets = nn.Parameter(
            torch.randn(config.embedding_dim) * 0.1
        )
        self.gamma_phase_offsets = nn.Parameter(
            torch.randn(config.embedding_dim) * 0.1
        )
        
        # Store max_seq_len for stable encoding during generation
        self.max_seq_len = getattr(config, 'max_seq_len', 512)
        
        # Amplitude modulation
        self.amplitude_modulation = nn.Parameter(
            torch.ones(config.embedding_dim)
        )
        
    def forward(self, positions: torch.Tensor, seq_length: int) -> torch.Tensor:
        """
        Generate theta-gamma positional encoding (JIT Wrapper).
        """
        # Use max_seq_len for normalization to ensure stability across varying sequence lengths
        # This prevents the encoding from "stretching" during autoregressive generation
        
        # FIX: Ensure we don't normalize by a length smaller than the current sequence
        # or the max length, to avoid phase wrapping issues.
        # But for generation stability, we essentially want a fixed coordinate system.
        stable_length = self.max_seq_len
        
        return self._compute_encoding(
            positions, 
            stable_length,
            self.theta_freq,
            self.gamma_freq,
            self.theta_phase_offsets,
            self.gamma_phase_offsets,
            self.amplitude_modulation
        )

    @staticmethod
    @torch.jit.script
    def _compute_encoding(positions: torch.Tensor, 
                          seq_length: int,
                          theta_freq: float, 
                          gamma_freq: float,
                          theta_offsets: torch.Tensor, 
                          gamma_offsets: torch.Tensor,
                          amp_mod: torch.Tensor) -> torch.Tensor:
        """
        JIT-compiled mathematical core.
        Fuses element-wise ops into a single kernel.
        """
        # Normalize positions to [0, 2π]
        # max(seq_length - 1, 1) ensures no division by zero
        denom = float(max(seq_length - 1, 1))
        normalized_pos = (positions.float() / denom) * (2.0 * math.pi)
        
        # Expand: [batch, seq_len, 1]
        normalized_pos = normalized_pos.unsqueeze(-1)
        
        # === Theta Phase ===
        theta_phases = normalized_pos + theta_offsets
        theta_encoding = torch.sin(theta_phases)
        
        # === Gamma Phase ===
        # Ratio calculation done inside kernel
        freq_ratio = gamma_freq / theta_freq
        gamma_phases = (normalized_pos * freq_ratio) + gamma_offsets
        
        # === Phase-Amplitude Coupling (PAC) ===
        # PAC: Gamma amplitude depends on Theta phase (cos)
        gamma_amplitude = (torch.cos(theta_phases) + 1.0) * 0.5
        gamma_encoding = gamma_amplitude * torch.sin(gamma_phases)
        
        # === Combine ===
        return (theta_encoding + 0.5 * gamma_encoding) * amp_mod
    
    def extra_repr(self) -> str:
        return (f'embedding_dim={self.embedding_dim}, '
                f'theta_freq={self.theta_freq:.1f}Hz, '
                f'gamma_freq={self.gamma_freq:.1f}Hz')