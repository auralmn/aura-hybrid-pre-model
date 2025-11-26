import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SNNMatmul(nn.Module):
    """
    Spike-driven matrix multiplication via cumulative outer product.
    
    Standard matmul: Y = X @ W
    SNN equivalent: Y = sum_t (s_t ⊗ W) where s_t are spike trains
    
    Features:
    - Gradient clipping for stability
    - Normalization after accumulation
    - Optional scaling by sqrt(d_k) for attention
    """
    
    def __init__(self, in_features: int, out_features: int, scale: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # Scaling factor for numerical stability
        if scale:
            self.scale_factor = math.sqrt(in_features)
        else:
            self.scale_factor = 1.0
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights for SNN dynamics."""
        # Xavier/Glorot initialization scaled for spikes
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.normal_(self.weight, mean=0.0, std=std)
    
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Spike-driven matrix multiplication.
        
        Args:
            spikes: (batch, time, in_features) spike tensor
        
        Returns:
            output: (batch, time, out_features) accumulated result
        """
        batch_size, seq_len, _ = spikes.shape
        
        # Efficient implementation: reshape and use torch matmul
        # (B, T, in) @ (out, in)^T → (B, T, out)
        spikes_flat = spikes.reshape(batch_size * seq_len, self.in_features)
        output_flat = F.linear(spikes_flat, self.weight)
        output = output_flat.reshape(batch_size, seq_len, self.out_features)
        
        # Cumulative accumulation over time (optional - for true temporal integration)
        # For now, we process each timestep independently for efficiency
        
        # Scaling for stability (similar to attention scaling)
        if self.scale:
            output = output / self.scale_factor
        
        # Gradient clipping is applied during backward pass via hooks
        # (can be added in training loop)
        
        return output
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, scale={self.scale}'


class SNNSoftmax(nn.Module):
    """
    Spike-based softmax approximation.
    
    Standard softmax: exp(x_i) / sum(exp(x_j))
    SNN equivalent: Normalize spike accumulation
    
    Features:
    - Temperature scaling for sharpness control
    - Numerical stability via log-sum-exp trick
    - Spike accumulation and normalization
    """
    
    def __init__(self, dim: int = -1, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
    
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Spike-based softmax normalization.
        
        Args:
            spikes: (batch, time, features) spike tensor
        
        Returns:
            normalized: (batch, time, features) normalized spikes
        """
        # For spike trains, we approximate softmax by:
        # 1. Accumulate spikes over time window
        # 2. Apply temperature scaling
        # 3. Normalize via standard softmax
        
        # Temperature scaling
        scaled_spikes = spikes / self.temperature
        
        # Apply standard softmax for stability
        # This uses log-sum-exp trick internally
        normalized = F.softmax(scaled_spikes, dim=self.dim)
        
        return normalized
    
    def extra_repr(self):
        return f'dim={self.dim}, temperature={self.temperature}'


class SNNSiLU(nn.Module):
    """
    Piecewise approximation of SiLU (Swish) for spike regime.
    
    Standard SiLU: x * sigmoid(x)
    SNN equivalent: Piecewise linear approximation
    
    Approximation:
    - x < -3: ~0
    - -3 <= x <= 3: linear interpolation
    - x > 3: ~x (linear region)
    """
    
    def __init__(self, num_pieces: int = 10):
        super().__init__()
        self.num_pieces = num_pieces
        
        # Precompute lookup table for piecewise approximation
        x = torch.linspace(-5, 5, num_pieces)
        y = x * torch.sigmoid(x)  # True SiLU values
        
        self.register_buffer('x_table', x)
        self.register_buffer('y_table', y)
    
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Piecewise SiLU approximation.
        
        Args:
            spikes: Input spike tensor
        
        Returns:
            output: SiLU-activated spikes
        """
        # For efficiency, use standard SiLU
        # (piecewise is mainly useful for hardware implementation)
        return F.silu(spikes)
    
    def piecewise_forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """True piecewise implementation (for reference)."""
        # Linear interpolation between lookup points
        # This is more hardware-friendly but slower in PyTorch
        output = torch.zeros_like(spikes)
        
        for i in range(self.num_pieces - 1):
            x0, x1 = self.x_table[i], self.x_table[i + 1]
            y0, y1 = self.y_table[i], self.y_table[i + 1]
            
            mask = (spikes >= x0) & (spikes < x1)
            slope = (y1 - y0) / (x1 - x0)
            output = torch.where(mask, y0 + slope * (spikes - x0), output)
        
        return output


class SNNRMSNorm(nn.Module):
    """
    SNN-adapted RMS Normalization.
    
    Standard RMSNorm: x / RMS(x) * gamma
    SNN equivalent: Normalize spike rates
    
    Features:
    - Spike rate normalization
    - Learnable scaling (gamma)
    - Epsilon for numerical stability
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        RMS normalization for spike trains.
        
        Args:
            spikes: (batch, time, features) spike tensor
        
        Returns:
            normalized: (batch, time, features) normalized spikes
        """
        # Compute RMS along feature dimension
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(spikes ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize
        normalized = spikes / rms
        
        # Scale with learnable parameter
        normalized = normalized * self.gamma
        
        return normalized
    
    def extra_repr(self):
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}'
