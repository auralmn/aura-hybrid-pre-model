import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class SpikeToContinuousBridge(nn.Module):
    """
    Converts spike trains to continuous representations for MoE routing.
    
    Methods:
    - Rate coding: Average spike rate over time window
    - Temporal coding: Weighted recent spikes
    - Phase coding: FFT-based frequency features
    """
    
    def __init__(
        self,
        spike_dim: int,
        output_dim: int,
        encoding: str = 'rate',  # 'rate', 'temporal', 'phase'
        time_window: int = 10
    ):
        super().__init__()
        
        self.spike_dim = spike_dim
        self.output_dim = output_dim
        self.encoding = encoding
        self.time_window = time_window
        
        # Learnable projection for dimensionality matching
        if spike_dim != output_dim:
            self.projection = nn.Linear(spike_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, spikes: torch.Tensor) -> np.ndarray:
        """
        Convert spikes to continuous features.
        
        Args:
            spikes: (batch, time, spike_dim) multi-bit spike tensor
        
        Returns:
            features: (batch, output_dim) numpy array for MoE routing
        """
        batch_size, seq_len, _ = spikes.shape
        
        if self.encoding == 'rate':
            # Rate coding: Average spike count over recent window
            window = min(self.time_window, seq_len)
            recent_spikes = spikes[:, -window:, :]  # Last N timesteps
            features = recent_spikes.mean(dim=1)  # (batch, spike_dim)
            
        elif self.encoding == 'temporal':
            # Temporal coding: Exponentially weighted recent spikes
            weights = torch.exp(-torch.arange(seq_len, 0, -1, dtype=spikes.dtype, device=spikes.device) / self.time_window)
            weights = weights.view(1, -1, 1)  # (1, time, 1)
            features = (spikes * weights).sum(dim=1)  # (batch, spike_dim)
            features = features / weights.sum()  # Normalize
            
        elif self.encoding == 'phase':
            # Phase coding: Frequency domain features
            # Use last time_window for FFT
            window = min(self.time_window, seq_len)
            recent = spikes[:, -window:, :]  # (batch, window, spike_dim)
            
            # Apply FFT along time dimension
            fft = torch.fft.rfft(recent, dim=1)  # (batch, freq, spike_dim)
            magnitude = torch.abs(fft)
            
            # Average across frequency bins
            features = magnitude.mean(dim=1)  # (batch, spike_dim)
        
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
        
        # Project to output dimension
        features = self.projection(features)  # (batch, output_dim)
        
        # Convert to numpy for MoE router
        features_np = features.detach().cpu().numpy()
        
        return features_np


class ContinuousToSpikeBridge(nn.Module):
    """
    Converts continuous expert outputs back to spike trains.
    
    Methods:
    - Poisson encoding: Spike probability ~ output magnitude
    - Threshold encoding: Spike when output exceeds threshold
    - Temporal spread: Distribute output across timesteps
    """
    
    def __init__(
        self,
        input_dim: int,
        spike_dim: int,
        encoding: str = 'poisson',  # 'poisson', 'threshold', 'temporal'
        num_timesteps: int = 10
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.spike_dim = spike_dim
        self.encoding = encoding
        self.num_timesteps = num_timesteps
        
        # Learnable projection
        if input_dim != spike_dim:
            self.projection = nn.Linear(input_dim, spike_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, continuous: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Convert continuous features to spike trains.
        
        Args:
            continuous: (batch, input_dim) numpy array from src.cliexpert outputs
            device: torch device for output tensor
        
        Returns:
            spikes: (batch, num_timesteps, spike_dim) spike tensor
        """
        # Convert to torch
        continuous_torch = torch.from_numpy(continuous).float().to(device)
        batch_size = continuous_torch.shape[0]
        
        # Project to spike dimension
        features = self.projection(continuous_torch)  # (batch, spike_dim)
        
        if self.encoding == 'poisson':
            # Poisson encoding: spike probability = sigmoid(features)
            rates = torch.sigmoid(features)  # (batch, spike_dim) in [0, 1]
            
            # Generate spikes over time
            spikes = torch.rand(batch_size, self.num_timesteps, self.spike_dim, device=device)
            spikes = (spikes < rates.unsqueeze(1)).float()  # Binary spikes
            
        elif self.encoding == 'threshold':
            # Threshold encoding: spike when feature > threshold
            threshold = 0.0
            spike_mask = (features > threshold).float()  # (batch, spike_dim)
            
            # Spread across timesteps
            spikes = spike_mask.unsqueeze(1).expand(-1, self.num_timesteps, -1)
            
        elif self.encoding == 'temporal':
            # Temporal spread: distribute magnitude across time
            # Normalize features to [0, num_timesteps]
            features_norm = torch.sigmoid(features) * self.num_timesteps
            
            # Create spike trains with magnitude-dependent timing
            spikes = torch.zeros(batch_size, self.num_timesteps, self.spike_dim, device=device)
            
            for t in range(self.num_timesteps):
                # Spike at timestep t if features_norm > t
                spikes[:, t, :] = (features_norm > t).float()
        
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
        
        return spikes
