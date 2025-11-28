import torch
import torch.nn as nn

class SpikeToContinuousBridge(nn.Module):
    """
    GPU-Native Spike Bridge.
    """
    def __init__(self, spike_dim, output_dim, encoding='rate', time_window=10):
        super().__init__()
        self.encoding = encoding
        self.time_window = time_window
        self.proj = nn.Linear(spike_dim, output_dim) if spike_dim != output_dim else nn.Identity()
        
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        spikes: [Batch, Time, Dim]
        Returns: [Batch, Out_Dim]
        """
        # Rate coding: Mean over window
        if self.encoding == 'rate':
            recent = spikes[:, -self.time_window:, :]
            feats = recent.mean(dim=1)
            
        elif self.encoding == 'temporal':
            T = spikes.shape[1]
            # [1, T, 1]
            weights = torch.exp(torch.arange(T, device=spikes.device).float() / self.time_window)
            weights = weights.view(1, -1, 1)
            feats = (spikes * weights).sum(dim=1) / (weights.sum() + 1e-6)
            
        elif self.encoding == 'phase':
            # FFT on GPU
            recent = spikes[:, -self.time_window:, :]
            fft = torch.fft.rfft(recent, dim=1)
            feats = torch.abs(fft).mean(dim=1)
            
        else:
            feats = spikes.mean(dim=1)
            
        return self.proj(feats)

class ContinuousToSpikeBridge(nn.Module):
    """
    GPU-Native Continuous -> Spike.
    """
    def __init__(self, input_dim, spike_dim, encoding='poisson', num_timesteps=10):
        super().__init__()
        self.encoding = encoding
        self.num_timesteps = num_timesteps
        self.proj = nn.Linear(input_dim, spike_dim) if input_dim != spike_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, In_Dim]
        Returns: [Batch, Time, Spike_Dim]
        """
        feat = self.proj(x)
        batch, dim = feat.shape
        dtype = x.dtype
        
        if self.encoding == 'poisson':
            rates = torch.sigmoid(feat).unsqueeze(1)
            rand = torch.rand(batch, self.num_timesteps, dim, device=x.device, dtype=dtype)
            return (rand < rates).to(dtype)
            
        elif self.encoding == 'temporal':
            norm = torch.sigmoid(feat) * self.num_timesteps
            time_idx = torch.arange(self.num_timesteps, device=x.device, dtype=dtype).view(1, -1, 1)
            return (norm.unsqueeze(1) > time_idx).to(dtype)
            
        return torch.zeros(batch, self.num_timesteps, dim, device=x.device, dtype=dtype)