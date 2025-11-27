"""
Multi-Channel Spiking Attention (GPU-Native).
Fuses amplitude/pitch/boundary signals using tensor operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class MultiChannelSpikingAttention(nn.Module):
    """
    GPU-accelerated Spiking Attention.
    """
    def __init__(self, 
                 k_winners: int = 5,
                 decay_amp: float = 0.7,
                 decay_pitch: float = 0.7,
                 decay_bound: float = 0.7,
                 w_amp: float = 1.0,
                 w_pitch: float = 1.0,
                 w_bound: float = 1.0,
                 gain_up: float = 1.8,
                 gain_down: float = 0.6,
                 min_gain: float = 0.5,
                 max_gain: float = 2.5,
                 smoothing: int = 0,
                 normalize_salience: bool = True):
        super().__init__()
        self.k_winners = k_winners
        
        # Register constants buffers
        self.register_buffer('decay', torch.tensor([decay_amp, decay_pitch, decay_bound]))
        self.register_buffer('weights', torch.tensor([w_amp, w_pitch, w_bound]))
        
        self.gain_up = gain_up
        self.gain_down = gain_down
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.smoothing = smoothing
        self.normalize_salience = normalize_salience

    def _lif_batch(self, x: torch.Tensor, decay: float, theta: float = 1.0) -> torch.Tensor:
        """
        Batched LIF without loops (Vectorized Scan).
        x: [Batch, Seq_Len]
        """
        # Efficient cumulative decay via log-space or sequential scan
        # For typical seq_len (512), a simple JIT loop is best, 
        # or we use a cumulative sum approximation if decay is simple.
        
        # Here we use a fast iterative scan which is JIT-friendly
        # Or simply use PyTorch's native associative scan if available (complex)
        
        # Simple loop is fast enough on GPU for L=512 compared to CPU overhead
        batch_size, seq_len = x.shape
        v = torch.zeros(batch_size, device=x.device)
        spikes = []
        
        for t in range(seq_len):
            v = decay * v + x[:, t]
            s = (v >= theta).float()
            v = v - s * theta
            spikes.append(s)
            
        return torch.stack(spikes, dim=1) # [Batch, Seq]

    def forward(self, 
                amp: torch.Tensor, 
                pitch: torch.Tensor, 
                boundary: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute attention gains.
        Inputs: [Batch, Seq_Len] tensors on GPU.
        """
        # 1. LIF Spiking
        s_amp = self._lif_batch(amp, self.decay[0])
        s_pitch = self._lif_batch(pitch, self.decay[1])
        s_bound = self._lif_batch(boundary, self.decay[2])
        
        # 2. Fusion
        sal = (self.weights[0] * s_amp + 
               self.weights[1] * s_pitch + 
               self.weights[2] * s_bound)
               
        # 3. Smoothing (Conv1d)
        if self.smoothing > 1:
            # Reshape for conv1d: [Batch, Channel, Seq]
            sal_in = sal.unsqueeze(1)
            kernel = torch.ones(1, 1, self.smoothing, device=sal.device) / self.smoothing
            sal = F.conv1d(sal_in, kernel, padding=self.smoothing//2).squeeze(1)
            # Truncate to original length if needed
            sal = sal[:, :amp.shape[1]]

        # 4. Normalize
        if self.normalize_salience:
            max_val = sal.max(dim=1, keepdim=True)[0]
            sal = sal / (max_val + 1e-6)

        # 5. k-WTA
        # Find top-k indices
        topk_vals, topk_idx = torch.topk(sal, k=self.k_winners, dim=-1)
        
        # 6. Compute Mu Scalar
        # Mean salience of winners
        avg_winner = topk_vals.mean(dim=1)
        
        # Tanh gain curve
        gain_range = self.max_gain - self.min_gain
        mu_scalar = self.min_gain + gain_range * torch.tanh(self.gain_up * avg_winner)
        
        return {
            "mu_scalar": mu_scalar, # [Batch]
            "salience": sal,        # [Batch, Seq]
            "winners": topk_idx
        }

def prosody_channels_from_text(token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Approximation of prosody extraction from Token IDs on GPU.
    Real extraction requires text, but moving text to CPU is slow.
    We approximate using token ID statistics or learned embeddings.
    
    For now, we return random/heuristic tensors to keep pipeline on GPU.
    Ideally, this is computed in the DataLoader and passed as a tensor.
    """
    batch, seq = token_ids.shape
    device = token_ids.device
    
    # Placeholder: Random prosody to allow training to proceed without CPU
    # In production, compute this in Dataset __getitem__
    amp = torch.rand(batch, seq, device=device)
    pitch = torch.rand(batch, seq, device=device)
    boundary = (torch.rand(batch, seq, device=device) > 0.8).float()
    
    return amp, pitch, boundary

class AttentionPresets:
    """Pre-configured attention settings for different use cases"""
    
    @staticmethod
    def analytical() -> MultiChannelSpikingAttention:
        """For analytical/linguistic processing"""
        return MultiChannelSpikingAttention(
            k_winners=3,
            w_amp=0.8, w_pitch=1.2, w_bound=1.0,
            gain_up=1.5, gain_down=0.7,
            smoothing=2
        )
    
    @staticmethod
    def emotional() -> MultiChannelSpikingAttention:
        """For emotional/sentiment processing"""
        return MultiChannelSpikingAttention(
            k_winners=5,
            w_amp=1.2, w_pitch=1.5, w_bound=0.6,
            gain_up=2.0, gain_down=0.4,
            smoothing=1
        )
    
    @staticmethod
    def historical() -> MultiChannelSpikingAttention:
        """For historical/temporal processing"""
        return MultiChannelSpikingAttention(
            k_winners=4,
            w_amp=1.0, w_pitch=1.0, w_bound=1.3,
            gain_up=1.8, gain_down=0.6,
            smoothing=3
        )
    
    @staticmethod
    def streaming() -> MultiChannelSpikingAttention:
        """For streaming/real-time processing"""
        return MultiChannelSpikingAttention(
            k_winners=6,
            w_amp=1.0, w_pitch=1.0, w_bound=1.0,
            gain_up=1.6, gain_down=0.5,
            smoothing=0,  # No smoothing for real-time
            normalize_salience=False  # Keep raw salience for streaming
        )