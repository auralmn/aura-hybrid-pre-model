import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class MultiChannelSpikingAttention(nn.Module):
    def __init__(self, k_winners=5, decay_amp=0.7, decay_pitch=0.7, decay_bound=0.7,
                 w_amp=1.0, w_pitch=1.0, w_bound=1.0, gain_up=1.8, gain_down=0.6,
                 min_gain=0.5, max_gain=2.5, smoothing=0, normalize_salience=True):
        super().__init__()
        self.k_winners = k_winners
        self.register_buffer('decay', torch.tensor([decay_amp, decay_pitch, decay_bound]))
        self.register_buffer('weights', torch.tensor([w_amp, w_pitch, w_bound]))
        self.gain_up = gain_up
        self.gain_down = gain_down
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.smoothing = smoothing
        self.normalize_salience = normalize_salience

    def _lif_batch(self, x, decay, theta=1.0):
        batch, seq = x.shape
        v = torch.zeros(batch, device=x.device)
        spikes = []
        for t in range(seq):
            v = decay * v + x[:, t]
            s = (v >= theta).float()
            v = v - s * theta
            spikes.append(s)
        return torch.stack(spikes, dim=1)

    def forward(self, amp, pitch, boundary):
        s_amp = self._lif_batch(amp, self.decay[0])
        s_pitch = self._lif_batch(pitch, self.decay[1])
        s_bound = self._lif_batch(boundary, self.decay[2])
        
        sal = (self.weights[0] * s_amp + self.weights[1] * s_pitch + self.weights[2] * s_bound)
        
        if self.smoothing > 1:
            k = torch.ones(1, 1, self.smoothing, device=sal.device) / self.smoothing
            sal = F.conv1d(sal.unsqueeze(1), k, padding=self.smoothing//2).squeeze(1)[:, :sal.shape[1]]
            
        if self.normalize_salience:
            sal = sal / (sal.max(dim=1, keepdim=True)[0] + 1e-6)
            
        topk_vals, topk_idx = torch.topk(sal, k=self.k_winners, dim=-1)
        avg_winner = topk_vals.mean(dim=1)
        
        gain_range = self.max_gain - self.min_gain
        mu_scalar = self.min_gain + gain_range * torch.tanh(self.gain_up * avg_winner)
        
        return {"mu_scalar": mu_scalar, "salience": sal, "winners": topk_idx}

def prosody_channels_from_text(token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic prosody from token IDs."""
    # Deterministic hash-like transform
    # Token IDs are int. Float conversion + trig = pseudo-random but deterministic.
    t = token_ids.float()
    amp = torch.sin(t * 0.1).abs()
    pitch = torch.cos(t * 0.05).abs()
    boundary = (torch.sin(t * 0.3) > 0.8).float()
    return amp, pitch, boundary