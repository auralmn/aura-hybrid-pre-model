"""
Prosody Attention Bridge (GPU-Native).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .multi_channel_attention import MultiChannelSpikingAttention, prosody_channels_from_text

class ProsodyAttentionBridge(nn.Module):
    def __init__(self, k_winners: int = 5):
        super().__init__()
        self.attention = MultiChannelSpikingAttention(k_winners=k_winners)
        
    def forward(self, input_ids: torch.Tensor, token_strings=None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.
        Args:
            input_ids: [Batch, Seq_Len]
            token_strings: Ignored in GPU mode (prosody inferred or passed separately)
        """
        # 1. Extract Prosody (Approximation on GPU)
        # In a real pipeline, pass 'prosody' tensor directly to this forward method
        amp, pitch, boundary = prosody_channels_from_text(input_ids)
        
        # 2. Compute Attention
        result = self.attention(amp, pitch, boundary)
        
        # 3. Create Gain Mask
        # Base gain
        mu = result['mu_scalar'].unsqueeze(1) # [Batch, 1]
        salience = result['salience']         # [Batch, Seq]
        
        # Modulation: Gain * (1 + Salience)
        attention_gains = mu * (1.0 + salience)
        
        return attention_gains, result