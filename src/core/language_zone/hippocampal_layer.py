"""
Hippocampal Transformer Layer

Standard Transformer Encoder Layer augmented with:
1. HippocampalProsodyAttention (Prosody + Memory)
2. Feed-Forward Network
3. Layer Normalization & Residuals
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.core.language_zone.hippocampal_attention import HippocampalProsodyAttention

class HippocampalTransformerLayer(nn.Module):
    """
    Single layer of the Hippocampal Transformer.
    
    Structure:
    Input -> LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    """
    
    def __init__(self, config, hippocampus):
        super().__init__()
        self.config = config
        
        # 1. Attention Sublayer
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = HippocampalProsodyAttention(config, hippocampus)
        
        # 2. Feed-Forward Sublayer
        self.ffn_norm = nn.LayerNorm(config.embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embedding_dim, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.embedding_dim),
            nn.Dropout(config.dropout)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        prosody: Optional[torch.Tensor] = None,
        use_memory: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: [batch, seq_len, embedding_dim]
            prosody: [batch, seq_len, 4]
            use_memory: Whether to use hippocampal memory
        """
        # 1. Attention Block (Pre-Norm)
        normed_hidden = self.attention_norm(hidden_states)
        attn_output, _ = self.attention(
            normed_hidden, 
            prosody=prosody, 
            use_memory=use_memory
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # 2. Feed-Forward Block (Pre-Norm)
        normed_hidden = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(normed_hidden)
        hidden_states = hidden_states + ffn_output
        
        return hidden_states
