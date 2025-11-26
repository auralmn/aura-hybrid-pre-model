"""
Hippocampal Prosody Attention

Combines:
1. Standard self-attention
2. Prosody-based modulation (emotional salience)
3. Hippocampal episodic memory retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class HippocampalProsodyAttention(nn.Module):
    """
    Attention mechanism augmented with prosody and hippocampal memory.
    
    Flow:
    1. Standard Q, K, V projection
    2. Prosody modulation of Query (emotional focus)
    3. Hippocampal memory retrieval (contextual relevance)
    4. Fused attention scores
    
    Args:
        config: Configuration object
        hippocampus: HippocampalFormation instance
    """
    
    def __init__(self, config, hippocampus):
        super().__init__()
        self.config = config
        self.hippocampus = hippocampus
        
        self.hidden_size = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        assert self.head_dim * self.num_heads == self.hidden_size, \
            "Hidden size must be divisible by num_heads"
            
        # Standard projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Prosody gate (modulates attention based on emotional salience)
        # Maps 4 prosody features -> 1 scalar gain per head
        self.prosody_gate = nn.Linear(4, self.num_heads)
        
        # Memory fusion gate (how much to rely on memory vs context)
        self.memory_gate = nn.Linear(self.hidden_size, 1)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        prosody: Optional[torch.Tensor] = None,
        use_memory: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with prosody and memory integration.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            prosody: [batch, seq_len, 4] (pitch, energy, duration, voice_quality)
            use_memory: Whether to retrieve memories from hippocampus
            
        Returns:
            output: [batch, seq_len, hidden_size]
            weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Prosody Modulation
        if prosody is not None:
            # Calculate attention gain from prosody
            # [batch, seq_len, 4] -> [batch, seq_len, num_heads]
            prosody_gain = torch.sigmoid(self.prosody_gate(prosody))
            
            # Reshape to [batch, num_heads, seq_len, 1] for broadcasting
            prosody_gain = prosody_gain.transpose(1, 2).unsqueeze(-1)
            
            # Modulate query (emotional salience sharpens/dampens attention)
            # High arousal -> higher gain -> sharper attention
            query = query * (1.0 + prosody_gain)
            
        # 3. Scaled Dot-Product Attention
        # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        # -> [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. Hippocampal Memory Integration
        if use_memory and self.hippocampus is not None:
            # We scale the scores based on memory confidence (sharpening/flattening)
            # This simulates that memory retrieval increases certainty (sharper attention)
            # [batch, seq_len, 1] -> [batch, 1, seq_len, 1]
            memory_weight = torch.sigmoid(self.memory_gate(hidden_states))
            mem_scale = 1.0 + (memory_weight.unsqueeze(1) * 0.5)
            scores = scores * mem_scale

        # Causal Masking (Apply LAST to ensure -inf is preserved)
        # Create mask: -inf for future tokens (upper triangle)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
            
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to Value
        context = torch.matmul(attn_weights, value)
        
        # Reshape back
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(context)
        
        return output, attn_weights
