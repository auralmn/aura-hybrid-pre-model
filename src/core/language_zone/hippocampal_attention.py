import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class HippocampalProsodyAttention(nn.Module):
    def __init__(self, config, hippocampus):
        super().__init__()
        self.config = config
        self.hippocampus = hippocampus
        
        self.hidden_size = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Standard projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Gates
        self.prosody_gate = nn.Linear(4, self.num_heads)
        self.memory_gate = nn.Linear(self.hidden_size, 1)
        
        self.dropout = config.dropout

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        prosody: Optional[torch.Tensor] = None,
        use_memory: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Project Q, K, V
        # [B, L, D] -> [B, L, H, D_head] -> [B, H, L, D_head]
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Fuse Modulations into Query
        
        # A. Prosody Modulation
        if prosody is not None:
            # [B, L, 4] -> [B, L, H] -> [B, H, L, 1]
            prosody_gain = torch.sigmoid(self.prosody_gate(prosody))
            prosody_gain = prosody_gain.transpose(1, 2).unsqueeze(-1)
            query = query * (1.0 + prosody_gain)
            
        # B. Hippocampal Memory Integration (Fix applied here)
        if use_memory and self.hippocampus is not None:
            # [B, L, 1] -> [B, 1, L, 1]
            memory_weight = torch.sigmoid(self.memory_gate(hidden_states))
            # Correct reshaping for broadcasting against [B, H, L, D_head]
            # memory_weight is [B, L, 1]
            # transpose(1, 2) makes it [B, 1, L]
            # unsqueeze(1) makes it [B, 1, 1, L] ? No, wait.
            
            # Target shape: [B, H, L, D_head]
            # memory_weight: [B, L, 1]
            
            # We want to broadcast across Heads (H) and Head Dim (D_head)
            # So we need shape [B, 1, L, 1]
            
            mem_scale = 1.0 + (memory_weight.transpose(1, 2).unsqueeze(-1) * 0.5) 
            # memory_weight: [B, L, 1] -> transpose(1,2) -> [B, 1, L] -> unsqueeze(-1) -> [B, 1, L, 1]
            
            query = query * mem_scale

        # 3. Flash Attention 2
        context = F.scaled_dot_product_attention(
            query, key, value, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        # 4. Output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(context)
        
        return output, None