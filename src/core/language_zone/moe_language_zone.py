"""
GPU-Native MoE Language Zone.

Optimized to remove AsyncIO/CPU bottlenecks.
Integrates Liquid MoE routing directly into the PyTorch forward pass.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from src.core.language_zone.spike_bridge import SpikeToContinuousBridge, ContinuousToSpikeBridge
from src.core.language_zone.snn_expert import SNNExpert
from src.core.language_zone.gif_neuron import GIFNeuron
from src.core.liquid_moe import LiquidMoERouter


class MoELanguageZone(nn.Module):
    """
    Language Zone with integrated Liquid MoE routing (GPU Native).
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        moe_hidden_dim: int = 64
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.moe_hidden_dim = moe_hidden_dim
        self.top_k = top_k
        self.num_experts = num_experts
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.encoder = GIFNeuron(embed_dim, hidden_dim, L=16)
        
        self.spike_to_continuous = SpikeToContinuousBridge(
            spike_dim=hidden_dim,
            output_dim=moe_hidden_dim,
            encoding='rate',
            time_window=10
        )
        
        self.experts = nn.ModuleDict({
            f'expert_{i}': SNNExpert(
                input_dim=moe_hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=moe_hidden_dim
            )
            for i in range(num_experts)
        })
        
        self.moe_router = LiquidMoERouter(
            in_dim=moe_hidden_dim,
            hidden_dim=64,
            num_experts=num_experts,
            top_k=top_k
        )
        
        self.continuous_to_spike = ContinuousToSpikeBridge(
            input_dim=moe_hidden_dim,
            spike_dim=hidden_dim,
            encoding='poisson'
        )
        
        self.decoder = GIFNeuron(hidden_dim, embed_dim, L=16)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Synchronous GPU forward pass with MoE routing.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        embeds = self.embeddings(input_ids)
        spikes_enc, _ = self.encoder(embeds) 
        
        spikes_flat = spikes_enc.reshape(batch_size * seq_len, 1, self.hidden_dim)
        continuous_flat = self.spike_to_continuous(spikes_flat)
        
        route_out = self.moe_router(continuous_flat)
        topk_indices = route_out['indices']
        topk_weights = route_out['weights']
        
        expert_outputs = torch.zeros(batch_size * seq_len, self.moe_hidden_dim, device=device, dtype=embeds.dtype)
        
        for i in range(self.num_experts):
            is_selected = (topk_indices == i)
            if not is_selected.any():
                continue
            ex_out = self.experts[f'expert_{i}'].predict(continuous_flat)
            w = (topk_weights * is_selected.float()).sum(dim=1).unsqueeze(1)
            expert_outputs += ex_out * w
            
        spikes_moe = self.continuous_to_spike(expert_outputs)
        
        # Handle variable timesteps
        num_timesteps = spikes_moe.shape[1] if spikes_moe.dim() == 3 else 1
        if spikes_moe.dim() == 2:
            spikes_moe = spikes_moe.unsqueeze(1)
        spikes_moe = spikes_moe.view(batch_size, seq_len, num_timesteps, self.hidden_dim)
        spikes_moe_avg = spikes_moe.mean(dim=2)
        
        decoded, _ = self.decoder(spikes_moe_avg)
        logits = self.output_proj(decoded)
        
        return logits, {
            'probs': route_out['probs'].view(batch_size, seq_len, -1).detach().cpu()
        }

    def setup_moe_router(self, *args, **kwargs):
        pass 
    
    def forward_async(self, *args, **kwargs):
        raise DeprecationWarning("Use .forward() instead of .forward_async() for GPU execution")