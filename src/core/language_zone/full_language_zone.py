"""
GPU-Native Full Language Zone (Temporal Cortex).

Integrates:
- Prosody-driven Attention (GPU)
- JIT-Compiled SNN Encoder/Decoder
- Vectorized Liquid MoE Router

Refactored to work as a Cortical Region within the Natural Brain.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from src.core.language_zone.prosody_attention import ProsodyAttentionBridge
from src.core.language_zone.gif_neuron import GIFNeuron
from src.core.language_zone.spike_bridge import SpikeToContinuousBridge, ContinuousToSpikeBridge
from src.core.language_zone.snn_expert import SNNExpert
from src.core.liquid_moe import LiquidMoERouter
from src.base.snn_brain_zones import BrainZoneConfig


class FullLanguageZone(nn.Module):
    """
    The Language Cortex (Temporal Lobe).
    
    Receives:
    1. Thalamic Input (Gated Semantic Embeddings) -> processed by SNNs
    2. Raw Tokens (Input IDs) -> processed by Prosody Bridge for modulation
    
    Flow:
    Input -> Prosody Modulator -> SNN Encoder -> Liquid MoE -> Experts -> SNN Decoder -> Output
    """
    
    def __init__(self, config: BrainZoneConfig, vocab_size: int):
        super().__init__()
        
        self.config = config
        self.vocab_size = vocab_size
        self.embed_dim = config.d_model
        self.hidden_dim = config.max_neurons 
        self.moe_hidden_dim = 64
        self.num_experts = 8
        self.top_k = 2
        
        # 1. Prosody Attention
        self.prosody_attention = ProsodyAttentionBridge(k_winners=self.top_k)
        
        # 2. SNN Encoder (GIF)
        self.encoder = GIFNeuron(self.embed_dim, self.hidden_dim, L=16)
        
        # 3. Spike Bridge (Pre-Routing)
        self.spike_to_continuous = SpikeToContinuousBridge(
            spike_dim=self.hidden_dim,
            output_dim=self.moe_hidden_dim,
            encoding='rate'
        )
        
        # 4. SNN Experts
        self.experts = nn.ModuleDict({
            f'expert_{i}': SNNExpert(
                input_dim=self.moe_hidden_dim,
                hidden_dim=self.hidden_dim // 2,
                output_dim=self.moe_hidden_dim
            )
            for i in range(self.num_experts)
        })
        
        # 5. Liquid MoE Router
        self.moe_router = LiquidMoERouter(
            in_dim=self.moe_hidden_dim,
            hidden_dim=64,
            num_experts=self.num_experts,
            top_k=self.top_k
        )
        
        # 6. Spike Bridge (Post-Routing)
        self.continuous_to_spike = ContinuousToSpikeBridge(
            input_dim=self.moe_hidden_dim,
            spike_dim=self.hidden_dim,
            encoding='poisson'
        )
        
        # 7. SNN Decoder (GIF)
        self.decoder = GIFNeuron(self.hidden_dim, self.embed_dim, L=16)
        
        # 8. Output Norm
        self.output_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, 
                inputs_embeds: torch.Tensor, 
                input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        
        # 1. Prosody Modulation
        if input_ids is not None:
            attention_gains, _ = self.prosody_attention(input_ids)
            modulated_input = inputs_embeds * attention_gains.unsqueeze(-1)
        else:
            modulated_input = inputs_embeds
            attention_gains = None

        # 2. Encode to Spikes
        spikes_enc, _ = self.encoder(modulated_input)
        
        # 3. Prepare for MoE Routing
        spikes_flat = spikes_enc.reshape(batch_size * seq_len, 1, self.hidden_dim)
        continuous_flat = self.spike_to_continuous(spikes_flat)
        
        # 4. Route (Liquid MoE)
        flat_gains = attention_gains.view(-1, 1) if attention_gains is not None else None
        route_out = self.moe_router(continuous_flat, attn_gain=flat_gains)
        
        topk_indices = route_out['indices']
        topk_weights = route_out['weights']
        
        # 5. Execute Experts
        expert_outputs = torch.zeros_like(continuous_flat)
        
        for i in range(self.num_experts):
            is_selected = (topk_indices == i)
            if not is_selected.any(): continue
                
            ex_out = self.experts[f'expert_{i}'].predict(continuous_flat)
            w = (topk_weights * is_selected.float()).sum(dim=1).unsqueeze(1)
            expert_outputs += ex_out * w
            
        # 6. Convert back to Spikes
        # FIXED: Removed 'device' argument
        spikes_moe = self.continuous_to_spike(expert_outputs)
        
        spikes_moe = spikes_moe.view(batch_size, seq_len, -1, self.hidden_dim)
        spikes_moe_avg = spikes_moe.mean(dim=2)
        
        # 7. Decode
        if attention_gains is not None:
            spikes_moe_avg = spikes_moe_avg * attention_gains.unsqueeze(-1)
            
        decoded, _ = self.decoder(spikes_moe_avg)
        
        # 8. Output Norm
        return self.output_norm(decoded)