"""
GPU-Native Full Language Zone (Temporal Cortex) - Sparse Optimized.

Integrates:
- Prosody-driven Attention
- JIT-Compiled SNN Encoder/Decoder
- Vectorized Liquid MoE Router with SPARSE EXECUTION
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
    def __init__(self, config: BrainZoneConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embed_dim = config.d_model
        self.hidden_dim = config.max_neurons 
        self.moe_hidden_dim = 64
        self.num_experts = 8
        self.top_k = 2
        
        self.prosody_attention = ProsodyAttentionBridge(k_winners=self.top_k)
        self.encoder = GIFNeuron(self.embed_dim, self.hidden_dim, L=16)
        
        self.spike_to_continuous = SpikeToContinuousBridge(
            spike_dim=self.hidden_dim,
            output_dim=self.moe_hidden_dim,
            encoding='rate'
        )
        
        self.experts = nn.ModuleDict({
            f'expert_{i}': SNNExpert(
                input_dim=self.moe_hidden_dim,
                hidden_dim=self.hidden_dim // 2,
                output_dim=self.moe_hidden_dim
            )
            for i in range(self.num_experts)
        })
        
        self.moe_router = LiquidMoERouter(
            in_dim=self.moe_hidden_dim,
            hidden_dim=64,
            num_experts=self.num_experts,
            top_k=self.top_k
        )
        
        self.continuous_to_spike = ContinuousToSpikeBridge(
            input_dim=self.moe_hidden_dim,
            spike_dim=self.hidden_dim,
            encoding='poisson'
        )
        
        self.decoder = GIFNeuron(self.hidden_dim, self.embed_dim, L=16)
        self.output_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, 
                inputs_embeds: torch.Tensor, 
                input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        # Flatten: [Batch*Seq, Hidden]
        spikes_flat = spikes_enc.reshape(batch_size * seq_len, 1, self.hidden_dim)
        continuous_flat = self.spike_to_continuous(spikes_flat) # [Batch*Seq, MoE_Dim]
        
        # 4. Route (Liquid MoE)
        flat_gains = attention_gains.view(-1, 1) if attention_gains is not None else None
        route_out = self.moe_router(continuous_flat, attn_gain=flat_gains)
        
        # indices: [B*T, k], weights: [B*T, k]
        topk_indices = route_out['indices']
        topk_weights = route_out['weights']
        
        # 5. Sparse Expert Execution
        # Only run experts on tokens that selected them
        expert_outputs = torch.zeros_like(continuous_flat)
        
        for i in range(self.num_experts):
            # Identify which tokens selected this expert
            # mask: [B*T, k] -> True if expert i was selected at rank k
            selection_mask = (topk_indices == i)
            
            # Collapse mask to find tokens that selected expert i at *any* rank
            # token_mask: [B*T] boolean
            token_mask = selection_mask.any(dim=1)
            
            if not token_mask.any():
                continue
                
            # Gather inputs for this expert
            # indices: [Num_Active]
            active_indices = torch.where(token_mask)[0]
            active_inputs = continuous_flat[active_indices]
            
            # Run expert (Process only active tokens)
            # [Num_Active, Dim] -> Expert -> [Num_Active, Dim]
            active_out = self.experts[f'expert_{i}'].predict(active_inputs)
            
            # Calculate aggregate weight for these tokens
            # Sum weights across ranks (in case same expert selected twice, though unlikely)
            # active_weights: [Num_Active, 1]
            active_weights = (topk_weights[active_indices] * selection_mask[active_indices].float()).sum(dim=1, keepdim=True)
            
            # Weighted output
            weighted_out = active_out * active_weights
            
            # Scatter add back to global output
            # We can use index_add_ or simple indexing since indices are unique
            expert_outputs.index_add_(0, active_indices, weighted_out)
            
        # 6. Convert back to Spikes
        # [Batch*Seq, Dim] -> Spikes
        spikes_moe = self.continuous_to_spike(expert_outputs)
        
        spikes_moe = spikes_moe.view(batch_size, seq_len, -1, self.hidden_dim)
        spikes_moe_avg = spikes_moe.mean(dim=2)
        
        # 7. Decode
        if attention_gains is not None:
            spikes_moe_avg = spikes_moe_avg * attention_gains.unsqueeze(-1)
            
        decoded, _ = self.decoder(spikes_moe_avg)
        
        # 8. Output Norm
        return self.output_norm(decoded)