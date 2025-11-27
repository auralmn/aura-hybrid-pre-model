import torch
import torch.nn as nn
from typing import Optional, Dict

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
        self.embed_dim = config.d_model
        self.hidden_dim = config.max_neurons 
        self.moe_hidden_dim = 64
        self.num_experts = 8
        self.top_k = 2
        
        self.prosody_attention = ProsodyAttentionBridge(k_winners=self.top_k)
        self.encoder = GIFNeuron(self.embed_dim, self.hidden_dim, L=16)
        self.spike_to_continuous = SpikeToContinuousBridge(self.hidden_dim, self.moe_hidden_dim, 'rate')
        self.experts = nn.ModuleDict({
            f'expert_{i}': SNNExpert(self.moe_hidden_dim, self.hidden_dim // 2, self.moe_hidden_dim)
            for i in range(self.num_experts)
        })
        self.moe_router = LiquidMoERouter(self.moe_hidden_dim, 64, self.num_experts, self.top_k)
        self.continuous_to_spike = ContinuousToSpikeBridge(self.moe_hidden_dim, self.hidden_dim, 'poisson')
        self.decoder = GIFNeuron(self.hidden_dim, self.embed_dim, L=16)
        self.output_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, inputs_embeds: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = inputs_embeds.shape
        device = inputs_embeds.device
        
        # 1. Prosody (Deterministic)
        if input_ids is not None:
            attention_gains, _ = self.prosody_attention(input_ids)
            inputs = inputs_embeds * attention_gains.unsqueeze(-1)
        else:
            inputs = inputs_embeds
            attention_gains = None

        # 2. Encode (Stateless)
        spikes_enc, _ = self.encoder(inputs, state=None)
        
        # 3. Route (Stateless)
        spikes_flat = spikes_enc.reshape(batch * seq, 1, self.hidden_dim)
        continuous = self.spike_to_continuous(spikes_flat)
        
        flat_gains = attention_gains.view(-1, 1) if attention_gains is not None else None
        route_out = self.moe_router(continuous, attn_gain=flat_gains)
        
        indices = route_out['indices']
        weights = route_out['weights']
        
        # 4. Sparse Expert Exec (Stateless Experts)
        output_flat = torch.zeros_like(continuous)
        for i in range(self.num_experts):
            mask = (indices == i).any(dim=1)
            if not mask.any(): continue
            
            active_idx = torch.where(mask)[0]
            active_in = continuous[active_idx]
            
            # Predict calls forward(state=None) internally
            active_out = self.experts[f'expert_{i}'].predict(active_in)
            
            # Weighted combination
            w_sum = (weights[active_idx] * (indices[active_idx] == i).float()).sum(dim=1, keepdim=True)
            output_flat.index_add_(0, active_idx, active_out * w_sum)
            
        # 5. Decode (Stateless)
        spikes_moe = self.continuous_to_spike(output_flat)
        spikes_moe = spikes_moe.view(batch, seq, -1, self.hidden_dim).mean(dim=2)
        
        if attention_gains is not None:
            spikes_moe = spikes_moe * attention_gains.unsqueeze(-1)
            
        decoded, _ = self.decoder(spikes_moe, state=None)
        return self.output_norm(decoded)