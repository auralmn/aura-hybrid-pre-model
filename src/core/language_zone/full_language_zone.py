"""
GPU-Native Full Language Zone.

Integrates:
- Prosody-driven Attention (GPU)
- JIT-Compiled SNN Encoder/Decoder
- Vectorized Liquid MoE Router
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

# Import GPU-native components
from core.language_zone.prosody_attention import ProsodyAttentionBridge
from core.language_zone.gif_neuron import GIFNeuron
from core.language_zone.spike_bridge import SpikeToContinuousBridge, ContinuousToSpikeBridge
from core.language_zone.snn_expert import SNNExpert
from core.liquid_moe import LiquidMoERouter


class FullLanguageZone(nn.Module):
    """
    Complete Language Zone (GPU Native).
    
    Flow:
    Input -> Prosody Attention -> SNN Encoder -> Liquid MoE -> SNN Experts -> SNN Decoder -> Output
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        moe_hidden_dim: int = 64,
        attention_preset: str = 'analytical'
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.moe_hidden_dim = moe_hidden_dim
        self.top_k = top_k
        self.num_experts = num_experts
        
        # 1. Prosody Attention (GPU)
        self.prosody_attention = ProsodyAttentionBridge(
            k_winners=top_k,
            # Note: preset logic handled inside bridge or passed as config
        )
        
        # 2. Embeddings
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # 3. SNN Encoder (GIF)
        # Note: We use standard GIFNeuron, assuming it supports tensor inputs.
        # Ideally this should be the JIT-compiled version from src/base/neuron.py if possible,
        # but here we stick to the language_zone version for compatibility.
        self.encoder = GIFNeuron(embed_dim, hidden_dim, L=16)
        
        # 4. Spike Bridge (Pre-Routing)
        self.spike_to_continuous = SpikeToContinuousBridge(
            spike_dim=hidden_dim,
            output_dim=moe_hidden_dim,
            encoding='rate'
        )
        
        # 5. SNN Experts
        self.experts = nn.ModuleDict({
            f'expert_{i}': SNNExpert(
                input_dim=moe_hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=moe_hidden_dim
            )
            for i in range(num_experts)
        })
        
        # 6. Liquid MoE Router (GPU Native)
        self.moe_router = LiquidMoERouter(
            in_dim=moe_hidden_dim,
            hidden_dim=64,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # 7. Spike Bridge (Post-Routing)
        self.continuous_to_spike = ContinuousToSpikeBridge(
            input_dim=moe_hidden_dim,
            spike_dim=hidden_dim,
            encoding='poisson'
        )
        
        # 8. SNN Decoder (GIF)
        self.decoder = GIFNeuron(hidden_dim, embed_dim, L=16)
        
        # 9. Output Head
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    # Backward compatibility stub
    def setup_moe_router(self, *args, **kwargs):
        pass 

    def forward(
        self,
        input_ids: torch.Tensor,
        token_strings: Optional[List[List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Synchronous GPU forward pass.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Prosody Attention
        # Returns: [Batch, Seq], Metadata
        # We pass input_ids directly to GPU prosody extractor
        attention_gains, attn_metadata = self.prosody_attention(input_ids)
        
        # 2. Embed
        embeds = self.embeddings(input_ids) # [B, T, D]
        
        # 3. Encode (Modulated by Prosody)
        # Note: GIFNeuron needs to support gain modulation.
        # If not, we apply gain to input embeddings:
        modulated_embeds = embeds * attention_gains.unsqueeze(-1)
        spikes_enc, _ = self.encoder(modulated_embeds)
        
        # 4. Prepare for MoE Routing
        # Reshape to flatten time: [B*T, H]
        # This allows batched routing for the whole sequence
        spikes_flat = spikes_enc.reshape(batch_size * seq_len, 1, self.hidden_dim)
        continuous_flat = self.spike_to_continuous(spikes_flat) # [B*T, MoE_Dim]
        
        # 5. Route (Liquid MoE)
        # Pass attention gains to modulate router temperature
        # Flatten gains: [B*T, 1]
        flat_gains = attention_gains.view(-1, 1)
        
        route_out = self.moe_router(continuous_flat, attn_gain=flat_gains)
        
        topk_indices = route_out['indices'] # [B*T, k]
        topk_weights = route_out['weights'] # [B*T, k]
        
        # 6. Execute Experts (Vectorized)
        expert_outputs = torch.zeros_like(continuous_flat)
        
        # Expand input for expert processing: [B*T, 1, Dim]
        expert_input = continuous_flat.unsqueeze(1)
        
        for i in range(self.num_experts):
            # Mask: where is expert i selected?
            is_selected = (topk_indices == i) # [B*T, k]
            if not is_selected.any():
                continue
                
            # Compute expert output for ALL inputs (simple masking)
            # In highly sparse scenarios, gather/scatter is better, but for 8 experts masking is fine
            ex_out = self.experts[f'expert_{i}'].predict(continuous_flat)
            
            # Weighted addition
            # Sum weights where selected
            w = (topk_weights * is_selected.float()).sum(dim=1).unsqueeze(1)
            expert_outputs += ex_out * w
            
        # 7. Convert back to Spikes
        spikes_moe = self.continuous_to_spike(expert_outputs, device)
        
        # Reshape to sequence: [B, T, Time_Internal, H]
        spikes_moe = spikes_moe.view(batch_size, seq_len, -1, self.hidden_dim)
        spikes_moe_avg = spikes_moe.mean(dim=2) # Mean over internal simulation steps
        
        # 8. Decode (Modulated)
        # Apply gains again to decoder input? Or assume encoder handled it.
        # Let's apply gains to decoder for consistency with original design.
        modulated_spikes = spikes_moe_avg * attention_gains.unsqueeze(-1)
        decoded, _ = self.decoder(modulated_spikes)
        
        # 9. Output
        logits = self.output_proj(decoded)
        
        info = {
            'attention': attn_metadata,
            'routing': route_out['probs'].view(batch_size, seq_len, -1).detach(),
            'prosody_stats': {
                'mean_gain': attention_gains.mean().item(),
                'max_gain': attention_gains.max().item()
            }
        }
        
        return logits, info

    # Async compatibility stub (deprecated)
    def forward_async(self, *args, **kwargs):
        raise DeprecationWarning("Use .forward() for GPU execution")
    
    def forward_sync(self, *args, **kwargs):
        return self.forward(*args, **kwargs)