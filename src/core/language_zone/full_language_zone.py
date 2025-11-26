import torch
import torch.nn as nn
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple

from .prosody_attention import ProsodyAttentionBridge
from .prosody_gif import ProsodyModulatedGIF
from .spike_bridge import SpikeToContinuousBridge, ContinuousToSpikeBridge
from .snn_expert import SNNExpert
from .moe_language_zone import TorchExpertWrapper


class FullLanguageZone(nn.Module):
    """
    Complete Language Zone with:
    - Prosody-driven attention (amplitude/pitch/boundary spikes)
    - Prosody-modulated GIF neurons
    - Liquid MoE routing
    - SNN experts
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
        
        # Prosody-driven attention
        self.prosody_attention = ProsodyAttentionBridge(
            attention_preset=attention_preset,
            k_winners=top_k
        )
        
        # Binary embedding layer
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Prosody-modulated encoder
        self.encoder = ProsodyModulatedGIF(
            embed_dim, hidden_dim, L=16,
            attention_modulation_strength=0.3
        )
        
        # Spike → continuous bridge for MoE routing
        self.spike_to_continuous = SpikeToContinuousBridge(
            spike_dim=hidden_dim,
            output_dim=moe_hidden_dim,
            encoding='rate'
        )
        
        # SNN experts
        self.experts = nn.ModuleDict({
            f'expert_{i}': SNNExpert(
                input_dim=moe_hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=moe_hidden_dim
            )
            for i in range(num_experts)
        })
        
        # Liquid MoE router (set via setup_moe_router)
        self.moe_router = None
        self.top_k = top_k
        self.moe_hidden_dim = moe_hidden_dim
        
        # Continuous → spike bridge
        self.continuous_to_spike = ContinuousToSpikeBridge(
            input_dim=moe_hidden_dim,
            spike_dim=hidden_dim,
            encoding='poisson'
        )
        
        # Prosody-modulated decoder
        self.decoder = ProsodyModulatedGIF(
            hidden_dim, embed_dim, L=16,
            attention_modulation_strength=0.2
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def setup_moe_router(self, LiquidMoERouter):
        """Setup Liquid MoE router with SNN experts."""
        expert_dict = {name: TorchExpertWrapper(expert) 
                      for name, expert in self.experts.items()}
        
        self.moe_router = LiquidMoERouter(
            experts=expert_dict,
            in_dim=self.moe_hidden_dim,
            hidden_dim=64,
            top_k=self.top_k
        )
    
    def forward_sync(
        self,
        input_ids: torch.Tensor,
        token_strings: Optional[List[List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Synchronous forward (no MoE routing, for testing/training).
        
        Args:
            input_ids: (batch, seq_len) token IDs
            token_strings: Optional token string lists for prosody extraction
            attention_mask: Optional attention mask
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            info: Dict with attention statistics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Compute prosody-driven attention gains
        attention_gains, attn_metadata = self.prosody_attention(input_ids, token_strings)
        
        # 2. Embed tokens
        embeds = self.embeddings(input_ids)
        
        # 3. Encode with prosody modulation
        spikes_enc, _ = self.encoder(
            embeds,
            state=None,
            attention_gains=attention_gains
        )
        
        # 4. Bypass MoE, go straight to decoder
        # (Full MoE requires async or batch-optimized router)
        
        # 5. Decode spikes
        decoded, _ = self.decoder(
            spikes_enc,
            state=None,
            attention_gains=attention_gains
        )
        
        # 6. Project to vocabulary
        logits = self.output_proj(decoded)
        
        info = {
            'attention': attn_metadata,
            'prosody_stats': {
                'mean_gain': attention_gains.mean().item(),
                'max_gain': attention_gains.max().item(),
                'min_gain': attention_gains.min().item()
            }
        }
        
        return logits, info
    
    async def forward_async(
        self,
        input_ids: torch.Tensor,
        token_strings: Optional[List[List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Async forward with full MoE routing (requires setup_moe_router).
        
        Args:
            input_ids: (batch, seq_len) token IDs
            token_strings: Optional token string lists for prosody extraction
            attention_mask: Optional attention mask
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            info: Dict with routing and attention statistics
        """
        if self.moe_router is None:
            # Fall back to sync version
            return self.forward_sync(input_ids, token_strings, attention_mask)
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Compute prosody-driven attention gains
        attention_gains, attn_metadata = self.prosody_attention(input_ids, token_strings)
        
        # 2. Embed tokens
        embeds = self.embeddings(input_ids)
        
        # 3. Encode with prosody modulation
        spikes_enc, _ = self.encoder(
            embeds,
            state=None,
            attention_gains=attention_gains
        )
        
        # 4. Route through MoE with prosody-informed gains
        outputs_list = []
        routing_stats = {
            'expert_usage': {},
            'energy_total': 0.0,
            'prosody_influence': []
        }
        
        for t in range(seq_len):
            spikes_t = spikes_enc[:, t:t+1, :]
            continuous_t = self.spike_to_continuous(spikes_t)
            
            batch_outputs = []
            for b in range(batch_size):
                x_b = continuous_t[b]
                
                # Use prosody gain as attn_gain for MoE router
                attn_gain = float(attention_gains[b, t].cpu().item())
                
                # Route with prosody modulation
                route_info = await self.moe_router.route(x_b, attn_gain=attn_gain)
                
                y_hat = route_info['y_hat']
                batch_outputs.append(y_hat)
                
                # Track statistics
                for expert_name, gate in route_info['topk']:
                    routing_stats['expert_usage'][expert_name] = \
                        routing_stats['expert_usage'].get(expert_name, 0) + gate
                routing_stats['energy_total'] += route_info['energy_j']
                routing_stats['prosody_influence'].append(attn_gain)
            
            outputs_t = np.array(batch_outputs).reshape(batch_size, 1)
            outputs_list.append(outputs_t)
        
        # 5. Convert MoE outputs back to spikes
        outputs_np = np.concatenate(outputs_list, axis=1)
        outputs_np = np.expand_dims(outputs_np, -1)
        outputs_np = np.repeat(outputs_np, self.moe_hidden_dim, axis=-1)
        
        spikes_moe = self.continuous_to_spike(
            outputs_np.reshape(-1, self.moe_hidden_dim),
            device=device
        )
        
        spikes_moe = spikes_moe.view(batch_size, seq_len, -1, self.hidden_dim)
        spikes_moe = spikes_moe.mean(dim=2)
        
        # 6. Decode with prosody modulation
        decoded, _ = self.decoder(
            spikes_moe,
            state=None,
            attention_gains=attention_gains
        )
        
        # 7. Project to vocabulary
        logits = self.output_proj(decoded)
        
        info = {
            'routing': routing_stats,
            'attention': attn_metadata,
            'prosody_stats': {
                'mean_gain': attention_gains.mean().item(),
                'max_gain': attention_gains.max().item(),
                'min_gain': attention_gains.min().item()
            }
        }
        
        return logits, info
    
    def forward(self, input_ids, token_strings=None, attention_mask=None):
        """Default forward (uses sync version)."""
        return self.forward_sync(input_ids, token_strings, attention_mask)
