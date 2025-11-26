import torch
import torch.nn as nn
import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple

from src.core.language_zone.spike_bridge import SpikeToContinuousBridge, ContinuousToSpikeBridge
from src.core.language_zone.snn_expert import SNNExpert
from src.core.language_zone.gif_neuron import GIFNeuron


class TorchExpertWrapper:
    """Wrapper to make torch SNNExpert compatible with numpy MoE router interface."""
    
    def __init__(self, torch_expert: SNNExpert):
        self.expert = torch_expert
    
    def predict(self, x: np.ndarray) -> float:
        return self.expert.predict(x)
    
    async def update(self, x: np.ndarray, target: float, token_ids: List[int], attention_bundle):
        # Async wrapper
        await asyncio.to_thread(self.expert.update, x, target, token_ids, attention_bundle)
    
    def state_dict(self) -> Dict:
        return self.expert.state_dict()
    
    def load_state_dict(self, state: Dict):
        self.expert.load_state_dict(state)


class MoELanguageZone(nn.Module):
    """
    Language Zone with integrated Liquid MoE routing.
    
    Architecture:
    1. Input spikes → GIF encoder
    2. Spike → continuous bridge
    3. Liquid MoE routing
    4. Expert processing (SNN-based)
    5. Continuous → spike bridge
    6. GIF decoder
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
        
        # Binary embedding layer
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder: embeddings → spikes
        self.encoder = GIFNeuron(embed_dim, hidden_dim, L=16)
        
        # Spike → continuous bridge for MoE routing
        self.spike_to_continuous = SpikeToContinuousBridge(
            spike_dim=hidden_dim,
            output_dim=moe_hidden_dim,
            encoding='rate'
        )
        
        # Create SNN experts
        self.experts = nn.ModuleDict({
            f'expert_{i}': SNNExpert(
                input_dim=moe_hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=moe_hidden_dim
            )
            for i in range(num_experts)
        })
        
        # Liquid MoE router (will be set externally)
        self.moe_router = None
        self.top_k = top_k
        self.moe_hidden_dim = moe_hidden_dim
        
        # Continuous → spike bridge after MoE
        self.continuous_to_spike = ContinuousToSpikeBridge(
            input_dim=moe_hidden_dim,
            spike_dim=hidden_dim,
            encoding='poisson'
        )
        
        # Decoder: spikes → output
        self.decoder = GIFNeuron(hidden_dim, embed_dim, L=16)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def setup_moe_router(self, LiquidMoERouter):
        """
        Setup Liquid MoE router with SNN experts.
        
        Call this after initialization to inject your router class.
        """
        # Create expert dict for MoE router
        # Wrap torch experts to provide numpy interface
        expert_dict = {}
        for name, expert in self.experts.items():
            expert_dict[name] = TorchExpertWrapper(expert)
        
        # Initialize router
        self.moe_router = LiquidMoERouter(
            experts=expert_dict,
            in_dim=self.moe_hidden_dim,
            hidden_dim=64,  # Router's internal hidden dim
            top_k=self.top_k
        )
    
    async def forward_async(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Async forward pass with MoE routing.
        
        Args:
            input_ids: (batch, seq_len) token ids
            attention_mask: Optional (batch, seq_len) mask
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            routing_info: Dict with MoE statistics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Embed tokens
        embeds = self.embeddings(input_ids)  # (batch, seq_len, embed_dim)
        
        # 2. Encode to spikes
        spikes_enc, _ = self.encoder(embeds, state=None)  # (batch, seq_len, hidden_dim)
        
        # 3. Convert spikes to continuous for MoE routing
        # Route per sequence position
        outputs_list = []
        routing_stats = {'expert_usage': {}, 'energy_total': 0.0}
        
        for t in range(seq_len):
            spikes_t = spikes_enc[:, t:t+1, :]  # (batch, 1, hidden_dim)
            continuous_t = self.spike_to_continuous(spikes_t)  # (batch, moe_hidden_dim) numpy
            
            # 4. Route through MoE (per batch element)
            batch_outputs = []
            for b in range(batch_size):
                x_b = continuous_t[b]  # (moe_hidden_dim,)
                
                # Route to experts
                route_info = await self.moe_router.route(x_b, attn_gain=1.0)
                
                # Get expert predictions
                y_hat = route_info['y_hat']
                batch_outputs.append(y_hat)
                
                # Track routing stats
                for expert_name, gate in route_info['topk']:
                    routing_stats['expert_usage'][expert_name] = \
                        routing_stats['expert_usage'].get(expert_name, 0) + gate
                routing_stats['energy_total'] += route_info['energy_j']
            
            # Collect batch outputs
            outputs_t = np.array(batch_outputs).reshape(batch_size, 1)  # (batch, 1)
            outputs_list.append(outputs_t)
        
        # Stack outputs over time
        outputs_np = np.concatenate(outputs_list, axis=1)  # (batch, seq_len)
        outputs_np = np.expand_dims(outputs_np, -1)  # (batch, seq_len, 1)
        
        # Expand to moe_hidden_dim (repeat along feature dim)
        outputs_np = np.repeat(outputs_np, self.moe_hidden_dim, axis=-1)  # (batch, seq_len, moe_hidden_dim)
        
        # 5. Convert back to spikes
        spikes_moe = self.continuous_to_spike(
            outputs_np.reshape(-1, self.moe_hidden_dim),
            device=device
        )  # (batch*seq_len, num_timesteps, hidden_dim)
        
        # Reshape and average over internal timesteps
        spikes_moe = spikes_moe.view(batch_size, seq_len, -1, self.hidden_dim)
        spikes_moe = spikes_moe.mean(dim=2)  # (batch, seq_len, hidden_dim)
        
        # 6. Decode spikes to embeddings
        decoded, _ = self.decoder(spikes_moe, state=None)  # (batch, seq_len, embed_dim)
        
        # 7. Project to vocabulary
        logits = self.output_proj(decoded)  # (batch, seq_len, vocab_size)
        
        return logits, routing_stats
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Synchronous forward wrapper (blocks on async).
        """
        return asyncio.run(self.forward_async(input_ids, attention_mask))
    
    def forward_sync(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimized synchronous forward (no async overhead).
        Use this for production/training.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # For now, this is a simplified version without MoE routing
        # In production, vectorize the MoE router for batch processing
        
        # 1. Embed tokens
        embeds = self.embeddings(input_ids)
        
        # 2. Encode to spikes
        spikes_enc, _ = self.encoder(embeds, state=None)
        
        # 3. For now, bypass MoE and go straight to decoder
        # (MoE integration requires async or batch-optimized router)
        
        # 4. Decode spikes
        decoded, _ = self.decoder(spikes_enc, state=None)
        
        # 5. Project to vocabulary
        logits = self.output_proj(decoded)
        
        routing_stats = {'expert_usage': {}, 'energy_total': 0.0}
        
        return logits, routing_stats
