"""
GPU-Native MoE Language Zone.

Optimized to remove AsyncIO/CPU bottlenecks.
Integrates Liquid MoE routing directly into the PyTorch forward pass.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

# Import GPU-native components
from core.language_zone.spike_bridge import SpikeToContinuousBridge, ContinuousToSpikeBridge
from core.language_zone.snn_expert import SNNExpert
from core.language_zone.gif_neuron import GIFNeuron
# Ensure we import the GPU-native router we created earlier
from core.liquid_moe import LiquidMoERouter


class MoELanguageZone(nn.Module):
    """
    Language Zone with integrated Liquid MoE routing (GPU Native).
    
    Architecture:
    1. Input spikes → GIF encoder
    2. Spike → continuous bridge (Rate Coding)
    3. Liquid MoE routing (Batched)
    4. Expert processing (SNN-based)
    5. Continuous → spike bridge (Poisson)
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
        self.moe_hidden_dim = moe_hidden_dim
        self.top_k = top_k
        self.num_experts = num_experts
        
        # 1. Binary embedding layer
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Encoder: embeddings → spikes
        self.encoder = GIFNeuron(embed_dim, hidden_dim, L=16)
        
        # 3. Spike → continuous bridge for MoE routing
        self.spike_to_continuous = SpikeToContinuousBridge(
            spike_dim=hidden_dim,
            output_dim=moe_hidden_dim,
            encoding='rate',
            time_window=10
        )
        
        # 4. SNN Experts (ModuleDict for registration)
        self.experts = nn.ModuleDict({
            f'expert_{i}': SNNExpert(
                input_dim=moe_hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=moe_hidden_dim
            )
            for i in range(num_experts)
        })
        
        # 5. Liquid MoE Router (Initialized internally to ensure GPU compat)
        self.moe_router = LiquidMoERouter(
            in_dim=moe_hidden_dim,
            hidden_dim=64,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # 6. Continuous → spike bridge after MoE
        self.continuous_to_spike = ContinuousToSpikeBridge(
            input_dim=moe_hidden_dim,
            spike_dim=hidden_dim,
            encoding='poisson'
        )
        
        # 7. Decoder: spikes → output
        self.decoder = GIFNeuron(hidden_dim, embed_dim, L=16)
        
        # 8. Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Synchronous GPU forward pass with MoE routing.
        
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
        
        # 2. Encode to spikes (GIF maintains state internally or we pass None)
        # Note: GIFNeuron needs to support sequence processing or we loop here.
        # Assuming our optimized GIFNeuron can handle (B, T, D) -> (B, T, H)
        spikes_enc, _ = self.encoder(embeds) 
        
        # 3. Convert spikes to continuous features for Routing
        # We need routing decisions at each timestep
        # To avoid a slow loop, we can process the whole sequence if bridges allow
        
        # If spike_to_continuous supports sequences:
        # [B, T, Spike_Dim] -> [B, T, MoE_Dim]
        # Our rate coding bridge uses a window, so we can apply it convolutionally or iteratively
        
        # For efficiency, let's treat (Batch * Seq) as a large batch
        spikes_flat = spikes_enc.reshape(batch_size * seq_len, 1, self.hidden_dim)
        # Bridge expects [Batch, Time, Dim]
        continuous_flat = self.spike_to_continuous(spikes_flat) # [Batch*Seq, MoE_Dim]
        
        # 4. Route through Liquid MoE
        # Router processes [Batch*Seq, Dim] in parallel
        route_out = self.moe_router(continuous_flat)
        
        # route_out['indices']: [Batch*Seq, top_k]
        # route_out['weights']: [Batch*Seq, top_k]
        topk_indices = route_out['indices']
        topk_weights = route_out['weights']
        
        # 5. Expert Dispatch (Vectorized)
        # We need to run specific experts for specific tokens.
        # Standard MoE approach: Permute tokens to experts, run, permute back.
        # Or masked run: Run all experts, mask outputs.
        
        # Since num_experts is small (e.g. 8), masked run is often faster than sparse scatter/gather
        # on modern GPUs due to memory access patterns.
        
        expert_outputs = torch.zeros(batch_size * seq_len, self.moe_hidden_dim, device=device)
        
        # Reshape inputs for experts: [Batch*Seq, 1, Dim] -> Expert -> [Batch*Seq, Dim]
        expert_input = continuous_flat.unsqueeze(1) 
        
        # Iterate over experts
        for i in range(self.num_experts):
            # Find where this expert is selected (in top_k)
            # mask: [Batch*Seq, top_k] boolean
            is_selected = (topk_indices == i)
            
            # If expert is not selected by anyone, skip
            if not is_selected.any():
                continue
                
            # Run expert on ALL inputs (or masked subset)
            # Running on all is simpler for batching, masking later
            # For 8 experts, this is 8x compute but regular.
            # Optimization: Mask inputs?
            # Creating dynamic batches causes sync. Let's run full batch and mask output.
            
            # [Batch*Seq, 1, Dim] -> [Batch*Seq, Out_Dim]
            ex_out = self.experts[f'expert_{i}'].predict(continuous_flat)
            
            # Weighted sum into output
            # Weight for this expert: sum weights across top_k where index matches
            # w: [Batch*Seq]
            w = (topk_weights * is_selected.float()).sum(dim=1).unsqueeze(1)
            
            expert_outputs += ex_out * w
            
        # 6. Convert back to spikes
        # [Batch*Seq, Dim] -> Spikes
        spikes_moe = self.continuous_to_spike(expert_outputs, device)
        
        # Reshape back to sequence: [Batch, Seq, Time_Internal, Dim]
        spikes_moe = spikes_moe.view(batch_size, seq_len, -1, self.hidden_dim)
        # Mean over internal time
        spikes_moe_avg = spikes_moe.mean(dim=2)
        
        # 7. Decode
        decoded, _ = self.decoder(spikes_moe_avg)
        
        # 8. Project
        logits = self.output_proj(decoded)
        
        return logits, {
            'probs': route_out['probs'].view(batch_size, seq_len, -1).detach().cpu()
        }

    # Backward compatibility
    def setup_moe_router(self, *args, **kwargs):
        pass # Router is now built-in
    
    def forward_async(self, *args, **kwargs):
        # Redirect to synchronous forward
        # Wrap result in future if caller expects awaitable (hacky)
        raise DeprecationWarning("Use .forward() instead of .forward_async() for GPU execution")