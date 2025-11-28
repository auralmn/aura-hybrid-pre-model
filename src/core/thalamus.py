"""
The Thalamus: Sensory Gateway and Cortical Router.

Responsibilities:
1. Gating: Filters raw input based on salience.
2. Routing: Directs signals to appropriate Cortical Regions (Liquid MoE).
3. Modulation: Adjusts routing based on Limbic feedback (fear/excitement).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from src.core.liquid_moe import LiquidMoERouter

class Thalamus(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 region_names: list[str], 
                 hidden_dim: int = 256,
                 top_k: int = 3):
        super().__init__()
        self.d_model = d_model
        self.region_names = region_names
        
        # Liquid State Machine for dynamic routing
        self.router = LiquidMoERouter(
            in_dim=d_model,
            hidden_dim=hidden_dim,
            num_experts=len(region_names),
            top_k=min(top_k, len(region_names))
        )
        
        # Sensory Gating (Gain control)
        # Gates input amplitude based on signal strength + top-down modulation
        self.sensory_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, 
                x: torch.Tensor, 
                limbic_state: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Forward pass through Thalamic nuclei.
        
        Args:
            x: Raw Input [Batch, Seq, Dim]
            limbic_state: Dict with 'arousal', 'valence' (from Amygdala)
            
        Returns:
            routed_signals: Dict mapping region_name -> Gated Input Tensor
            stats: Routing statistics
        """
        batch_size = x.shape[0]
        
        # 1. Sensory Gating (Thalamic Reticular Nucleus function)
        gate = self.sensory_gate(x)
        
        # Modulate gate with arousal if provided
        if limbic_state and 'arousal' in limbic_state:
            arousal = limbic_state['arousal']
            gate = torch.clamp(gate * (1.0 + arousal), 0.0, 1.0)
            
        gated_input = x * gate
        
        # 2. Cortical Routing (Liquid State Dynamics)
        pooled_x = gated_input.mean(dim=1)
        
        # Create attention gain tensor with proper dtype
        attn_gain = None
        if limbic_state:
            arousal_val = limbic_state.get('arousal', 0.0)
            attn_gain = torch.full((batch_size, 1), arousal_val, device=x.device, dtype=x.dtype)
             
        routing_out = self.router(pooled_x, attn_gain=attn_gain)
        
        # 3. Dispatch Signals
        indices = routing_out['indices']
        weights = routing_out['weights']
        
        routed_signals = {}
        active_indices = torch.unique(indices)
        
        for idx in active_indices:
            region_idx = idx.item()
            if region_idx >= len(self.region_names): 
                continue
            
            region_name = self.region_names[region_idx]
            
            # Create mask for this region
            is_selected = (indices == idx)
            
            # Calculate region-specific gain
            region_gain = (weights * is_selected.float()).sum(dim=1).view(-1, 1, 1)
            
            routed_signals[region_name] = gated_input * region_gain
            
        return routed_signals, routing_out['probs']