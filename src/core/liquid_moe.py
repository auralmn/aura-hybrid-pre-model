"""
GPU-Native Liquid Mixture-of-Experts Router.

Optimizations:
- Stateless LiquidCell for Checkpoint compatibility.
- Pure functional data flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class LiquidCellConfig:
    in_dim: int
    hidden_dim: int
    dt: float = 0.02
    tau_min: float = 0.02
    tau_max: float = 2.0

class LiquidCell(nn.Module):
    """
    Differentiable Liquid Time-Constant (LTC) Cell (Stateless).
    """
    def __init__(self, config: LiquidCellConfig):
        super().__init__()
        self.in_dim = config.in_dim
        self.hidden_dim = config.hidden_dim
        self.dt = config.dt
        self.tau_min = config.tau_min
        self.tau_max = config.tau_max
        
        # Parameters
        self.W = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.U = nn.Linear(config.in_dim, config.hidden_dim)
        self.V = nn.Linear(config.in_dim, config.hidden_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward step.
        x: [Batch, In_Dim]
        h_prev: Optional [Batch, Hidden] (Defaults to 0)
        """
        batch_size = x.size(0)
        
        # Initialize state locally if not provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            
        # Compute time constant
        # tau = tau_min + softplus(V*x + c)
        vx = self.V(x)
        tau = self.tau_min + F.softplus(vx)
        tau = torch.clamp(tau, max=self.tau_max)
        
        # Compute dynamics
        # dh/dt = -h/tau + tanh(W*h + U*x + b)
        gates = torch.tanh(self.W(h_prev) + self.U(x))
        dh = -h_prev / (tau + 1e-6) + gates
        
        # Euler step
        h_new = h_prev + self.dt * dh
        return h_new

class LiquidMoERouter(nn.Module):
    """
    Liquid MoE Router (Stateless Wrapper).
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        
        # Liquid Gating Network
        config = LiquidCellConfig(in_dim, hidden_dim)
        self.cell = LiquidCell(config)
        
        # Project hidden state to expert logits
        self.gate_proj = nn.Linear(hidden_dim, num_experts)
        
        # Usage tracking (buffer is safe if we don't need gradients for it)
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.temperature = 1.0

    def forward(self, x: torch.Tensor, attn_gain: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Route inputs to experts.
        """
        # 1. Update Liquid State (Single step from zero state for routing)
        # Since we treat each token independently in the current flow, we start from 0.
        h = self.cell(x, h_prev=None)
        
        # 2. Compute Logits
        logits = self.gate_proj(h)
        
        # 3. Apply Temperature Scaling
        if attn_gain is not None:
            if attn_gain.dim() == 1: attn_gain = attn_gain.unsqueeze(1)
            # Expand scalar/vector gain to match logits batch size if needed
            # Typically attn_gain is [Batch*Seq, 1] matching x [Batch*Seq, Dim]
            temp = torch.clamp(self.temperature / (attn_gain + 1e-6), min=0.1, max=5.0)
            logits = logits / temp
        else:
            logits = logits / self.temperature
            
        # 4. Softmax
        probs = F.softmax(logits, dim=-1)
        
        # 5. Top-K Selection
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Normalize weights
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 6. Update Usage Stats (No gradients needed)
        if self.training:
            with torch.no_grad():
                batch_usage = torch.zeros_like(self.expert_usage)
                flat_indices = topk_indices.view(-1)
                # One-hot scatter would be better, but count is fine
                # Only update for valid indices
                flat_indices = flat_indices.long()
                batch_usage.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
                
                # EMA Update
                self.expert_usage.mul_(0.99).add_(batch_usage / x.size(0), alpha=0.01)
            
        return {
            'weights': topk_weights,
            'indices': topk_indices,
            'probs': probs
        }