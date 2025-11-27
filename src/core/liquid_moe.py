"""
GPU-Native Liquid Mixture-of-Experts Router.

Optimizations:
- Pure PyTorch implementation (replaces NumPy/SciPy/AsyncIO).
- Batched execution for high throughput.
- JIT-compatible gating network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
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
    Differentiable Liquid Time-Constant (LTC) Cell.
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
        
        self.register_buffer('h', torch.zeros(1, config.hidden_dim))

    def reset_state(self, batch_size: int = 1):
        if self.h.size(0) != batch_size:
            self.h = torch.zeros(batch_size, self.hidden_dim, device=self.W.weight.device)
        else:
            self.h.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward step.
        x: [Batch, In_Dim]
        """
        batch_size = x.size(0)
        if self.h.size(0) != batch_size:
            self.reset_state(batch_size)
            
        # Compute time constant
        # tau = tau_min + softplus(V*x + c)
        vx = self.V(x)
        tau = self.tau_min + F.softplus(vx)
        tau = torch.clamp(tau, max=self.tau_max)
        
        # Compute dynamics
        # dh/dt = -h/tau + tanh(W*h + U*x + b)
        gates = torch.tanh(self.W(self.h) + self.U(x))
        dh = -self.h / (tau + 1e-6) + gates
        
        # Euler step
        self.h = self.h + self.dt * dh
        return self.h

class LiquidMoERouter(nn.Module):
    """
    Liquid MoE Router (GPU Native).
    Routes inputs to top-k experts based on liquid dynamics.
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
        
        # Usage tracking (for load balancing)
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.temperature = 1.0

    def forward(self, x: torch.Tensor, attn_gain: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Route inputs to experts.
        Args:
            x: Input features [Batch, In_Dim]
            attn_gain: Optional gain modulation from Prosody [Batch, 1]
        """
        # 1. Update Liquid State
        h = self.cell(x)
        
        # 2. Compute Logits
        logits = self.gate_proj(h)
        
        # 3. Apply Temperature Scaling (modulated by attention/prosody)
        # High attention -> Low temp -> Sharper focus
        if attn_gain is not None:
            # attn_gain shape: [Batch] or [Batch, 1]
            if attn_gain.dim() == 1: attn_gain = attn_gain.unsqueeze(1)
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
        
        # 6. Update Usage Stats (EMA)
        with torch.no_grad():
            batch_usage = torch.zeros_like(self.expert_usage)
            # Scatter add usage counts
            # Flatten indices to scatter
            flat_indices = topk_indices.view(-1)
            batch_usage.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
            # EMA Update
            self.expert_usage = 0.99 * self.expert_usage + 0.01 * (batch_usage / x.size(0))
            
        return {
            'weights': topk_weights,  # [Batch, k]
            'indices': topk_indices,  # [Batch, k]
            'probs': probs            # [Batch, Num_Experts]
        }