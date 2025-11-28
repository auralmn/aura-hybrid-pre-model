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
    """Stateless Liquid Cell."""
    def __init__(self, config: LiquidCellConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.dt = config.dt
        self.tau_min = config.tau_min
        self.tau_max = config.tau_max
        
        self.W = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.U = nn.Linear(config.in_dim, config.hidden_dim)
        self.V = nn.Linear(config.in_dim, config.hidden_dim)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        # Always init fresh if None (Stateless)
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            
        vx = self.V(x)
        tau = self.tau_min + F.softplus(vx)
        tau = torch.clamp(tau, max=self.tau_max)
        
        gates = torch.tanh(self.W(h_prev) + self.U(x))
        dh = -h_prev / (tau + 1e-6) + gates
        
        return h_prev + self.dt * dh

class LiquidMoERouter(nn.Module):
    """Stateless Router."""
    def __init__(self, in_dim: int, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.config = LiquidCellConfig(in_dim, hidden_dim)
        self.cell = LiquidCell(self.config)
        self.gate_proj = nn.Linear(hidden_dim, num_experts)
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.temperature = 1.0

    def forward(self, x: torch.Tensor, attn_gain: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        # 1. One-step liquid dynamics from zero state
        h = self.cell(x, h_prev=None)
        
        # 2. Logits
        logits = self.gate_proj(h)
        
        # 3. Temperature scaling
        if attn_gain is not None:
            if attn_gain.dim() == 1: 
                attn_gain = attn_gain.unsqueeze(1)
            temp = torch.clamp(self.temperature / (attn_gain + 1e-6), min=0.1, max=5.0)
            logits = logits / temp
        else:
            logits = logits / self.temperature
            
        # 4. Top-K selection with numerical stability
        probs = F.softmax(logits, dim=-1)
        
        # Clamp top_k to available experts
        k = min(self.top_k, self.num_experts)
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 5. Stats (No Grad)
        if self.training:
            with torch.no_grad():
                batch_usage = torch.zeros_like(self.expert_usage)
                flat_indices = topk_indices.reshape(-1).long()
                ones = torch.ones(flat_indices.numel(), device=x.device, dtype=self.expert_usage.dtype)
                batch_usage.index_add_(0, flat_indices, ones)
                self.expert_usage.mul_(0.99).add_(batch_usage / max(x.size(0), 1), alpha=0.01)
            
        return {
            'weights': topk_weights,
            'indices': topk_indices,
            'probs': probs
        }