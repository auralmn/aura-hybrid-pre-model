"""
GPU-Native Spike-Timing Dependent Plasticity (STDP) Learner.

Optimizations:
- Replaced token-by-token Python loops with vectorized tensor operations.
- Implemented eligibility traces (exponential decay) using parallel prefix scans (cumsum) or recurrence.
- Updates weights in-place on GPU using scatter_add_.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class STDPLearner(nn.Module):
    """
    Vectorized STDP learner for token salience.
    Learns temporal associations using GPU-accelerated eligibility traces.
    """
    def __init__(self, 
                 vocab_size: int,
                 learning_rate_plus: float = 0.01, 
                 learning_rate_minus: float = 0.012,
                 time_window: int = 5,
                 w_min: float = 0.0,
                 w_max: float = 1.0,
                 decay: float = 0.99,
                 device: str = 'cuda'):
        super().__init__()
        
        self.lr_plus = learning_rate_plus
        self.lr_minus = learning_rate_minus
        self.w_min = w_min
        self.w_max = w_max
        self.decay = decay
        self.tau = time_window  # Time constant for trace decay
        
        # Dense weight vector on GPU [Vocab]
        # Much faster than a dictionary for 32k-50k vocab
        self.register_buffer('token_weights', torch.ones(vocab_size, device=device) * 0.5)
        
        # Hyperparameters for trace calculation
        self.trace_decay = torch.exp(torch.tensor(-1.0 / self.tau, device=device))

    def process_sequence(self, token_ids: torch.Tensor, spikes: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Process batch of sequences and update weights in parallel.
        
        Args:
            token_ids: [Batch, Seq_Len] int tensor
            spikes: [Batch, Seq_Len] float tensor (1.0 = spike, 0.0 = no spike).
                    If None, assumes all tokens spike (1.0).
        """
        batch_size, seq_len = token_ids.shape
        device = self.token_weights.device
        
        # Ensure inputs are on correct device
        if token_ids.device != device:
            token_ids = token_ids.to(device)
        
        if spikes is None:
            spikes = torch.ones_like(token_ids, dtype=torch.float32, device=device)
        elif spikes.device != device:
            spikes = spikes.to(device)
            
        # 1. Compute Eligibility Traces (Vectorized Recurrence)
        # trace[t] = trace[t-1] * decay + spike[t]
        # We can approximate this efficiently or iterate just over seq_len (fast on GPU)
        traces = torch.zeros_like(spikes)
        curr_trace = torch.zeros(batch_size, device=device)
        
        # Simple loop is usually fast enough for Seq_Len ~512 compared to full Python logic
        # For ultra-long sequences, a parallel scan (associative scan) would be better.
        for t in range(seq_len):
            curr_trace = curr_trace * self.trace_decay + spikes[:, t]
            traces[:, t] = curr_trace
            
        # 2. Compute Weight Updates (LTP)
        # dW = lr * (post_spike * pre_trace)
        # Here "post" is the current token event. "pre" is the history (trace).
        # We simplify: Reinforce tokens that appear during high trace activity.
        
        # delta[b, t] = lr * trace[b, t] * spike[b, t]
        # (This implements: "spike happened while recent history was active")
        updates = self.lr_plus * traces * spikes
        
        # 3. Scatter Add updates to global weights
        # We flatten the batch to apply all updates at once
        flat_tokens = token_ids.view(-1)
        flat_updates = updates.view(-1)
        
        # Accumulate updates into a buffer same size as vocab
        grad_accum = torch.zeros_like(self.token_weights)
        grad_accum.scatter_add_(0, flat_tokens, flat_updates)
        
        # 4. Apply updates
        self.token_weights += grad_accum
        
        # 5. Decay and Clamp
        # (Apply global decay occasionally or scale learning rate)
        # Here we apply decay to all weights (dense operation)
        self.token_weights *= self.decay
        self.token_weights.clamp_(self.w_min, self.w_max)
        
        return {
            "mean_weight": self.token_weights.mean().item(),
            "max_weight": self.token_weights.max().item(),
            "active_count": (self.token_weights > 0.01).sum().item()
        }

    def get_modulations(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get modulation factors for a batch of tokens (Lookup).
        Returns: [Batch, Seq_Len]
        """
        # F.embedding is essentially a fast lookup for our scalar weights
        # weights: [Vocab] -> [Vocab, 1] for embedding compat
        w = F.embedding(token_ids, self.token_weights.unsqueeze(1)).squeeze(-1)
        
        # Modulation = 1.0 + alpha * weight
        return 1.0 + (0.2 * w)

    def state_dict(self):
        return {"token_weights": self.token_weights}

    def load_state_dict(self, state):
        self.token_weights = state["token_weights"]