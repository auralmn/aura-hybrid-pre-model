import torch
import torch.nn as nn
import math


class Synapsis(nn.Module):
    """
    Synaptic layer for spike-driven transformations in SNNs.
    
    Features:
    - Efficient spike-based linear operation: y = W · s + b
    - Optional STDP-based plasticity for online learning
    - Stateless design (consistent with GIF Neuron)
    - SNN-aware initialization for stable spiking dynamics
    
    Args:
        in_features: Input spike dimension
        out_features: Output current dimension
        enable_plasticity: Enable STDP-like weight updates
        stdp_lr: Learning rate for STDP (if plasticity enabled)
        trace_decay: Exponential decay for spike traces (if plasticity enabled)
        target_firing_rate: Expected input firing rate for initialization
        
    Example:
        >>> syn = Synapsis(128, 256, enable_plasticity=False)
        >>> spikes = torch.randint(0, 2, (4, 100, 128), dtype=torch.float32)
        >>> currents, state = syn(spikes, state=None)
        >>> currents.shape
        torch.Size([4, 100, 256])
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        enable_plasticity=False,
        stdp_lr=0.001,
        trace_decay=0.95,
        target_firing_rate=0.3,
        bias=True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.enable_plasticity = enable_plasticity
        self.stdp_lr = stdp_lr
        self.trace_decay = trace_decay
        self.target_firing_rate = target_firing_rate
        
        # Synaptic weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # STDP eligibility traces (not parameters, just buffers for tracking)
        if self.enable_plasticity:
            # These will be managed as state, not persistent buffers
            pass
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        SNN-aware initialization.
        
        Weights are scaled based on:
        - Fan-in (number of input connections)
        - Expected firing rate (sparsity of spikes)
        
        Goal: Keep postsynaptic currents in reasonable range for downstream neurons.
        """
        # Standard deviation scaled for spiking regime
        # Lower firing rate → need larger weights to compensate for sparsity
        std = 1.0 / math.sqrt(self.in_features * self.target_firing_rate)
        
        nn.init.normal_(self.weight, mean=0.0, std=std)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, spikes, state=None):
        """
        Compute synaptic currents from input spikes.
        
        Args:
            spikes: Input spike tensor (batch, time, in_features)
                   Can be binary {0,1} or continuous (e.g., from GIF multi-bit)
            state: Optional plasticity state (pre_trace, post_trace)
                   Only used if enable_plasticity=True
        
        Returns:
            currents: Output synaptic currents (batch, time, out_features)
            new_state: Updated plasticity state (or None if plasticity disabled)
        """
        batch_size, seq_len, _ = spikes.shape
        
        # Initialize plasticity state if needed
        if self.enable_plasticity and state is None:
            pre_trace = torch.zeros(batch_size, self.in_features, 
                                   device=spikes.device, dtype=spikes.dtype)
            post_trace = torch.zeros(batch_size, self.out_features,
                                    device=spikes.device, dtype=spikes.dtype)
            state = (pre_trace, post_trace)
        
        # Efficient computation: single matmul across time dimension
        # Reshape: (B, T, in) → (B*T, in)
        spikes_flat = spikes.reshape(batch_size * seq_len, self.in_features)
        
        # Linear transform: (B*T, in) @ (out, in)^T → (B*T, out)
        currents_flat = torch.nn.functional.linear(spikes_flat, self.weight, self.bias)
        
        # Reshape back: (B*T, out) → (B, T, out)
        currents = currents_flat.reshape(batch_size, seq_len, self.out_features)
        
        # Update plasticity traces if enabled
        if self.enable_plasticity:
            new_state = self._update_traces(spikes, currents, state)
        else:
            new_state = None
        
        return currents, new_state
    
    def _update_traces(self, pre_spikes, post_currents, state):
        """
        Update STDP eligibility traces (experimental).
        
        Standard STDP rule:
        - Pre-before-post: Potentiation (LTP)
        - Post-before-pre: Depression (LTD)
        
        Traces track recent spiking history with exponential decay.
        
        Args:
            pre_spikes: Presynaptic spikes (batch, time, in_features)
            post_currents: Postsynaptic currents (batch, time, out_features)
            state: Current (pre_trace, post_trace)
        
        Returns:
            new_state: Updated (pre_trace, post_trace)
        """
        pre_trace, post_trace = state
        batch_size, seq_len, _ = pre_spikes.shape
        
        # Accumulate traces over time
        # Simplified: use mean over time as proxy for temporal integration
        # Full STDP would require timestep-by-timestep update
        
        # Average spike activity over sequence
        pre_activity = pre_spikes.mean(dim=1)  # (batch, in_features)
        post_activity = post_currents.mean(dim=1)  # (batch, out_features)
        
        # Update traces with decay
        pre_trace = self.trace_decay * pre_trace + (1 - self.trace_decay) * pre_activity
        post_trace = self.trace_decay * post_trace + (1 - self.trace_decay) * post_activity
        
        # Note: Actual weight updates would happen in training loop
        # using correlation between pre_trace and post_trace
        # This just maintains the traces for external use
        
        return (pre_trace, post_trace)
    
    def apply_stdp_update(self, pre_trace, post_trace):
        """
        Apply STDP weight update (experimental, call from src.training loop).
        
        Args:
            pre_trace: Presynaptic trace (batch, in_features)
            post_trace: Postsynaptic trace (batch, out_features)
        
        This method should be called externally after forward pass if you want
        to apply STDP updates. Example:
        
            currents, state = syn(spikes, state)
            if state is not None:
                pre_trace, post_trace = state
                syn.apply_stdp_update(pre_trace, post_trace)
        """
        if not self.enable_plasticity:
            return
        
        # Batch average for stability
        pre_trace_avg = pre_trace.mean(dim=0)  # (in_features,)
        post_trace_avg = post_trace.mean(dim=0)  # (out_features,)
        
        # STDP update: ΔW = η * (post ⊗ pre)
        # Outer product: (out, 1) × (1, in) → (out, in)
        dw = self.stdp_lr * torch.outer(post_trace_avg, pre_trace_avg)
        
        # Update weights (no_grad to avoid affecting autograd)
        with torch.no_grad():
            self.weight.data += dw
            
            # Optional: weight clipping to prevent runaway
            self.weight.data.clamp_(-10.0, 10.0)
    
    def extra_repr(self):
        """String representation for debugging."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'plasticity={self.enable_plasticity}, '
                f'bias={self.bias is not None}')
