import torch
import torch.nn as nn
import math

class MultiBitSurrogate(torch.autograd.Function):
    """
    Piecewise Linear Surrogate Gradient for Multi-bit Spiking.
    Forward: Returns floor(input) clipped to [0, L].
    Backward: Approximates gradient with a piecewise linear function.
    """
    @staticmethod
    def forward(ctx, input, L):
        ctx.save_for_backward(input)
        ctx.L = L
        # Quantize: floor(input)
        spikes = torch.floor(input)
        # Clip to [0, L]
        spikes = torch.clamp(spikes, 0, L)
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        L = ctx.L
        
        # Piecewise linear surrogate with dampening
        # Gradient is 1 near integer boundaries, 0 far away
        # We define "near" as within +/- 0.5 of an integer?
        # Actually, standard surrogate is usually triangular or rectangular around 0.
        # For multi-bit, we want gradients to flow when input is close to a threshold crossing.
        # Thresholds are at 1.0, 2.0, 3.0...
        # But here input is normalized V/V_th. So thresholds are integers.
        
        frac = input - torch.floor(input)
        # We want peak gradient at integer boundaries? 
        # No, usually we want gradient to flow when V is close to threshold.
        # Here, V/V_th crossing integer K means spike count changes.
        
        # Let's use a triangular window around 0.5 (midpoint) or around integers?
        # If we use floor(), the jump happens at integers.
        # So we want gradient to be non-zero around integers.
        # frac is distance from src.basesrc.coresrc.basefloor. 
        # If input is 1.9, frac is 0.9. Close to 2.0.
        # If input is 2.1, frac is 0.1. Close to 2.0.
        
        # Distance to nearest integer:
        dist = torch.abs(input - torch.round(input))
        
        # Triangular window: max 1 at integer, 0 at +/- 0.5
        grad_scale = torch.clamp(1.0 - 2.0 * dist, 0.0, 1.0)
        
        # Only flow gradients within valid range [0, L+1]
        in_range = (input >= 0.0) & (input <= L + 1.0)
        
        grad_input = grad_output * in_range.float() * grad_scale
        
        return grad_input, None

class GIFNeuron(nn.Module):
    """
    Generalized Integrate-and-Fire (GIF) Neuron (Production Grade).
    
    Enhancements Applied:
    1. Multi-bit surrogate with peaked gradients at quantization boundaries
    2. Stateless design - explicit state management via init_state()
    3. Numerical stability via membrane potential clamping
    4. SNN-aware initialization targeting ~30% firing rate
    5. Adaptive threshold (EI_IF inspired)
    6. Clear reset API with init_state() helper
    7. Bounded gradients in surrogate function
    8. Comprehensive test coverage (gradient flow, stability, etc.)
    
    Note: For deep networks, use gradient clipping in training:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    """
    def __init__(self, input_dim, hidden_dim, L=16, dt=1.0, tau=10.0, 
                 threshold=1.0, alpha=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.dt = dt
        self.tau = tau
        self.threshold = threshold
        self.alpha = alpha
        
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.decay = math.exp(-dt / tau)
        
        self.reset_parameters()

    def reset_parameters(self):
        """SNN-aware weight initialization targeting ~30% firing rate."""
        fan_in = self.input_dim
        expected_rate = 0.3
        target_V = 0.5 * self.threshold
        std = target_V / (math.sqrt(fan_in) * expected_rate)
        nn.init.normal_(self.linear.weight, 0, std)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def init_state(self, batch_size, device=None, dtype=None):
        """
        Initialize fresh state for a new sequence.
        
        Args:
            batch_size: Batch size for state tensors
            device: Device for state tensors (default: CPU)
            dtype: Data type for state tensors (default: float32)
        
        Returns:
            (v, theta): Tuple of membrane potential and threshold tensors
        """
        device = device or torch.device('cpu')
        dtype = dtype or torch.float32
        
        v = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        theta = torch.full((batch_size, self.hidden_dim), self.threshold, 
                          device=device, dtype=dtype)
        return v, theta
    
    def process_long_sequence(self, x_chunks):
        """
        Process long sequence split into chunks while maintaining state.
        
        Args:
            x_chunks: List of input tensors, each (batch, time_chunk, input_dim)
        
        Returns:
            output: Concatenated spikes (batch, total_time, hidden_dim)
            state: Final state after processing all chunks
        """
        if not x_chunks:
            raise ValueError("x_chunks cannot be empty")
        
        batch_size = x_chunks[0].shape[0]
        device = x_chunks[0].device
        dtype = x_chunks[0].dtype
        
        state = self.init_state(batch_size, device=device, dtype=dtype)
        
        outputs = []
        for chunk in x_chunks:
            out, state = self.forward(chunk, state=state)
            outputs.append(out)
        
        return torch.cat(outputs, dim=1), state

    def forward(self, x, state=None):
        """
        Forward pass.
        Args:
            x: Input tensor (Batch, Time, Input_Dim)
            state: Optional tuple (v, theta) to initialize state
        Returns:
            output: Spike tensor (Batch, Time, Hidden_Dim)
            state: Final state (v, theta)
        """
        batch_size, seq_len, _ = x.shape
        
        # Always create fresh state if not provided (Stateless design)
        if state is None:
            v = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            theta = torch.full((batch_size, self.hidden_dim), self.threshold,
                              device=x.device, dtype=x.dtype)
        else:
            v, theta = state
            
        h = self.linear(x)
        spikes_list = []
        
        for t in range(seq_len):
            i_t = h[:, t, :]
            
            # Update membrane potential
            v = v * self.decay + i_t
            
            # Numerical stability: Clamp V to prevent explosion
            # Limit to +/- 2*L*theta (enough dynamic range, but bounded)
            clamp_limit = self.L * theta * 2.0
            v = torch.clamp(v, -clamp_limit, clamp_limit)
            
            # Multi-bit spike generation
            normalized_v = v / theta
            spike = MultiBitSurrogate.apply(normalized_v, self.L)
            
            # Soft reset
            v = v - spike * theta
            
            # Threshold adaptation (EI_IF-inspired)
            if self.alpha > 0:
                # theta increases with spikes, decays back to base
                theta = theta + self.alpha * spike - self.alpha * (theta - self.threshold)
            
            spikes_list.append(spike)
            
        spikes = torch.stack(spikes_list, dim=1)
        
        return spikes, (v, theta)


class BalancedGIFNeuron(nn.Module):
    """
    Balanced GIF Neuron with Excitatory/Inhibitory populations.
    
    Architecture (Dale's Principle):
    - First 80% of neurons are excitatory (positive currents only)
    - Last 20% of neurons are inhibitory (negative currents only)
    - E neurons: indices [0, exc_dim), I neurons: [exc_dim, hidden_dim)
    
    This implements biological E/I balance for improved stability and
    sparse firing patterns.
    """
    def __init__(self, input_dim, hidden_dim, L=16, dt=1.0, tau=10.0, 
                 threshold=1.0, alpha=0.01, inhibition_ratio=0.2):
        super().__init__()
        
        # Store parameters (same as GIFNeuron)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.dt = dt
        self.tau = tau
        self.threshold = threshold
        self.alpha = alpha
        self.decay = math.exp(-dt / tau)
        
        # Calculate E/I sizes
        self.inh_ratio = inhibition_ratio
        self.exc_dim = int(hidden_dim * (1.0 - inhibition_ratio))
        self.inh_dim = hidden_dim - self.exc_dim  # Remainder goes to inhibitory
        
        # Create separate E/I layers
        self.linear_exc = nn.Linear(input_dim, self.exc_dim)
        self.linear_inh = nn.Linear(input_dim, self.inh_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # SNN-aware initialization for E/I populations
        fan_in = self.input_dim
        std = (self.threshold * 0.5) / (math.sqrt(fan_in) * 0.3)
        
        # Excitatory weights (positive bias)
        nn.init.normal_(self.linear_exc.weight, 0, std)
        if self.linear_exc.bias is not None:
            nn.init.zeros_(self.linear_exc.bias)
        
        # Inhibitory weights (typically stronger to balance)
        # Scale by 1/ratio to compensate for smaller population
        inh_scale = std * (1.0 / self.inh_ratio)
        nn.init.normal_(self.linear_inh.weight, 0, inh_scale)
        if self.linear_inh.bias is not None:
            nn.init.zeros_(self.linear_inh.bias)
    
    def init_state(self, batch_size, device=None, dtype=None):
        """
        Initialize fresh state for a new sequence.
        
        Args:
            batch_size: Batch size for state tensors
            device: Device for state tensors (default: CPU)
            dtype: Data type for state tensors (default: float32)
        
        Returns:
            (v, theta): Tuple of membrane potential and threshold tensors
        """
        device = device or torch.device('cpu')
        dtype = dtype or torch.float32
        
        v = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        theta = torch.full((batch_size, self.hidden_dim), self.threshold,
                          device=device, dtype=dtype)
        return v, theta
    
    def process_long_sequence(self, x_chunks):
        """
        Process long sequence in chunks (same as GIFNeuron).
        
        Args:
            x_chunks: List of input tensors, each (batch, time_chunk, input_dim)
        
        Returns:
            output: Concatenated spikes (batch, total_time, hidden_dim)
            state: Final state after processing all chunks
        """
        if not x_chunks:
            raise ValueError("x_chunks cannot be empty")
        
        batch_size = x_chunks[0].shape[0]
        device = x_chunks[0].device
        dtype = x_chunks[0].dtype
        
        state = self.init_state(batch_size, device=device, dtype=dtype)
        
        outputs = []
        for chunk in x_chunks:
            out, state = self.forward(chunk, state=state)
            outputs.append(out)
        
        return torch.cat(outputs, dim=1), state
    
    def forward(self, x, state=None):
        """
        Forward pass with E/I balance.
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize state for full hidden_dim
        if state is None:
            v = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            theta = torch.full((batch_size, self.hidden_dim), self.threshold,
                              device=x.device, dtype=x.dtype)
        else:
            v, theta = state
        
        # Compute E/I currents separately
        # Shape: (batch, time, exc_dim) and (batch, time, inh_dim)
        h_exc = self.linear_exc(x)
        h_inh = self.linear_inh(x)
        
        spikes_list = []
        
        for t in range(seq_len):
            # Excitatory input (positive only)
            i_exc = torch.nn.functional.relu(h_exc[:, t, :])
            
            # Inhibitory input (negative only)
            i_inh = -torch.nn.functional.relu(h_inh[:, t, :])
            
            # Combine E and I currents
            # Concatenate along the neuron dimension
            i_t = torch.cat([i_exc, i_inh], dim=1)
            
            # Update membrane potential
            v = v * self.decay + i_t
            
            # Numerical stability
            clamp_limit = self.L * theta * 2.0
            v = torch.clamp(v, -clamp_limit, clamp_limit)
            
            # Multi-bit spike generation
            normalized_v = v / theta
            spike = MultiBitSurrogate.apply(normalized_v, self.L)
            
            # Soft reset
            v = v - spike * theta
            
            # Threshold adaptation
            if self.alpha > 0:
                theta = theta + self.alpha * spike - self.alpha * (theta - self.threshold)
            
            spikes_list.append(spike)
        
        spikes = torch.stack(spikes_list, dim=1)
        
        return spikes, (v, theta)

