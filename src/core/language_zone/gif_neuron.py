import torch
import torch.nn as nn
import math

class MultiBitSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, L):
        ctx.save_for_backward(input)
        ctx.L = L
        spikes = torch.floor(input)
        spikes = torch.clamp(spikes, 0, L)
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        L = ctx.L
        dist = torch.abs(input - torch.round(input))
        grad_scale = torch.clamp(1.0 - 2.0 * dist, 0.0, 1.0)
        in_range = (input >= 0.0) & (input <= L + 1.0)
        return grad_output * in_range.float() * grad_scale, None

class GIFNeuron(nn.Module):
    """
    Stateless GIF Neuron for Language Zone.
    """
    def __init__(self, input_dim, hidden_dim, L=16, dt=1.0, tau=10.0, 
                 threshold=1.0, alpha=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.decay = math.exp(-dt / tau)
        self.threshold = threshold
        self.alpha = alpha
        
    def reset_state(self):
        pass

    def forward(self, x: torch.Tensor, state=None) -> Tuple[torch.Tensor, Any]:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Always initialize fresh state if not provided
        if state is None:
            v = torch.zeros(batch_size, self.hidden_dim, device=device)
            theta = torch.full((batch_size, self.hidden_dim), self.threshold, device=device)
        else:
            v, theta = state
            
        h = self.linear(x)
        spikes_list = []
        
        for t in range(seq_len):
            i_t = h[:, t, :]
            v = v * self.decay + i_t
            
            # Clamp
            clamp_limit = self.L * theta * 2.0
            v = torch.clamp(v, -clamp_limit, clamp_limit)
            
            # Spike
            normalized_v = v / (theta + 1e-6)
            spike = MultiBitSurrogate.apply(normalized_v, self.L)
            
            # Reset
            v = v - spike * theta
            
            # Adapt threshold
            if self.alpha > 0:
                theta = theta + self.alpha * spike - self.alpha * (theta - self.threshold)
                
            spikes_list.append(spike)
            
        return torch.stack(spikes_list, dim=1), (v, theta)