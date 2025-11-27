import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto

# [Keep Enums and DataClasses: NeuronalStateEnum, MaturationStage, etc. - unchanged]
class NeuronalStateEnum(Enum):
    RESTING = auto(); FIRING = auto(); REFRACTORY = auto(); FATIGUED = auto(); DREAM = auto()
    DIFFERENTIATED = auto(); PROGENITOR = auto(); MIGRATING = auto(); MYELINATED = auto()
    SYNAPTIC_PLASTICITY = auto(); LEARNING = auto(); CONSOLIDATING = auto(); SUBCONSCIOUS = auto()
    SHADOW = auto(); TRANSITORY = auto(); DEAD = auto(); UNKNOWN = auto()

class MaturationStage(Enum):
    PROGENITOR = auto(); MIGRATING = auto(); DIFFERENTIATED = auto(); MYELINATED = auto()

class ActivityState(Enum):
    RESTING = auto(); FIRING = auto(); REFRACTORY = auto()

@dataclass
class Synapse:
    target: Any; weight: float; plasticity: float = 0.0

@dataclass
class BaseNeuronConfig:
    input_dim: int; hidden_dim: int
    dt: float = 0.02; tau_min: float = 0.02; tau_max: float = 2.0
    def __post_init__(self):
        rng = np.random.default_rng(42)
        self.W_hidden = rng.normal(0, 0.1, (self.hidden_dim, self.hidden_dim))
        self.W_input = rng.normal(0, 0.1, (self.hidden_dim, self.input_dim))
        self.W_tau = rng.normal(0, 0.1, (self.hidden_dim, self.input_dim))
        self.bias = np.zeros(self.hidden_dim); self.tau_bias = np.zeros(self.hidden_dim)
        self.state = np.zeros(self.hidden_dim)

@dataclass
class NeuronalState:
    # [Keep original init arguments]
    def __init__(self, kind, position, membrane_potential=0.0, gene_expression=None, 
                 cell_cycle='G1', maturation='migrating', activity='resting',
                 connections=None, environment=None, plasticity=None,
                 W_hidden=None, W_input=None, W_tau=None, bias=None, tau_bias=None, 
                 state=None, maturation_stage=None, activity_state=None):
        self.kind = kind; self.position = position; self.membrane_potential = membrane_potential
        self.gene_expression = gene_expression or {}; self.cell_cycle = cell_cycle
        self.maturation = maturation; self.activity = activity; self.connections = connections or []
        self.environment = environment or {}; self.plasticity = plasticity or {}; self.fatigue = 0.0
        self.W_hidden = W_hidden; self.W_input = W_input; self.W_tau = W_tau
        self.bias = bias; self.tau_bias = tau_bias; self.state = state
        self.maturation_stage = maturation_stage or MaturationStage.PROGENITOR
        self.activity_state = activity_state or ActivityState.RESTING
        self.synapse = Synapse(target=None, weight=0.0)

    def update_fatigue(self, activity_level): 
        if activity_level == 'firing': self.fatigue = min(1.0, self.fatigue + 0.1)
        else: self.fatigue = max(0.0, self.fatigue - 0.01)
    def update_potential(self, input_current):
        self.membrane_potential += input_current
        if self.membrane_potential > 1.0: self.activity = 'firing'; self.membrane_potential = 0.0
        else: self.activity = 'resting'
    def differentiate(self, signals):
        if self.gene_expression.get('Neurogenin', 0) > 0.8 and signals.get('Wnt',0) > 0.7:
            self.maturation = 'differentiated'

# --- JIT-Compatible Surrogates ---

class LearnableSurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, slope):
        ctx.save_for_backward(input, slope)
        return (input > 0).to(input.dtype)
    @staticmethod
    def backward(ctx, grad_output):
        input, slope = ctx.saved_tensors
        # Fast sigmoid derivative approximation
        denom = (slope * input.abs() + 1.0) ** 2
        grad_input = grad_output / denom
        slope_grad = -2 * grad_output * input.abs() / ((slope * input.abs() + 1.0) ** 3)
        return grad_input, slope_grad.sum()

# Helper for JIT (wraps the autograd function)
def surrogate_spike(x, slope):
    return LearnableSurrogateGradient.apply(x, slope)

# --- Optimized Vectorized LIF ---

class VectorizedLIFNeuron(nn.Module):
    """Vectorized LIF with batch processing."""
    def __init__(self, size: int, beta: float = 0.5, threshold: float = 0.6, 
                 init_slope: float = 15.0, event_bus: Optional[object] = None, name: str = None):
        super().__init__()
        self.size = size
        self.name = name or "LIF"
        self._event_bus = event_bus
        self.register_buffer("beta", torch.ones(size) * beta)
        self.register_buffer("threshold", torch.ones(size) * threshold)
        # Slope as parameter to allow learning
        self.slope = nn.Parameter(torch.ones(size) * init_slope)
        self.mem = None

    def reset_mem(self):
        self.mem = None

    def forward(self, input_: torch.Tensor):
        # input_: [Batch, Size]
        if self.mem is None or self.mem.shape != input_.shape:
            self.mem = torch.zeros_like(input_)
            
        self.mem = self.beta * self.mem + input_
        spk = surrogate_spike(self.mem - self.threshold, self.slope)
        self.mem = self.mem - spk * self.threshold
        
        # Low-overhead monitoring
        if self._event_bus and spk.requires_grad is False: # Check avoids graph issues
             # Use a very simple check to avoid sync overhead
             pass 
             
        return spk, self.mem

# --- JIT Izhikevich Neuron ---

class IzhikevichNeuron(nn.Module):
    """
    JIT-Compiled Izhikevich Model.
    Supports processing full sequences in a single CUDA kernel launch.
    """
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=6.0, dt=0.2):
        super().__init__()
        self.register_buffer("a", torch.tensor(float(a)))
        self.register_buffer("b", torch.tensor(float(b)))
        self.register_buffer("c", torch.tensor(float(c)))
        self.register_buffer("d", torch.tensor(float(d)))
        self.register_buffer("dt", torch.tensor(float(dt)))
        self.v = None
        self.u = None

    def reset_state(self, batch_size: int, device: torch.device):
        self.v = torch.full((batch_size,), -65.0, device=device)
        self.u = self.b * self.v

    @torch.jit.export
    def forward_sequence(self, I_seq: torch.Tensor) -> torch.Tensor:
        """
        Process a sequence of inputs [Batch, Time] or [Batch, Time, Dim].
        Returns spikes [Batch, Time, Dim].
        """
        # Ensure state matches batch/dim
        batch_shape = I_seq.shape[0] if I_seq.dim() == 2 else I_seq.shape[0] * I_seq.shape[2]
        feature_dim = 1 if I_seq.dim() == 2 else I_seq.shape[2]
        
        # Flatten for processing: [Batch*Dim, Time] or [Batch, Time]
        if I_seq.dim() == 3:
            flat_I = I_seq.permute(0, 2, 1).reshape(-1, I_seq.shape[1])
        else:
            flat_I = I_seq
            
        if self.v is None or self.v.shape[0] != flat_I.shape[0]:
            self.v = torch.full((flat_I.shape[0],), -65.0, device=I_seq.device, dtype=I_seq.dtype)
            self.u = self.b * self.v

        # Run JIT loop
        spikes = self._jit_step_loop(flat_I, self.v, self.u, self.a, self.b, self.c, self.d, self.dt)
        
        # Update state (detach to truncate BPTT if needed, or keep for stateful RNN)
        self.v = spikes[1]
        self.u = spikes[2]
        
        # Reshape output back
        spike_out = spikes[0]
        if I_seq.dim() == 3:
            spike_out = spike_out.view(I_seq.shape[0], feature_dim, I_seq.shape[1]).permute(0, 2, 1)
            
        return spike_out

    @torch.jit.export
    def _jit_step_loop(self, I: torch.Tensor, v: torch.Tensor, u: torch.Tensor, 
                       a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, 
                       d: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Internal JIT loop."""
        T = I.shape[1]
        spikes_list = torch.jit.annotate(List[torch.Tensor], [])
        
        for t in range(T):
            i_t = I[:, t]
            
            # Update v
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + i_t
            v = v + dt * dv
            
            # Update u
            du = a * (b * v - u)
            u = u + dt * du
            
            # Spike condition
            spk = (v >= 30.0).to(v.dtype)
            
            # Reset
            v = torch.where(spk > 0.0, c, v)
            u = torch.where(spk > 0.0, u + d, u)
            
            spikes_list.append(spk)
            
        return torch.stack(spikes_list, dim=1), v, u

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        """Compat wrapper."""
        return self.forward_sequence(I)

# --- JIT AdEx Neuron ---

class AdExNeuron(nn.Module):
    """JIT-Compiled AdEx Neuron."""
    def __init__(self, C=200., g_L=10., E_L=-70., V_T=-50., Delta_T=2., 
                 tau_w=120., a=0., b=0., R=1., V_reset=-65., V_spike=30., dt=0.1):
        super().__init__()
        tau_m = C / max(1e-6, g_L)
        self.register_buffer("params", torch.tensor([
            tau_m, E_L, V_T, Delta_T, R, tau_w, a, b, V_reset, V_spike, dt
        ]))
        self.V = None; self.w = None

    def reset_state(self):
        self.V = None; self.w = None

    @torch.jit.export
    def forward_sequence(self, I_seq: torch.Tensor) -> torch.Tensor:
        if I_seq.dim() == 3:
            flat_I = I_seq.permute(0, 2, 1).reshape(-1, I_seq.shape[1])
        else:
            flat_I = I_seq
            
        if self.V is None or self.V.shape[0] != flat_I.shape[0]:
            # Initialize to E_L (params[1])
            self.V = torch.full((flat_I.shape[0],), self.params[1], device=I_seq.device, dtype=I_seq.dtype)
            self.w = torch.zeros_like(self.V)

        spikes = self._jit_step_loop(flat_I, self.V, self.w, self.params)
        
        self.V, self.w = spikes[1], spikes[2]
        spike_out = spikes[0]
        
        if I_seq.dim() == 3:
            spike_out = spike_out.view(I_seq.shape[0], I_seq.shape[2], I_seq.shape[1]).permute(0, 2, 1)
        return spike_out

    @torch.jit.export
    def _jit_step_loop(self, I: torch.Tensor, V: torch.Tensor, w: torch.Tensor, 
                       params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unpack params
        tau_m, E_L, V_T, Delta_T, R, tau_w, a, b, V_reset, V_spike, dt = (
            params[0], params[1], params[2], params[3], params[4], 
            params[5], params[6], params[7], params[8], params[9], params[10]
        )
        
        T = I.shape[1]
        spikes_list = torch.jit.annotate(List[torch.Tensor], [])
        
        for t in range(T):
            i_t = I[:, t]
            
            # dV/dt
            exp_term = Delta_T * torch.exp((V - V_T) / Delta_T)
            dV = (-(V - E_L) + exp_term - R * w + R * i_t) / tau_m
            V = V + dt * dV
            
            # dw/dt
            dw = (a * (V - E_L) - w) / tau_w
            w = w + dt * dw
            
            # Spike
            spk = (V >= V_spike).to(V.dtype)
            
            # Reset
            V = torch.where(spk > 0.0, V_reset, V)
            w = torch.where(spk > 0.0, w + b, w)
            
            spikes_list.append(spk)
            
        return torch.stack(spikes_list, dim=1), V, w

# [Legacy AdaptiveLIFNeuron kept for compatibility but Vectorized preferred]
class AdaptiveLIFNeuron(nn.Module):
    # Wrapper around VectorizedLIF for single-instance compatibility
    def __init__(self, beta=0.5, threshold=0.6, init_slope=15.0, event_bus=None, name=None):
        super().__init__()
        self.core = VectorizedLIFNeuron(1, beta, threshold, init_slope, event_bus, name)
    def forward(self, x): return self.core(x)
    def reset_mem(self): self.core.reset_mem()