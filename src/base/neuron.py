from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

class NeuronalStateEnum(Enum):
    RESTING = auto()         # Baseline polarization, ready to fire
    FIRING = auto()          # Action potential, spiking
    REFRACTORY = auto()      # Temporarily unable to fire after spike
    FATIGUED = auto()        # Reduced excitability due to recent activity
    DREAM = auto()           # Offline, restoration, or concept discovery
    DIFFERENTIATED = auto()  # Mature, specialized function
    PROGENITOR = auto()      # Immature, dividing/neurogenic
    MIGRATING = auto()       # Moving to final location/layer
    MYELINATED = auto()      # With myelin, rapid conduction
    SYNAPTIC_PLASTICITY = auto() # Actively remodeling synapses
    LEARNING = auto()        # Current learning/adaptation phase
    CONSOLIDATING = auto()   # Solidifying traces, LTM formation
    SUBCONSCIOUS = auto()    # Involved in implicit/gated processes
    SHADOW = auto()          # Backup trace, not directly involved
    TRANSITORY = auto()      # Temporary or short-lived activation
    DEAD = auto()            # Cell death or pruned (for completeness)
    UNKNOWN = auto()         # Catch-all for undefined state

class MaturationStage(Enum):
    PROGENITOR = auto()
    MIGRATING = auto()
    DIFFERENTIATED = auto()
    MYELINATED = auto()

class ActivityState(Enum):
    RESTING = auto()
    FIRING = auto()
    REFRACTORY = auto()

from dataclasses import dataclass
from typing import List, Optional








@dataclass
class NeuronalState:
    def __init__(self, kind, position, membrane_potential=0.0, 
                 gene_expression=None, cell_cycle='G1', 
                 maturation='migrating', activity='resting',
                 connections=None, environment=None, plasticity=None,
                 W_hidden=None, W_input=None, W_tau=None, bias=None, tau_bias=None, state=None, maturation_stage=None, activity_state=None):
        self.kind = kind
        self.position = position          # Spatial coordinates, e.g., (x, y, z)
        self.membrane_potential = membrane_potential
        self.gene_expression = gene_expression or {}   # {'Neurogenin': 0.8, ...}
        self.cell_cycle = cell_cycle      # 'G1', 'S', 'G2', 'M'
        self.maturation = maturation      # 'progenitor', 'migrating', etc.
        self.activity = activity          # 'resting', 'firing', etc.
        self.connections = connections or []
        self.environment = environment or {}    # {'BDNF': 0.3, 'Wnt': 0.5, ...}
        self.plasticity = plasticity or {}      # LTP/LTD/STDP traces
        self.fatigue = 0.0
        self.W_hidden = W_hidden
        self.W_input = W_input
        self.W_tau = W_tau
        self.bias = bias
        self.tau_bias = tau_bias
        self.state = state
        self.maturation_stage = maturation_stage or MaturationStage.PROGENITOR
        self.activity_state = activity_state or ActivityState.RESTING
        self.synapse = Synapse(target=None, weight=0.0)
        
   
    def update_fatigue(self, activity_level): 
        # Increase with activity, recover if resting
        if activity_level == 'firing':
            self.fatigue = min(1.0, self.fatigue + 0.1)
        else:
            self.fatigue = max(0.0, self.fatigue - 0.01)

    def update_potential(self, input_current):
        # Simple integrate-and-fire model as example
        self.membrane_potential += input_current
        if self.membrane_potential > 1.0:
            self.activity = 'firing'
            self.membrane_potential = 0.0
        else:
            self.activity = 'resting'
    
    def differentiate(self, signals):
        # Change neuronal state based on gene expression and signals
        if self.gene_expression.get('Neurogenin', 0) > 0.8 and signals.get('Wnt',0) > 0.7:
            self.maturation = 'differentiated'



    

@dataclass
class Synapse:
    target: Any
    weight: float
    plasticity: float = 0.0

@dataclass
class BaseNeuronConfig:
    input_dim: int
    hidden_dim: int
    dt: float = 0.02
    tau_min: float = 0.02
    tau_max: float = 2.0
    
    def __post_init__(self):
        """Initialize neuron parameters"""
        rng = np.random.default_rng(42)
        
        # Weight matrices
        self.W_hidden = rng.normal(0, 0.1, (self.hidden_dim, self.hidden_dim))
        self.W_input = rng.normal(0, 0.1, (self.hidden_dim, self.input_dim))
        self.W_tau = rng.normal(0, 0.1, (self.hidden_dim, self.input_dim))
        
        # Biases
        self.bias = np.zeros(self.hidden_dim)
        self.tau_bias = np.zeros(self.hidden_dim)
        
        # State
        self.state = np.zeros(self.hidden_dim)


@dataclass
class Neuron:
    zone: str
    type: str
    structure: str
    neurotransmitter: str
    percentage: Optional[float] = None
    state: Optional[NeuronalState] = None


pyramidal = Neuron(
    zone="cerebral cortex",
    type="pyramidal",
    structure="multipolar, triangular soma, apical and basal dendrites",
    neurotransmitter="glutamate",
    percentage=75.0
)

granule = Neuron(
    zone="cerebellum",
    type="granule",
    structure="small, multipolar",
    neurotransmitter="glutamate",
    percentage=75.0
)

purkinje = Neuron(
    zone="cerebellum",
    type="Purkinje",
    structure="large, pear-shaped, extensive dendritic arbor",
    neurotransmitter="GABA",
    percentage=0.5
)

# ---------------------------------------------------------------------------------
# Torch-based adaptive LIF neuron with learnable surrogate gradient and monitoring
# ---------------------------------------------------------------------------------

import torch
import torch.nn as nn


class LearnableSurrogateGradient(torch.autograd.Function):
    """Learnable fast-sigmoid surrogate gradient with adaptive slope parameter.

    Forward: binary step. Backward: smooth surrogate controlled by slope.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input, slope)
        return (input > 0).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, slope = ctx.saved_tensors
        denom = (slope * input.abs() + 1.0) ** 2
        grad_input = grad_output / denom
        slope_grad = -2 * grad_output * input.abs() / ((slope * input.abs() + 1.0) ** 3)
        # Sum gradient w.r.t. scalar/parameter slope
        return grad_input, slope_grad.sum()


class AdaptiveLIFNeuron(nn.Module):
    """Leaky-Integrate-and-Fire neuron with learnable surrogate slope.

    Optionally emits 'neuron_fired' events via an injected event bus that
    provides a `broadcast_neuron_fired(dict)` method.
    """

    def __init__(self, beta: float = 0.5, threshold: float = 0.6, init_slope: float = 15.0, event_bus: Optional[object] = None, name: Optional[str] = None):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
        self.slope = nn.Parameter(torch.tensor(float(init_slope)))
        self._event_bus = event_bus
        self.name = name or self.__class__.__name__
        self.register_buffer("_mem_initialized", torch.tensor(False), persistent=False)
        self.mem: Optional[torch.Tensor] = None

    def reset_mem(self) -> None:
        self.mem = None
        if hasattr(self, "_mem_initialized"):
            self._mem_initialized.zero_()

    def forward(self, input_: torch.Tensor):
        if self.mem is None:
            self.mem = torch.zeros_like(input_)
            if hasattr(self, "_mem_initialized"):
                self._mem_initialized.fill_(True)

        # LIF dynamics
        self.mem = self.beta * self.mem + input_
        spk = LearnableSurrogateGradient.apply(self.mem - self.threshold, self.slope)
        self.mem = self.mem - spk * self.threshold  # reset on spike

        # Optional observability via event bus
        if self._event_bus is not None:
            try:
                spikes = spk.detach()
                mem_det = self.mem.detach()
                mem_std = float(mem_det.std(unbiased=False).item()) if mem_det.numel() > 1 else 0.0
                details: Dict[str, Any] = {
                    "layer": self.name,
                    "firing_rate": float(min(max(spikes.float().mean().item(), 0.0), 0.999)),
                    "surrogate_slope": float(self.slope.detach().item()),
                    "mem_mean": float(mem_det.mean().item()),
                    "mem_std": mem_std,
                }
                # Avoid hard dependency on EventBus type
                if hasattr(self._event_bus, "broadcast_neuron_fired"):
                    self._event_bus.broadcast_neuron_fired(details)
            except Exception:
                # Non-fatal if monitoring fails
                pass

        return spk, self.mem


class VectorizedLIFNeuron(nn.Module):
    """Vectorized Leaky-Integrate-and-Fire layer.

    Manages a layer of neurons in parallel using tensor operations.
    """

    def __init__(self, size: int, beta: float = 0.5, threshold: float = 0.6, 
                 init_slope: float = 15.0, event_bus: Optional[object] = None, name: Optional[str] = None):
        super().__init__()
        self.size = size
        self.name = name or self.__class__.__name__
        self._event_bus = event_bus

        # Parameters per neuron [size]
        self.beta = nn.Parameter(torch.ones(size) * float(beta))
        self.threshold = nn.Parameter(torch.ones(size) * float(threshold))
        self.slope = nn.Parameter(torch.ones(size) * float(init_slope))

        # State
        self.register_buffer("_mem_initialized", torch.tensor(False), persistent=False)
        self.mem: Optional[torch.Tensor] = None

    def reset_mem(self) -> None:
        self.mem = None
        if hasattr(self, "_mem_initialized"):
            self._mem_initialized.zero_()

    def forward(self, input_: torch.Tensor):
        # input_ shape: [batch_size, size]
        if self.mem is None or self.mem.shape != input_.shape:
            self.mem = torch.zeros_like(input_)
            if hasattr(self, "_mem_initialized"):
                self._mem_initialized.fill_(True)

        # LIF dynamics (vectorized)
        # mem = beta * mem + input
        self.mem = self.beta * self.mem + input_
        
        # Spike generation
        spk = LearnableSurrogateGradient.apply(self.mem - self.threshold, self.slope)
        
        # Reset mechanism: soft reset (subtract threshold)
        self.mem = self.mem - spk * self.threshold

        # Optional observability (sampled or aggregated to avoid overhead)
        if self._event_bus is not None:
            try:
                # Only broadcast if explicitly enabled/subscribed to avoid sync overhead
                if hasattr(self._event_bus, "should_broadcast") and self._event_bus.should_broadcast("neuron_fired"):
                    rate = float(spk.detach().float().mean().item())
                    if rate > 0.001:  # Filter silence
                        details = {
                            "layer": self.name,
                            "firing_rate": rate,
                            "mem_mean": float(self.mem.detach().mean().item()),
                        }
                        self._event_bus.broadcast_neuron_fired(details)
            except Exception:
                pass

        return spk, self.mem

class TrainingMonitor:
    """Monitor training stability metrics for torch models."""

    def __init__(self):
        self.gradient_norms = {"layer": [], "total": []}
        self.firing_rates: List[float] = []
        self.membrane_stats = {"mean": [], "std": [], "min": [], "max": []}
        self.surrogate_slopes: List[List[float]] = []
        self.losses: List[float] = []

    def log_gradient_norms(self, model: nn.Module, step: int) -> None:
        total_sq = 0.0
        layer_norms: Dict[str, float] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gn = float(param.grad.norm().item())
                    layer_norms[name] = gn
                    total_sq += gn ** 2
            self.gradient_norms["layer"].append(layer_norms)
            self.gradient_norms["total"].append(total_sq ** 0.5)

    def log_firing_rate(self, spike_data: torch.Tensor) -> None:
        self.firing_rates.append(float(spike_data.float().mean().item()))

    def log_membrane_stats(self, membrane_data: torch.Tensor) -> None:
        with torch.no_grad():
            self.membrane_stats["mean"].append(float(membrane_data.mean().item()))
            self.membrane_stats["std"].append(float(membrane_data.std().item()))
            self.membrane_stats["min"].append(float(membrane_data.min().item()))
            self.membrane_stats["max"].append(float(membrane_data.max().item()))

    def log_surrogate_slopes(self, model: nn.Module) -> None:
        slopes: List[float] = []
        for module in model.modules():
            if isinstance(module, AdaptiveLIFNeuron):
                slopes.append(float(module.slope.detach().item()))
        self.surrogate_slopes.append(slopes)

    def check_stability(self, tolerance: float = 1e-3) -> str:
        if len(self.gradient_norms["total"]) > 10:
            recent = self.gradient_norms["total"][-10:]
            if max(recent) > 100.0:
                return "exploding"
            if max(recent) < tolerance:
                return "vanishing"
        return "stable"


class AdaptiveSNN(nn.Module):
    """Example SNN using AdaptiveLIFNeuron with learnable surrogates."""

    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, event_bus: Optional[object] = None):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = AdaptiveLIFNeuron(beta=0.5, threshold=0.6, init_slope=15.0, event_bus=event_bus, name="lif1")
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = AdaptiveLIFNeuron(beta=0.5, threshold=0.6, init_slope=15.0, event_bus=event_bus, name="lif2")
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = AdaptiveLIFNeuron(beta=0.5, threshold=0.6, init_slope=15.0, event_bus=event_bus, name="lif3")

    def forward(self, x: torch.Tensor):
        # Reset membrane potentials at start of sequence
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()

        spk_rec: List[torch.Tensor] = []
        mem_rec: List[List[torch.Tensor]] = []

        for t in range(x.size(0)):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3)

            spk_rec.append(spk3)
            mem_rec.append([mem1, mem2, mem3])

        return torch.stack(spk_rec), mem_rec


# ---------------------------------------------------------------------------------
# Izhikevich spiking neuron (efficient 2-variable model)
# ---------------------------------------------------------------------------------

class IzhikevichNeuron(nn.Module):
    """Izhikevich spiking neuron model.

    dv/dt = 0.04 v^2 + 5 v + 140 - u + I
    du/dt = a (b v - u)
    When v >= 30: spike, then v <- c, u <- u + d

    Parameters a,b,c,d follow Izhikevich presets. Default dt = 0.2 ms (paper).
    """

    def __init__(self, a: float = 0.02, b: float = 0.2, c: float = 0.06, d: float = 6.0, dt: float = 0.2):
        super().__init__()
        self.register_buffer("a", torch.tensor(float(a)))
        self.register_buffer("b", torch.tensor(float(b)))
        self.register_buffer("c", torch.tensor(float(c)))
        self.register_buffer("d", torch.tensor(float(d)))
        self.register_buffer("dt", torch.tensor(float(dt)))
        # State variables
        self.register_buffer("v", torch.tensor(0.06))      
        self.register_buffer("u", (self.b * self.v).clone().detach())

    def reset_state(self) -> None:
        self.v = torch.tensor(0.06, device=self.v.device, dtype=self.v.dtype)
        self.u = self.b * self.v

    def step(self, I: torch.Tensor) -> torch.Tensor:
        """Advance one time step for given current I (broadcastable). Returns spike tensor (0/1)."""
        v = self.v
        u = self.u
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
        du = self.a * (self.b * v - u)
        v = v + self.dt * dv
        u = u + self.dt * du
        spk = (v >= 30.0).to(v.dtype)
        # Reset on spike
        v = torch.where(spk > 0, self.c.to(v.dtype), v)
        u = torch.where(spk > 0, u + self.d.to(u.dtype), u)
        # Store
        self.v = v
        self.u = u
        return spk

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        """Simulate over time.

        I shapes supported:
          - [T] scalar current per step
          - [T, N] current per step for N parallel units (broadcasts state)
        Returns spikes with same leading shape.
        """
        if I.dim() == 0:
            I = I.view(1)
        T = I.shape[0]
        spikes = []
        for t in range(T):
            spk = self.step(I[t])
            spikes.append(spk)
        return torch.stack(spikes, dim=0)


def load_izhikevich_presets(csv_path: str = "pattern.csv") -> Dict[str, Dict[str, float]]:
    """Load Izhikevich parameter presets from a CSV with an 'Izhikevich_Parameters' column.

    Returns a dict keyed by both 'Primary_Pattern' and 'Neuron_Type' (lowercased),
    mapping to {'a':..,'b':..,'c':..,'d':..}.
    """
    import csv
    presets: Dict[str, Dict[str, float]] = {}
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param_str = row.get("Izhikevich_Parameters", "")
                if not param_str:
                    continue
                parts = {p.split("=")[0].strip(): float(p.split("=")[1]) for p in param_str.split(",") if "=" in p}
                for key in (row.get("Primary_Pattern"), row.get("Neuron_Type")):
                    if key and all(k in parts for k in ("a","b","c","d")):
                        presets[key.strip().lower()] = {k: parts[k] for k in ("a","b","c","d")}
    except Exception:
        pass
    return presets


# ---------------------------------------------------------------------------------
# Izhikevich JSON presets integration (23 canonical firing patterns)
# ---------------------------------------------------------------------------------

def load_izhikevich_patterns_json(json_path: str) -> Dict[str, Dict[str, float]]:
    """Load Izhikevich patterns from a JSON file into a normalized dict.

    Supports multiple JSON layouts:
      1) Top-level list of pattern objects with fields: name/pattern, a,b,c,d, optional I
      2) Dict with key 'patterns' as a list (same as 1)
      3) Dict with key 'patterns' as a dict mapping ids -> { name, parameters: {a,b,c,d}, ... }
      4) Dict wrapped as {'izhikevich_23_firing_patterns': { 'patterns': { ... } }}

    Returns: { name_lower: { 'a': float, 'b': float, 'c': float, 'd': float, 'I': optional float } }
    """
    import json

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = json.load(f)

    # Unwrap known container
    if isinstance(raw, dict) and 'izhikevich_23_firing_patterns' in raw:
        raw = raw['izhikevich_23_firing_patterns']

    # Normalize the container holding the patterns
    container = raw
    if isinstance(container, dict):
        container = container.get('patterns', container.get('items', container))

    patterns: Dict[str, Dict[str, float]] = {}

    # Case A: container is already a list of pattern dicts
    if isinstance(container, list):
        iterable = container
        for item in iterable:
            if not isinstance(item, dict):
                continue
            name = item.get('name') or item.get('pattern') or item.get('Primary_Pattern') or 'unknown'
            a = _to_float(item.get('a'))
            b = _to_float(item.get('b'))
            c = _to_float(item.get('c'))
            d = _to_float(item.get('d'))
            if None in (a, b, c, d):
                # Some schemas nest parameters
                params = item.get('parameters') or {}
                a = a if a is not None else _to_float(params.get('a'))
                b = b if b is not None else _to_float(params.get('b'))
                c = c if c is not None else _to_float(params.get('c'))
                d = d if d is not None else _to_float(params.get('d'))
            if None in (a, b, c, d):
                continue
            I = item.get('I')
            if I is None:
                for k in ('Iext', 'current', 'input', 'I_bg', 'Ibg'):
                    if k in item:
                        I = item[k]
                        break
            I = _to_float(I) if I is not None else None
            entry: Dict[str, float] = {'a': a, 'b': b, 'c': c, 'd': d}
            if I is not None:
                entry['I'] = I
            patterns[str(name).strip().lower()] = entry
        return patterns

    # Case B: container is a dict mapping ids -> pattern dicts
    if isinstance(container, dict):
        for key, item in container.items():
            if not isinstance(item, dict):
                continue
            name = item.get('name') or item.get('pattern') or key
            params = item.get('parameters') if isinstance(item.get('parameters'), dict) else item
            a = _to_float(params.get('a'))
            b = _to_float(params.get('b'))
            c = _to_float(params.get('c'))
            d = _to_float(params.get('d'))
            if None in (a, b, c, d):
                continue
            # Optional drive current may sit at item-level
            I = item.get('I')
            if I is None:
                for k in ('Iext', 'current', 'input', 'I_bg', 'Ibg'):
                    if k in item:
                        I = item[k]
                        break
            I = _to_float(I) if I is not None else None
            entry: Dict[str, float] = {'a': a, 'b': b, 'c': c, 'd': d}
            if I is not None:
                entry['I'] = I
            patterns[str(name).strip().lower()] = entry
        return patterns

    # Fallback: unsupported structure
    return patterns


def create_izhikevich_from_pattern(pattern_name: str, patterns: Dict[str, Dict[str, float]], dt: float = 0.2) -> IzhikevichNeuron:
    """Instantiate an IzhikevichNeuron from a loaded patterns dict by name (case-insensitive)."""
    key = pattern_name.strip().lower()
    p = patterns.get(key)
    if not p:
        raise KeyError(f"Pattern '{pattern_name}' not found in patterns")
    return IzhikevichNeuron(a=p['a'], b=p['b'], c=p['c'], d=p['d'], dt=dt)


def simulate_izhikevich(neuron: IzhikevichNeuron, T: int = 200, I: float | None = None) -> torch.Tensor:
    """Run a short simulation and return spikes [T]. If I is None, uses zero current."""
    current = torch.full((int(T),), float(I) if I is not None else 0.0, device=neuron.v.device, dtype=neuron.v.dtype)
    return neuron(current)


# ---------------------------------------------------------------------------------
# Adaptive Exponential Integrate-and-Fire (AdEx) minimal implementation
# ---------------------------------------------------------------------------------

class AdExNeuron(nn.Module):
    """Vectorized AdEx neuron with simple Euler integration.

    tau_m * dV/dt = -(V - E_L) + Delta_T*exp((V - V_T)/Delta_T) - R*w + R*I
    tau_w * dw/dt = a*(V - E_L) - w
    Spike/reset: when V >= V_spike (~30 mV), set V = V_reset; w += b
    """

    def __init__(self, C: float = 200.0, g_L: float = 10.0, E_L: float = -70.0,
                 V_T: float = 0.6, Delta_T: float = 2.0, tau_w: float = 120.0,
                 a: float = 0.0, b: float = 0.0, R: float = 1.0,
                 V_reset: float = -65.0, V_spike: float = 30.0, dt: float = 0.1):
        super().__init__()
        # Use tau_m = C/g_L; keep R explicit (input scaling)
        tau_m = (C / max(1e-6, g_L))
        self.register_buffer("tau_m", torch.tensor(float(tau_m)))
        self.register_buffer("g_L", torch.tensor(float(g_L)))
        self.register_buffer("E_L", torch.tensor(float(E_L)))
        self.register_buffer("V_T", torch.tensor(float(V_T)))
        self.register_buffer("Delta_T", torch.tensor(float(Delta_T)))
        self.register_buffer("tau_w", torch.tensor(float(tau_w)))
        self.register_buffer("a", torch.tensor(float(a)))
        self.register_buffer("b", torch.tensor(float(b)))
        self.register_buffer("R", torch.tensor(float(R)))
        self.register_buffer("V_reset", torch.tensor(float(V_reset)))
        self.register_buffer("V_spike", torch.tensor(float(V_spike)))
        self.register_buffer("dt", torch.tensor(float(dt)))
        # State
        self.register_buffer("V", torch.tensor(float(E_L)))
        self.register_buffer("w", torch.tensor(0.0))

    def reset_state(self) -> None:
        self.V = self.E_L.clone()
        self.w = torch.tensor(0.0, device=self.V.device, dtype=self.V.dtype)

    def step(self, I: torch.Tensor) -> torch.Tensor:
        V = self.V
        w = self.w
        dt = self.dt
        # dV/dt
        exp_term = self.Delta_T * torch.exp((V - self.V_T) / self.Delta_T)
        dV = (-(V - self.E_L) + exp_term - self.R * w + self.R * I) / self.tau_m
        V = V + dt * dV
        # dw/dt
        dw = (self.a * (V - self.E_L) - w) / self.tau_w
        w = w + dt * dw
        spikes = (V >= self.V_spike).to(V.dtype)
        V = torch.where(spikes > 0, self.V_reset.to(V.dtype), V)
        w = torch.where(spikes > 0, w + self.b.to(w.dtype), w)
        self.V, self.w = V, w
        return spikes

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        if I.dim() == 0:
            I = I.view(1)
        T = I.shape[0]
        out = []
        for t in range(T):
            out.append(self.step(I[t]))
        return torch.stack(out, dim=0)