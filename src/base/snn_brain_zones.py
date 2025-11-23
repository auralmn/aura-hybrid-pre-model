#!/usr/bin/env python3
"""
Enhanced brain_zones.py with neuromorphic spiking capabilities
Builds on your existing BrainZone architecture
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import os
from enum import Enum
import torch
import torch.nn as nn

from base.layers import BaseLayer, BaseLayerConfig, BaseLayerContainerConfig
from maths.addition_linear import AdditionLinear
from maths.additive_receptance import AdditiveReceptance
from base.neuron import AdaptiveLIFNeuron, LearnableSurrogateGradient, NeuronalState, IzhikevichNeuron, AdExNeuron, load_izhikevich_patterns_json
from base.events import EventBus

import csv

class BrainZoneType(Enum):
    """Types of brain zones for specialized processing"""
    PREFRONTAL_CORTEX = "prefrontal_cortex"
    TEMPORAL_CORTEX = "temporal_cortex"
    HIPPOCAMPUS = "hippocampus"
    CEREBELLUM = "cerebellum"
    THALAMUS = "thalamus"
    AMYGDALA = "amygdala"
    BASAL_GANGLIA = "basal_ganglia"
    BRAINSTEM = "brainstem"
    OCCIPITAL_CORTEX = "occipital_cortex"
    PARIETAL_CORTEX = "parietal_cortex"
    INSULAR_CORTEX = "insular_cortex"

@dataclass
class SpikingNeuronConfig:
    """Configuration for spiking neuron properties per brain region"""
    neuron_type: str
    structure: str
    neurotransmitter: str
    percentage: float
    threshold: float = 0.6
    membrane_time_constant: float = 10.0
    reset_potential: float = 0.0
    refractory_period: float = 2.0
    init_surrogate_slope: float = 15.0
    beta_decay: float = 0.95
    learning_rate_modifier: float = 0.5
    # Optional Izhikevich parameters (when provided by pattern data)
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    d: Optional[float] = None
    i_bg: Optional[float] = None
    dt: float = 0.2
    # Optional model selection and parameters (e.g., AdEx)
    model_type: Optional[str] = None  # 'izhikevich' | 'adex' | None
    model_params: Optional[Dict[str, float]] = None

@dataclass
class BrainZoneConfig:
    name: str = ""
    max_neurons: int = 1024
    min_neurons: int = 256
    neuron_type: str = "liquid"
    gated: bool = False
    num_layers: int = 2
    base_layer_container_config: BaseLayerContainerConfig = None
    
    # Enhanced neuromorphic properties
    zone_type: Optional[BrainZoneType] = None
    d_model: int = 1024 
    use_spiking: bool = True
    spiking_configs: Optional[List[SpikingNeuronConfig]] = None
    content_specialization: Optional[List[str]] = None  # Content types this zone handles
    zone_weights: Dict[str, float] = None  # Weights for different processing types
    event_bus: Optional[EventBus] = None

    def __post_init__(self):
        """Initialize enhanced configurations"""
        if self.zone_weights is None:
            self.zone_weights = {}
        
        if self.spiking_configs is None and self.zone_type:
            self.spiking_configs = self._get_default_spiking_configs()
    
    def _get_default_spiking_configs(self) -> List[SpikingNeuronConfig]:
        """Get default spiking neuron configurations based on zone type"""
        configs = {
            BrainZoneType.PREFRONTAL_CORTEX: [
                SpikingNeuronConfig("pyramidal_executive", "reasoning neurons", "glutamate", 75.0,
                                  threshold=0.40, init_surrogate_slope=15.0, beta_decay=0.95),
                SpikingNeuronConfig("interneuron_attention", "attention gating", "GABA", 25.0,
                                  threshold=0.55, init_surrogate_slope=12.0, beta_decay=0.90),
            ],
            BrainZoneType.TEMPORAL_CORTEX: [
                SpikingNeuronConfig("pyramidal_memory", "memory encoding", "glutamate", 70.0,
                                  threshold=0.45, init_surrogate_slope=18.0, beta_decay=0.95),
                SpikingNeuronConfig("granule_semantic", "semantic associations", "glutamate", 25.0,
                                  threshold=0.45, init_surrogate_slope=20.0, beta_decay=0.92),
                SpikingNeuronConfig("interneuron_temporal", "temporal binding", "GABA", 5.0,
                                  threshold=0.45, init_surrogate_slope=15.0, beta_decay=0.90),
            ],
            BrainZoneType.HIPPOCAMPUS: [
                SpikingNeuronConfig("pyramidal_CA1", "memory consolidation", "glutamate", 40.0,
                                  threshold=0.45, init_surrogate_slope=18.0, beta_decay=0.96),
                SpikingNeuronConfig("pyramidal_CA3", "pattern completion", "glutamate", 35.0,
                                  threshold=0.45, init_surrogate_slope=20.0, beta_decay=0.94),
                SpikingNeuronConfig("granule_DG", "pattern separation", "glutamate", 25.0,
                                  threshold=0.45, init_surrogate_slope=22.0, beta_decay=0.90),
            ],
            BrainZoneType.CEREBELLUM: [
                SpikingNeuronConfig("granule_fine", "fine-tuning", "glutamate", 85.0,
                                  threshold=0.40, init_surrogate_slope=20.0, beta_decay=0.9),
                SpikingNeuronConfig("Purkinje_output", "output modulation", "GABA", 15.0,
                                  threshold=0.55, init_surrogate_slope=15.0, beta_decay=0.98),
            ],
            BrainZoneType.THALAMUS: [
                SpikingNeuronConfig("relay_neurons", "information relay", "glutamate", 85.0,
                                  threshold=0.45, init_surrogate_slope=18.0, beta_decay=0.91),
                SpikingNeuronConfig("interneuron_thalamic", "gating control", "GABA", 15.0,
                                  threshold=0.45, init_surrogate_slope=15.0, beta_decay=0.87),
            ],
            BrainZoneType.AMYGDALA: [
                SpikingNeuronConfig("pyramidal_emotional", "emotional processing", "glutamate", 70.0,
                                  threshold=0.45, init_surrogate_slope=18.0, beta_decay=0.93),
                SpikingNeuronConfig("interneuron_emotional", "emotional regulation", "GABA", 30.0,
                                  threshold=0.45, init_surrogate_slope=15.0, beta_decay=0.89),
            ],
        }
        
        return configs.get(self.zone_type, [
            SpikingNeuronConfig("default_pyramidal", "generic neurons", "glutamate", 80.0),
            SpikingNeuronConfig("default_interneuron", "generic inhibitory", "GABA", 20.0),
        ])

class EnhancedSpikingNeuron(nn.Module):
    """Enhanced spiking neuron with brain zone-specific properties"""
    
    def __init__(self, config: SpikingNeuronConfig, d_model: int, event_bus: Optional[EventBus] = None, zone_name: Optional[str] = None):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.event_bus = event_bus
        self.zone_name = zone_name
        self.use_izhikevich = all((config.a is not None, config.b is not None, config.c is not None, config.d is not None))
        # Always-on homeostatic learning targets
        self.target_firing_rate: float = float(2.0)
        self.homeostasis_lr = float(max(1e-5, min(0.01, config.learning_rate_modifier * 0.001)))  # ← GENTLER
        # Scalar bias current adapted by homeostasis (used for izh/adex); kept as buffer
        self.register_buffer('homeo_i', torch.tensor(0.0))
        # Smoothed firing rate (EMA)
        self.register_buffer('_fr_ema', torch.tensor(0.0), persistent=False)
        
        if not self.use_izhikevich and (self.config.model_type or '').lower() != 'adex':
            # Learnable parameters (LIF-like)
            self.beta = nn.Parameter(torch.tensor(config.beta_decay))
            self.threshold = nn.Parameter(torch.tensor(config.threshold))
            self.surrogate_slope = nn.Parameter(torch.tensor(config.init_surrogate_slope))
        
        # Neurotransmitter-specific weights
        if config.neurotransmitter == "glutamate":
            self.neurotransmitter_weight = nn.Parameter(torch.ones(d_model) * 1.2)
        elif config.neurotransmitter == "GABA":
            self.neurotransmitter_weight = nn.Parameter(torch.ones(d_model) * -0.8)
        else:
            self.neurotransmitter_weight = nn.Parameter(torch.ones(d_model))
        
        # State tracking
        if self.use_izhikevich:
            # One izh neuron per feature channel (share parameters, independent state)
            self.izh = IzhikevichNeuron(a=float(config.a), b=float(config.b), c=float(config.c), d=float(config.d), dt=float(config.dt))
            self.register_buffer('izh_v', torch.full((1, d_model), -65.0))
            self.register_buffer('izh_u', torch.full((1, d_model), float(config.b) * -65.0))
        elif (self.config.model_type or '').lower() == 'adex':
            p = self.config.model_params or {}
            self.adex = AdExNeuron(
                C=float(p.get('C', 200.0)), g_L=float(p.get('g_L', 10.0)), E_L=float(p.get('E_L', -70.0)),
                V_T=float(p.get('V_T', -50.0)), Delta_T=float(p.get('Delta_T', 2.0)), tau_w=float(p.get('tau_w', 120.0)),
                a=float(p.get('a', 0.0)), b=float(p.get('b', 0.0)), R=float(p.get('R', 1.0)),
                V_reset=float(p.get('V_reset', -65.0)), V_spike=float(p.get('V_spike', 30.0)), dt=float(p.get('dt', 0.1))
            )
        else:
            self.register_buffer('membrane_potential', torch.zeros(1, d_model))
            self.register_buffer('refractory_count', torch.zeros(1, d_model))
            self.register_buffer('fatigue_level', torch.zeros(1, d_model))
        
        # Activity history for monitoring
        self.register_buffer('activity_history', torch.zeros(100))
        self.activity_step = 0
        
        self.reset_state()
    
    def reset_state(self):
        """Reset neuron state"""
        if self.use_izhikevich:
            self.izh.reset_state()
            self.izh_v.fill_(-65.0)
            self.izh_u.fill_(float(self.config.b) * -65.0 if self.config.b is not None else -13.0)
        elif (self.config.model_type or '').lower() == 'adex':
            self.adex.reset_state()
        else:
            # Use zero baseline to operate with small positive thresholds
            self.membrane_potential.zero_()
            self.refractory_count.fill_(0)
            self.fatigue_level.fill_(0)
    
    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward pass with enhanced monitoring"""
        batch_size = input_current.size(0)
        
        # Expand state for batch processing
        if self.use_izhikevich:
            if self.izh_v.size(0) != batch_size:
                self.izh_v = torch.full((batch_size, self.d_model), -65.0, device=input_current.device, dtype=input_current.dtype)
                bval = float(self.config.b) if self.config.b is not None else 0.2
                self.izh_u = torch.full((batch_size, self.d_model), bval * -65.0, device=input_current.device, dtype=input_current.dtype)
        else:
            if self.membrane_potential.size(0) != batch_size:
                self.membrane_potential = self.membrane_potential.expand(batch_size, -1).contiguous()
                self.refractory_count = self.refractory_count.expand(batch_size, -1).contiguous()
                self.fatigue_level = self.fatigue_level.expand(batch_size, -1).contiguous()
        
        # Apply neurotransmitter modulation and fatigue, add small DC bias to encourage activity
        if self.use_izhikevich:
            # Vectorized Izhikevich update per channel
            v = self.izh_v
            u = self.izh_u
            I = input_current
            if self.config.i_bg is not None:
                I = I + float(self.config.i_bg)
            # Homeostatic bias current
            I = I + self.homeo_i
            # dv/dt and du/dt
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
            du = float(self.config.a) * (float(self.config.b) * v - u) if self.config.a is not None and self.config.b is not None else 0.02 * (0.2 * v - u)
            dt = float(self.config.dt) if self.config.dt is not None else 0.2
            v = v + dt * dv
            u = u + dt * du
            spikes = (v >= 30.0).to(input_current.dtype)
            # Reset
            cval = float(self.config.c) if self.config.c is not None else -65.0
            dval = float(self.config.d) if self.config.d is not None else 6.0
            v = torch.where(spikes > 0, torch.tensor(cval, device=v.device, dtype=v.dtype), v)
            u = torch.where(spikes > 0, u + dval, u)
            # Store back
            self.izh_v = v
            self.izh_u = u
            membrane = v
            spike_activity = spikes.float().mean()
            # Update history
            self.activity_history[self.activity_step % 100] = spike_activity
            self.activity_step += 1
            metrics = {
                'firing_rate': float(spike_activity.item()),
                'membrane_mean': float(membrane.mean().item()),
                'membrane_std': float(membrane.std(unbiased=False).item()) if membrane.numel() > 1 else 0.0,
                'surrogate_slope': 0.0,
                'fatigue_mean': 0.0,
                'neuron_type': self.config.neuron_type,
                'neurotransmitter': self.config.neurotransmitter
            }
            # Cap reported firing rate for monitoring stability
            metrics['firing_rate'] = float(max(0.0, min(metrics['firing_rate'], 0.999)))
            # Always-on homeostatic learning: adapt baseline current to track target firing
            with torch.no_grad():
                err = float(spike_activity.item()) - self.target_firing_rate
                self.homeo_i.add_(torch.tensor(-self.homeostasis_lr * err, device=self.homeo_i.device, dtype=self.homeo_i.dtype))
                self.homeo_i.clamp_(-5.0, 5.0)
            # Update EMA and include in metrics
            with torch.no_grad():
                self._fr_ema = self._fr_ema * 0.9 + 0.1 * spike_activity.detach()
            metrics['firing_rate_ema'] = float(self._fr_ema.item())
            if self.event_bus and spike_activity > 0:
                ev = metrics.copy()
                if self.zone_name:
                    ev['zone'] = self.zone_name
                self.event_bus.broadcast_neuron_fired(ev)
            return spikes, membrane, metrics
        elif (self.config.model_type or '').lower() == 'adex':
            # Collapse batch to mean current (coarse aggregation)
            I = input_current.mean(dim=0).mean()  # scalar drive
            # Add homeostatic current
            spk_scalar = self.adex.step(I + self.homeo_i)
            spikes = spk_scalar.expand(input_current.size(0), self.d_model)
            membrane = self.adex.V.expand(input_current.size(0), self.d_model)
            spike_activity = spikes.float().mean()
            self.activity_history[self.activity_step % 100] = spike_activity
            self.activity_step += 1
            metrics = {
                'firing_rate': float(spike_activity.item()),
                'membrane_mean': float(membrane.mean().item()),
                'membrane_std': float(membrane.std(unbiased=False).item()) if membrane.numel() > 1 else 0.0,
                'surrogate_slope': 0.0,
                'fatigue_mean': 0.0,
                'neuron_type': self.config.neuron_type,
                'neurotransmitter': self.config.neurotransmitter,
                'model': 'adex'
            }
            metrics['firing_rate'] = float(max(0.0, min(metrics['firing_rate'], 0.999)))
            # Homeostatic adaptation for AdEx bias current
            with torch.no_grad():
                err = float(spike_activity.item()) - self.target_firing_rate
                self.homeo_i.add_(torch.tensor(-self.homeostasis_lr * err, device=self.homeo_i.device, dtype=self.homeo_i.dtype))
                self.homeo_i.clamp_(-5.0, 5.0)
            with torch.no_grad():
                self._fr_ema = self._fr_ema * 0.9 + 0.1 * spike_activity.detach()
            metrics['firing_rate_ema'] = float(self._fr_ema.item())
            if self.event_bus and spike_activity > 0:
                ev = metrics.copy()
                if self.zone_name:
                    ev['zone'] = self.zone_name
                self.event_bus.broadcast_neuron_fired(ev)
            return spikes, membrane, metrics
        else:
            modulated_input = (input_current * self.neurotransmitter_weight * (1 - self.fatigue_level * 0.5)) + 0.2
        
        # Handle refractory period
        active_mask = (self.refractory_count <= 0).float()
        modulated_input = modulated_input * active_mask
        
        # LIF dynamics: stronger integration to ensure activity
        self.membrane_potential = (self.beta * self.membrane_potential + modulated_input)
        
        # Spike generation with learnable surrogate gradient (normalized thresholds)
        # Encourage spiking under moderate drive: clamp to small range
        thr_eff = torch.clamp(self.threshold, min=0.05, max=0.2)
        spikes = LearnableSurrogateGradient.apply(self.membrane_potential - thr_eff, self.surrogate_slope)

        
        # Reset and update states
        # Reset spiking units to zero baseline
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        
        # Update refractory period
        self.refractory_count = torch.maximum(
            self.refractory_count - 1,
            spikes * self.config.refractory_period
        )
        
        # Update fatigue
        spike_activity = spikes.float().mean()
        self.fatigue_level = torch.clamp(
            self.fatigue_level + spike_activity * 0.1 - 0.01, 0, 1)
        
        # Update activity history
        self.activity_history[self.activity_step % 100] = spike_activity
        self.activity_step += 1
        
        # Collect metrics
        numel_mem = int(self.membrane_potential.numel())
        metrics = {
            'firing_rate': float(spike_activity.item()),
            'membrane_mean': float(self.membrane_potential.mean().item()),
            'membrane_std': float(self.membrane_potential.std(unbiased=False).item()) if numel_mem > 1 else 0.0,
            'surrogate_slope': float(self.surrogate_slope.item()),
            'fatigue_mean': float(self.fatigue_level.mean().item()),
            'neuron_type': self.config.neuron_type,
            'neurotransmitter': self.config.neurotransmitter
        }
        metrics['firing_rate'] = float(max(0.0, min(metrics['firing_rate'], 0.999)))
        # Always-on homeostatic learning for LIF threshold
        with torch.no_grad():
            err = float(spike_activity.item()) - self.target_firing_rate
            self.threshold.add_(torch.tensor(self.homeostasis_lr * err * 1.5, device=self.threshold.device, dtype=self.threshold.dtype))
            # Keep threshold within a reasonable physiological range
            self.threshold.clamp_(-100.0, -10.0)
        
        # Update EMA and broadcast neuron fired event if event bus available
        with torch.no_grad():
            self._fr_ema = self._fr_ema * 0.9 + 0.1 * spike_activity.detach()
        metrics['firing_rate_ema'] = float(self._fr_ema.item())
        if self.event_bus and spike_activity > 0:
            ev = metrics.copy()
            if self.zone_name:
                ev['zone'] = self.zone_name
            self.event_bus.broadcast_neuron_fired(ev)
        
        return spikes, self.membrane_potential, metrics

class NeuromorphicBrainZone(nn.Module):
    """Enhanced brain zone with spiking neural networks"""
    
    def __init__(self, config: BrainZoneConfig):
        super().__init__()
        self.config = config
        
        if not config.use_spiking:
            # Fallback to basic processing if spiking disabled
            self.processing_mode = 'basic'
            self.basic_processor = nn.Sequential(
                nn.Linear(config.d_model, config.max_neurons),
                nn.ReLU(),
                nn.Linear(config.max_neurons, config.d_model)
            )
            return
        
        self.processing_mode = 'spiking'
        self.spiking_configs = config.spiking_configs or []
        # Ensure non-empty defaults to avoid silent zones
        if not self.spiking_configs:
            self.spiking_configs = [
                SpikingNeuronConfig("pyramidal_auto", "auto", "glutamate", 80.0,
                                    threshold=0.5, init_surrogate_slope=25.0, beta_decay=0.95, reset_potential=0.0),
                SpikingNeuronConfig("interneuron_auto", "auto", "GABA", 20.0,
                                    threshold=0.55, init_surrogate_slope=12.0, beta_decay=0.90, reset_potential=0.0),
            ]

        # Create neuron groups based on configurations
        self.neuron_groups = nn.ModuleDict()
        self.neuron_counts = {}

        # Ensure zones without explicit configs still allocate neurons proportionally
        total_neurons = max(1, int(config.max_neurons))

        # First pass: compute counts per config with floor, track order
        ordered_types: List[str] = []
        counts: List[int] = []
        for spiking_config in self.spiking_configs:
            neuron_type = spiking_config.neuron_type
            ordered_types.append(neuron_type)
            counts.append(max(1, int(total_neurons * spiking_config.percentage / 100.0)))

        # Adjust remainder to ensure exact total
        remainder = total_neurons - sum(counts)
        if remainder != 0 and counts:
            # Add remainder to the largest count (or last if tie)
            max_idx = max(range(len(counts)), key=lambda i: counts[i])
            counts[max_idx] = max(1, counts[max_idx] + remainder)

        # Build modules with final counts
        for neuron_type, count in zip(ordered_types, counts):
            self.neuron_counts[neuron_type] = count
            cfg = next(c for c in self.spiking_configs if c.neuron_type == neuron_type)
            self.neuron_groups[neuron_type] = EnhancedSpikingNeuron(cfg, count, config.event_bus, zone_name=config.name)
        # Per-group homeostatic bias (learned) and firing-rate EMA
        self.group_bias_params = nn.ParameterDict({nt: nn.Parameter(torch.zeros(1)) for nt in self.neuron_groups.keys()})
        # Split homeostatic targets for excitatory/inhibitory groups
        self.group_target_rate_exc: float = 0.18
        self.group_target_rate_inh: float = 0.22
        self.group_homeo_lr: float = 0.05
        self._group_fr_ema: Dict[str, float] = {nt: 0.0 for nt in self.neuron_groups.keys()}
        
        # Zone processing layers
        self.input_projection = AdditionLinear(config.d_model, total_neurons, bias=False)
        # Output projection acts on last dimension only. For [B,T,N], apply per-time-step
        self.output_projection = AdditionLinear(total_neurons, config.d_model, bias=False)
        
        # Zone-specific processing
        self.zone_processing = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(0.1)
        )
        
        # Activity monitoring
        self.register_buffer('zone_activity_history', torch.zeros(100, len(self.spiking_configs)))
        self.zone_activity_step = 0
    
    def forward(self, x: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process input through neuromorphic brain zone"""
        
        if self.processing_mode == 'basic':
            output = self.basic_processor(x)
            return output, {'mode': 'basic', 'zone_name': self.config.name}
        
        # Extract context parameters first
        dc_bias = 0.0
        boost = 1.0
        noise_std = 0.0
        force_spike = False
        rectify = True
        debug_spike = False
        try:
            if context and isinstance(context, dict):
                dc_bias = float(context.get('dc_bias', 0.0))
                boost = float(context.get('boost', 1.0))
                noise_std = float(context.get('noise_std', 0.0))
                force_spike = bool(context.get('force_spike', False))
                rectify = bool(context.get('rectify', True))
                debug_spike = bool(context.get('debug_spike', False))
        except Exception:
            pass
        
        # Enhanced neuromorphic processing
        if debug_spike:
            print(f"[DEBUG] {self.config.name} raw x shape={x.shape} mean={float(x.mean()):.3f} max={float(x.max()):.3f}")
        processed_input = self.zone_processing(x)
        if debug_spike:
            print(f"[DEBUG] {self.config.name} after zone_processing shape={processed_input.shape} mean={float(processed_input.mean()):.3f} max={float(processed_input.max()):.3f}")

        # Support inputs of shape [B, D] or [B, T, D]
        is_sequence = processed_input.dim() == 3

        if not is_sequence:
            zone_input = self.input_projection(processed_input)
        else:
            B, T, D = processed_input.shape
            zone_input = self.input_projection(processed_input.reshape(B * T, D)).reshape(B, T, -1)
        # Flip sign to convert distance-like outputs into similarity-like activations
        zone_input = -zone_input
        # Soften projection magnitude and clamp to a narrower stable range
        zone_input = zone_input * 0.5
        zone_input = torch.clamp(zone_input, -0.6, 0.6)
        
        if debug_spike:
            print(f"[DEBUG] {self.config.name} after input_projection zone_input shape={zone_input.shape} mean={float(zone_input.mean()):.3f} max={float(zone_input.max()):.3f}")

        zone_metrics: Dict[str, Any] = {}

        if not is_sequence:
            # Single-step processing [B, N]
            zone_outputs = []
            start_idx = 0
            for neuron_type, neuron_module in self.neuron_groups.items():
                count = self.neuron_counts[neuron_type]
                neuron_input = zone_input[..., start_idx:start_idx + count]
                # Apply learned per-group bias before gating
                neuron_input = neuron_input + self.group_bias_params[neuron_type]
                gate = AdditiveReceptance(d_model=count, d_ff=count)
                if force_spike:
                    gated_input = neuron_input
                else:
                    if neuron_input.dim() == 2:
                        gated_input = neuron_input * gate(neuron_input)
                    else:
                        gated_input = neuron_input * gate(neuron_input.reshape(-1, count)).reshape_as(neuron_input)
                if boost != 1.0:
                    gated_input = gated_input * boost
                if noise_std > 0.0:
                    gated_input = gated_input + torch.randn_like(gated_input) * noise_std
                if dc_bias != 0.0:
                    gated_input = gated_input + dc_bias
                if rectify:
                    gated_input = torch.relu(gated_input)
                if debug_spike:
                    try:
                        thr_val = float(getattr(neuron_module, 'threshold', torch.tensor(0.0)).detach().item()) if hasattr(neuron_module, 'threshold') else None
                        slp_val = float(getattr(neuron_module, 'surrogate_slope', torch.tensor(0.0)).detach().item()) if hasattr(neuron_module, 'surrogate_slope') else None
                        gi = gated_input
                        print(f"[DEBUG] {self.config.name}.{neuron_type} inp(mean={float(gi.mean()):.3f}, max={float(gi.max()):.3f}) thr={thr_val} slope={slp_val}")
                    except Exception:
                        pass
                spikes, membrane, metrics = neuron_module(gated_input)
                if debug_spike:
                    try:
                        print(f"[DEBUG] {self.config.name}.{neuron_type} spike_mean={float(spikes.float().mean().item()):.3f}")
                    except Exception:
                        pass
                zone_outputs.append(spikes)
                zone_metrics[neuron_type] = metrics
                # Homeostatic update: adapt bias toward target firing rate
                with torch.no_grad():
                    fr = float(spikes.float().mean().item())
                    self._group_fr_ema[neuron_type] = self._group_fr_ema[neuron_type] * 0.9 + 0.1 * fr
                    is_exc = neuron_type.lower().startswith(('pyramidal','granule','relay'))
                    target = self.group_target_rate_exc if is_exc else self.group_target_rate_inh
                    err = target - self._group_fr_ema[neuron_type]
                    self.group_bias_params[neuron_type].add_(torch.tensor(self.group_homeo_lr * err, device=self.group_bias_params[neuron_type].device))
                    self.group_bias_params[neuron_type].clamp_(-0.2, 0.2)
                start_idx += count

            combined_output = torch.cat(zone_outputs, dim=-1) if zone_outputs else zone_input
            output = self.output_projection(combined_output)
            # Calculate average spike rate for the 2D case
            avg_spike_rate = float(combined_output.float().mean().item()) if combined_output.numel() > 0 else 0.0
            if debug_spike:
                print(f"[DEBUG] {self.config.name} 2D combined_output shape={combined_output.shape} mean={avg_spike_rate:.3f}")
        else:
            # Sequence processing [B, T, N]
            B, T, N = zone_input.shape
            step_outputs: List[torch.Tensor] = []
            # Accumulate metrics per neuron type (use last step for reporting)
            metrics_last: Dict[str, Any] = {}
            for t in range(T):
                step_zone_outputs = []
                start_idx = 0
                step_input = zone_input[:, t, :]
                for neuron_type, neuron_module in self.neuron_groups.items():
                    count = self.neuron_counts[neuron_type]
                    neuron_input = step_input[:, start_idx:start_idx + count]
                    # Apply learned per-group bias before gating
                    neuron_input = neuron_input + self.group_bias_params[neuron_type]
                    gate = AdditiveReceptance(d_model=count, d_ff=count)
                    gated_input = neuron_input if force_spike else (neuron_input * gate(neuron_input))
                    if boost != 1.0:
                        gated_input = gated_input * boost
                    if noise_std > 0.0:
                        gated_input = gated_input + torch.randn_like(gated_input) * noise_std
                    if dc_bias != 0.0:
                        gated_input = gated_input + dc_bias
                    if rectify:
                        gated_input = torch.relu(gated_input)
                    if debug_spike:
                        try:
                            thr_val = float(getattr(neuron_module, 'threshold', torch.tensor(0.0)).detach().item()) if hasattr(neuron_module, 'threshold') else None
                            slp_val = float(getattr(neuron_module, 'surrogate_slope', torch.tensor(0.0)).detach().item()) if hasattr(neuron_module, 'surrogate_slope') else None
                            gi = gated_input
                            print(f"[DEBUG] {self.config.name}.{neuron_type}@t{t} inp(mean={float(gi.mean()):.3f}, max={float(gi.max()):.3f}) thr={thr_val} slope={slp_val}")
                        except Exception:
                            pass
                    spikes, membrane, metrics = neuron_module(gated_input)
                    if debug_spike:
                        try:
                            print(f"[DEBUG] {self.config.name}.{neuron_type}@t{t} spike_mean={float(spikes.float().mean().item()):.3f}")
                        except Exception:
                            pass
                    step_zone_outputs.append(spikes)
                    metrics_last[neuron_type] = metrics
                    # Homeostatic update per timestep
                    with torch.no_grad():
                        fr = float(spikes.float().mean().item())
                        self._group_fr_ema[neuron_type] = self._group_fr_ema[neuron_type] * 0.9 + 0.1 * fr
                        is_exc = neuron_type.lower().startswith(('pyramidal','granule','relay'))
                        target = self.group_target_rate_exc if is_exc else self.group_target_rate_inh
                        err = target - self._group_fr_ema[neuron_type]
                        self.group_bias_params[neuron_type].add_(torch.tensor(self.group_homeo_lr * err, device=self.group_bias_params[neuron_type].device))
                        self.group_bias_params[neuron_type].clamp_(-0.2, 0.2)
                    start_idx += count
                combined_step = torch.cat(step_zone_outputs, dim=-1) if step_zone_outputs else step_input
                step_outputs.append(combined_step)
            # Stack over time and project back to d_model
            combined_output = torch.stack(step_outputs, dim=1)  # [B, T, total_neurons]
            # Apply projection across last dimension using view
            output = self.output_projection(combined_output.reshape(B*T, -1)).reshape(B, T, -1)
            zone_metrics = metrics_last
            # Compute average spiking rate over the whole sequence for clearer activity reporting
            avg_spike_rate = float(combined_output.float().mean().item()) if combined_output.numel() > 0 else 0.0
            if debug_spike:
                print(f"[DEBUG] {self.config.name} 3D combined_output shape={combined_output.shape} mean={avg_spike_rate:.3f}")
        
        # Update zone activity history
        self._update_zone_activity_history(zone_metrics)
        
        # Compile comprehensive zone activity information
        # Use the calculated avg_spike_rate, not the last neuron's individual rate
        zone_activity = {
            'zone_name': self.config.name,
            'zone_type': self.config.zone_type.value if self.config.zone_type else 'unknown',
            'neuron_metrics': zone_metrics,
            'total_neurons': sum(self.neuron_counts.values()),
            'avg_firing_rate': avg_spike_rate,
            'context': context or {},
            'processing_mode': self.processing_mode
        }
        
        return output, zone_activity
    
    def _update_zone_activity_history(self, zone_metrics: Dict[str, Dict[str, Any]]):
        """Update activity history for monitoring"""
        activity_values = []
        for neuron_type in self.neuron_groups.keys():
            if neuron_type in zone_metrics:
                activity_values.append(zone_metrics[neuron_type]['firing_rate'])
            else:
                activity_values.append(0.0)
        
        if activity_values and len(activity_values) <= self.zone_activity_history.size(1):
            activity_tensor = torch.tensor(activity_values + [0.0] * (self.zone_activity_history.size(1) - len(activity_values)))
            self.zone_activity_history[self.zone_activity_step % 100] = activity_tensor[:self.zone_activity_history.size(1)]
            self.zone_activity_step += 1
    
    def get_activity_stats(self) -> Dict[str, Any]:
        """Get comprehensive activity statistics for this zone"""
        if self.zone_activity_step == 0:
            return {'zone_name': self.config.name, 'no_activity': True}
        
        recent_steps = min(self.zone_activity_step, 100)
        recent_activity = self.zone_activity_history[:recent_steps]
        
        stats = {
            'zone_name': self.config.name,
            'zone_type': self.config.zone_type.value if self.config.zone_type else 'unknown',
            'total_neurons': sum(self.neuron_counts.values()),
            'neuron_type_distribution': dict(self.neuron_counts),
            'neuron_type_stats': {}
        }
        
        for i, (neuron_type, count) in enumerate(self.neuron_counts.items()):
            if i < recent_activity.size(1):
                activity_data = recent_activity[:, i]
                stats['neuron_type_stats'][neuron_type] = {
                    'count': count,
                    'mean_firing_rate': float(activity_data.mean()),
                    'std_firing_rate': float(activity_data.std(unbiased=False)) if activity_data.numel() > 1 else 0.0,
                    'max_firing_rate': float(activity_data.max()),
                    'min_firing_rate': float(activity_data.min()),
                    'recent_activity': float(activity_data[-1]) if len(activity_data) > 0 else 0.0
                }
        
        return stats
    
    def check_zone_health(self) -> Dict[str, str]:
        """Check health status of neurons in this zone"""
        health_report = {}
        
        for neuron_type, neuron_module in self.neuron_groups.items():
            if hasattr(neuron_module, 'activity_history') and neuron_module.activity_step > 10:
                recent_activity = neuron_module.activity_history[max(0, neuron_module.activity_step-10):neuron_module.activity_step]
                avg_activity = float(recent_activity.mean())
                
                if avg_activity < 0.001:
                    health_report[neuron_type] = "silent"
                elif avg_activity > 0.8:
                    health_report[neuron_type] = "hyperactive"
                else:
                    health_report[neuron_type] = "healthy"
            else:
                health_report[neuron_type] = "initializing"
        
        return health_report

    # -----------------------------
    # Homeostatic state save/load
    # -----------------------------
    def export_homeostasis_state(self) -> Dict[str, Any]:
        """Export per-group homeostatic state (biases, EMA, targets)."""
        try:
            return {
                'zone_name': self.config.name,
                'biases': {k: float(v.detach().cpu().item()) for k, v in self.group_bias_params.items()},
                'ema': dict(self._group_fr_ema),
                'targets': {
                    'exc': float(self.group_target_rate_exc),
                    'inh': float(self.group_target_rate_inh),
                },
                'lr': float(self.group_homeo_lr),
            }
        except Exception:
            return {
                'zone_name': self.config.name,
                'biases': {},
                'ema': {},
                'targets': {'exc': 0.0, 'inh': 0.0},
                'lr': float(self.group_homeo_lr),
            }

    def import_homeostasis_state(self, state: Dict[str, Any]) -> None:
        """Import per-group homeostatic state (best-effort)."""
        try:
            biases = state.get('biases', {})
            for k, v in biases.items():
                if k in self.group_bias_params:
                    try:
                        self.group_bias_params[k].data.copy_(torch.tensor(float(v), dtype=self.group_bias_params[k].dtype, device=self.group_bias_params[k].device))
                    except Exception:
                        pass
            ema = state.get('ema', {})
            for k, v in ema.items():
                if k in self._group_fr_ema:
                    try:
                        self._group_fr_ema[k] = float(v)
                    except Exception:
                        pass
            t = state.get('targets', {})
            if 'exc' in t:
                try:
                    self.group_target_rate_exc = float(t['exc'])
                except Exception:
                    pass
            if 'inh' in t:
                try:
                    self.group_target_rate_inh = float(t['inh'])
                except Exception:
                    pass
            if 'lr' in state:
                try:
                    self.group_homeo_lr = float(state['lr'])
                except Exception:
                    pass
        except Exception:
            pass

    def save_homeostasis_state(self, filepath: str) -> bool:
        """Save homeostasis state to JSON file. Returns True on success."""
        try:
            import json, os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.export_homeostasis_state(), f, indent=2)
            return True
        except Exception:
            return False

    def load_homeostasis_state(self, filepath: str) -> bool:
        """Load homeostasis state from JSON file. Returns True on success."""
        try:
            import json, os
            if not os.path.isfile(filepath):
                return False
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.import_homeostasis_state(data)
            return True
        except Exception:
            return False

class BrainZone:
    """Enhanced BrainZone class maintaining compatibility with your existing architecture"""
    config: BrainZoneConfig
    layers: Dict[int, BaseLayer]
    neuromorphic_processor: Optional[NeuromorphicBrainZone]
    
    def __init__(self, config: BrainZoneConfig, layers: Dict[int, BaseLayer]):
        self.config = config
        self.layers = layers
        
        # Add neuromorphic processing capability
        if config.use_spiking:
            self.neuromorphic_processor = NeuromorphicBrainZone(config)
        else:
            self.neuromorphic_processor = None
    
    def get_config(self) -> BrainZoneConfig:
        return self.config
    
    def process(self, x: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process input through this brain zone"""
        if self.neuromorphic_processor:
            return self.neuromorphic_processor(x, context)
        else:
            # Fallback to basic processing through layers
            avg_spike_rate = float(combined_output.float().mean().item()) if combined_output.numel() > 0 else 0.0
            if debug_spike:
                print(f"[DEBUG] {self.config.name} 3D combined_output shape={combined_output.shape} mean={avg_spike_rate:.3f}")
        
        # Update zone activity history
        self._update_zone_activity_history(zone_metrics)
        
        # Compile comprehensive zone activity information
        # Use the calculated avg_spike_rate, not the last neuron's individual rate
        zone_activity = {
            'zone_name': self.config.name,
            'zone_type': self.config.zone_type.value if self.config.zone_type else 'unknown',
            'neuron_metrics': zone_metrics,
            'total_neurons': sum(self.neuron_counts.values()),
            'avg_firing_rate': avg_spike_rate,
            'context': context or {},
            'processing_mode': self.processing_mode
        }
        
        return output, zone_activity
    
    def _update_zone_activity_history(self, zone_metrics: Dict[str, Dict[str, Any]]):
        """Update activity history for monitoring"""
        activity_values = []
        for neuron_type in self.neuron_groups.keys():
            if neuron_type in zone_metrics:
                activity_values.append(zone_metrics[neuron_type]['firing_rate'])
            else:
                activity_values.append(0.0)
        
        if activity_values and len(activity_values) <= self.zone_activity_history.size(1):
            activity_tensor = torch.tensor(activity_values + [0.0] * (self.zone_activity_history.size(1) - len(activity_values)))
            self.zone_activity_history[self.zone_activity_step % 100] = activity_tensor[:self.zone_activity_history.size(1)]
            self.zone_activity_step += 1
    
    def get_activity_stats(self) -> Dict[str, Any]:
        """Get comprehensive activity statistics for this zone"""
        if self.zone_activity_step == 0:
            return {'zone_name': self.config.name, 'no_activity': True}
        
        recent_steps = min(self.zone_activity_step, 100)
        recent_activity = self.zone_activity_history[:recent_steps]
        
        stats = {
            'zone_name': self.config.name,
            'zone_type': self.config.zone_type.value if self.config.zone_type else 'unknown',
            'total_neurons': sum(self.neuron_counts.values()),
            'neuron_type_distribution': dict(self.neuron_counts),
            'neuron_type_stats': {}
        }
        
        for i, (neuron_type, count) in enumerate(self.neuron_counts.items()):
            if i < recent_activity.size(1):
                activity_data = recent_activity[:, i]
                stats['neuron_type_stats'][neuron_type] = {
                    'count': count,
                    'mean_firing_rate': float(activity_data.mean()),
                    'std_firing_rate': float(activity_data.std(unbiased=False)) if activity_data.numel() > 1 else 0.0,
                    'max_firing_rate': float(activity_data.max()),
                    'min_firing_rate': float(activity_data.min()),
                    'recent_activity': float(activity_data[-1]) if len(activity_data) > 0 else 0.0
                }
        
        return stats
    
    def check_zone_health(self) -> Dict[str, str]:
        """Check health status of neurons in this zone"""
        health_report = {}
        
        for neuron_type, neuron_module in self.neuron_groups.items():
            if hasattr(neuron_module, 'activity_history') and neuron_module.activity_step > 10:
                recent_activity = neuron_module.activity_history[max(0, neuron_module.activity_step-10):neuron_module.activity_step]
                avg_activity = float(recent_activity.mean())
                
                if avg_activity < 0.001:
                    health_report[neuron_type] = "silent"
                elif avg_activity > 0.8:
                    health_report[neuron_type] = "hyperactive"
                else:
                    health_report[neuron_type] = "healthy"
            else:
                health_report[neuron_type] = "initializing"
        
        return health_report

    # -----------------------------
    # Homeostatic state save/load
    # -----------------------------
    def export_homeostasis_state(self) -> Dict[str, Any]:
        """Export per-group homeostatic state (biases, EMA, targets)."""
        try:
            return {
                'zone_name': self.config.name,
                'biases': {k: float(v.detach().cpu().item()) for k, v in self.group_bias_params.items()},
                'ema': dict(self._group_fr_ema),
                'targets': {
                    'exc': float(self.group_target_rate_exc),
                    'inh': float(self.group_target_rate_inh),
                },
                'lr': float(self.group_homeo_lr),
            }
        except Exception:
            return {
                'zone_name': self.config.name,
                'biases': {},
                'ema': {},
                'targets': {'exc': 0.0, 'inh': 0.0},
                'lr': float(self.group_homeo_lr),
            }

    def import_homeostasis_state(self, state: Dict[str, Any]) -> None:
        """Import per-group homeostatic state (best-effort)."""
        try:
            biases = state.get('biases', {})
            for k, v in biases.items():
                if k in self.group_bias_params:
                    try:
                        self.group_bias_params[k].data.copy_(torch.tensor(float(v), dtype=self.group_bias_params[k].dtype, device=self.group_bias_params[k].device))
                    except Exception:
                        pass
            ema = state.get('ema', {})
            for k, v in ema.items():
                if k in self._group_fr_ema:
                    try:
                        self._group_fr_ema[k] = float(v)
                    except Exception:
                        pass
            t = state.get('targets', {})
            if 'exc' in t:
                try:
                    self.group_target_rate_exc = float(t['exc'])
                except Exception:
                    pass
            if 'inh' in t:
                try:
                    self.group_target_rate_inh = float(t['inh'])
                except Exception:
                    pass
            if 'lr' in state:
                try:
                    self.group_homeo_lr = float(state['lr'])
                except Exception:
                    pass
        except Exception:
            pass

    def save_homeostasis_state(self, filepath: str) -> bool:
        """Save homeostasis state to JSON file. Returns True on success."""
        try:
            import json, os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.export_homeostasis_state(), f, indent=2)
            return True
        except Exception:
            return False

    def load_homeostasis_state(self, filepath: str) -> bool:
        """Load homeostasis state from JSON file. Returns True on success."""
        try:
            import json, os
            if not os.path.isfile(filepath):
                return False
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.import_homeostasis_state(data)
            return True
        except Exception:
            if pattern_json_path:
                izh_map = load_izhikevich_keypatterns_map(pattern_json_path)
            precise_map = load_precise_adex_map_default()
            spiking_configs = build_spiking_configs_for_zone(zone_type, pattern_csv_path, izh_map, precise_map)

        config = BrainZoneConfig(
            name=zone_name,
            max_neurons=max_neurons,
            min_neurons=max_neurons // 2,
            zone_type=zone_type,
            d_model=d_model,
            use_spiking=True,
            event_bus=event_bus,
            spiking_configs=spiking_configs
        )
        
        # Create basic layers as fallback
        layers = {
            0: BaseLayerImplementation(BaseLayerConfig(
                name=f"{zone_name}_layer_0",
                input_dim=d_model,
                output_dim=max_neurons
            )),
            1: BaseLayerImplementation(BaseLayerConfig(
                name=f"{zone_name}_layer_1", 
                input_dim=max_neurons,
                output_dim=d_model
            ))
        }
        
        return self.create_brain_zone(config, layers)

"""
# Import the basic layer implementation for compatibility
from base.layers import BaseLayerImplementation

# Helper functions for easy zone creation
def create_prefrontal_cortex(d_model: int = 512, max_neurons: int = 512, 
                           event_bus: Optional[EventBus] = None,
                           pattern_csv_path: Optional[str] = None,
                           pattern_json_path: Optional[str] = None) -> BrainZone:

    factory = BrainZoneFactory()
    return factory.create_neuromorphic_zone(
        "prefrontal_cortex", BrainZoneType.PREFRONTAL_CORTEX, 
        d_model, max_neurons, event_bus, pattern_csv_path, pattern_json_path)

def create_temporal_cortex(d_model: int = 512, max_neurons: int = 512,
                         event_bus: Optional[EventBus] = None,
                         pattern_csv_path: Optional[str] = None,
                         pattern_json_path: Optional[str] = None) -> BrainZone:
    factory = BrainZoneFactory()
    return factory.create_neuromorphic_zone(
        "temporal_cortex", BrainZoneType.TEMPORAL_CORTEX,
        d_model, max_neurons, event_bus, pattern_csv_path, pattern_json_path)

def create_hippocampus(d_model: int = 512, max_neurons: int = 384,
                      event_bus: Optional[EventBus] = None,
                      pattern_csv_path: Optional[str] = None,
                      pattern_json_path: Optional[str] = None) -> BrainZone:
    factory = BrainZoneFactory()
    return factory.create_neuromorphic_zone(
        "hippocampus", BrainZoneType.HIPPOCAMPUS,
        d_model, max_neurons, event_bus, pattern_csv_path, pattern_json_path)

def create_cerebellum(d_model: int = 512, max_neurons: int = 256,
                     event_bus: Optional[EventBus] = None,
                     pattern_csv_path: Optional[str] = None,
                     pattern_json_path: Optional[str] = None) -> BrainZone:
    factory = BrainZoneFactory()
    return factory.create_neuromorphic_zone(
        "cerebellum", BrainZoneType.CEREBELLUM,
        d_model, max_neurons, event_bus, pattern_csv_path, pattern_json_path)


# -----------------------------
# Pattern CSV integration
# -----------------------------

def _region_hints_for_zone(zone_type: BrainZoneType) -> List[str]:
    if zone_type in (BrainZoneType.PREFRONTAL_CORTEX, BrainZoneType.TEMPORAL_CORTEX,
                     BrainZoneType.PARIETAL_CORTEX, BrainZoneType.OCCIPITAL_CORTEX):
        return ["neocortex", "cortex"]
    if zone_type == BrainZoneType.HIPPOCAMPUS:
        return ["hippocampus", "CA1", "CA3", "Dentate"]
    if zone_type == BrainZoneType.CEREBELLUM:
        return ["cerebellum", "Purkinje", "granule"]
    if zone_type == BrainZoneType.THALAMUS:
        return ["thalamus"]
    if zone_type == BrainZoneType.AMYGDALA:
        return ["amygdala"]
    if zone_type == BrainZoneType.BRAINSTEM:
        return ["brainstem", "raphe", "VTA", "SNc", "locus coeruleus"]
    return []


def _guess_neurotransmitter(neuron_type: str) -> str:
    nt = neuron_type.lower()
    if "pyramidal" in nt:
        return "glutamate"
    if "interneuron" in nt or "fast spiking" in nt or "purkinje" in nt:
        return "GABA"
    return "glutamate"


def load_izhikevich_keypatterns_map(json_path: str) -> Dict[str, Dict[str, float]]:
    try:
        import json
        with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        models = data.get('comprehensive_neuron_models', {}).get('models', {})
        izh = models.get('1_izhikevich', {})
        kp = izh.get('key_patterns', {})
        out: Dict[str, Dict[str, float]] = {}
        for k, v in kp.items():
            if all(x in v for x in ('a','b','c','d')):
                out[str(k).strip().lower()] = {k2: float(v[k2]) for k2 in ('a','b','c','d')}
        return out
    except Exception:
        return {}

def load_precise_adex_map_default() -> Dict[str, Dict[str, float]]:
    path = os.path.join(os.getcwd(), 'precise_patterns_params.json')
    if not os.path.isfile(path):
        return {}
    try:
        import json
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        models = data.get('precise_neuron_model_parameters', {}).get('models', {})
        adex = models.get('adaptive_exponential', {}).get('firing_patterns', {})
        out: Dict[str, Dict[str, float]] = {}
        for name, spec in adex.items():
            params = spec.get('parameters') or {}
            out[str(name).strip().lower()] = {k: float(v) for k, v in params.items() if isinstance(v, (int,float))}
        return out
    except Exception:
        return {}


def build_spiking_configs_for_zone(zone_type: BrainZoneType, csv_path: str, izh_map: Optional[Dict[str, Dict[str, float]]] = None, adex_map: Optional[Dict[str, Dict[str, float]]] = None) -> List[SpikingNeuronConfig]:
    rows: List[Dict[str, str]] = []
    hints = [h.lower() for h in _region_hints_for_zone(zone_type)]
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                region = (row.get("Brain_Region") or "").lower()
                if any(h in region for h in hints):
                    rows.append(row)
    except Exception:
        rows = []
    if not rows:
        return []
    pct = max(1.0, 100.0 / len(rows))
    configs: List[SpikingNeuronConfig] = []
    # Try to read explicit percentages if provided; otherwise distribute evenly
    explicit_pcts: List[Optional[float]] = []
    for r in rows:
        pct_str = r.get('Percent') or r.get('Percentage') or r.get('%')
        try:
            explicit_pcts.append(float(pct_str) if pct_str is not None and pct_str != '' else None)
        except Exception:
            explicit_pcts.append(None)
    if any(p is not None for p in explicit_pcts):
        # Normalize provided percentages to sum to 100
        total_provided = sum(p for p in explicit_pcts if p is not None)
        if total_provided and total_provided > 0:
            norm = 100.0 / total_provided
            norm_pcts = [(p * norm) if p is not None else 0.0 for p in explicit_pcts]
        else:
            norm_pcts = [None] * len(rows)
    else:
        norm_pcts = [None] * len(rows)

    for idx, r in enumerate(rows):
        neuron_type = r.get("Neuron_Type") or "unknown"
        primary = r.get("Primary_Pattern") or "unspecified"
        nt = _guess_neurotransmitter(neuron_type)
        # Parse optional Izhi params if present in CSV row
        a=b=c=d=Ibg=None
        try:
            param_str = r.get("Izhikevich_Parameters") or ""
            if param_str:
                parts = {p.split("=")[0].strip(): float(p.split("=")[1]) for p in param_str.split(",") if "=" in p}
                a = parts.get('a'); b = parts.get('b'); c = parts.get('c'); d = parts.get('d')
        except Exception:
            pass
        # If missing, try to map Primary_Pattern to key_patterns
        if (a is None or b is None or c is None or d is None) and izh_map:
            key = primary.lower().strip()
            # Normalize common names
            replacements = {
                'regular spiking (rs)': 'regular_spiking',
                'intrinsically bursting (ib)': 'bursting',
                'chattering (ch)': 'chattering',
                'fast spiking (fs)': 'fast_spiking',
                'low threshold spiking (lts)': None,
            }
            mapped = replacements.get(key)
            if mapped is None and 'regular spiking' in key:
                mapped = 'regular_spiking'
            if mapped is None and 'fast spiking' in key:
                mapped = 'fast_spiking'
            if mapped is None and 'burst' in key:
                mapped = 'bursting'
            if mapped is None and 'chatter' in key:
                mapped = 'chattering'
            if mapped and mapped in izh_map:
                vals = izh_map[mapped]
                a = vals.get('a'); b = vals.get('b'); c = vals.get('c'); d = vals.get('d')

        # If still no izh params, try AdEx precise mapping for known cortical patterns
        model_type = None
        model_params = None
        if (a is None or b is None or c is None or d is None) and adex_map:
            k = primary.lower().strip()
            # map typical names to adex keys
            alias = {
                'regular spiking (rs)': 'regular_spiking',
                'fast spiking (fs)': 'fast_spiking',
                'intrinsically bursting (ib)': 'intrinsic_bursting',
                'chattering (ch)': 'chattering',
            }
            ak = alias.get(k)
            if ak is None:
                if 'regular spiking' in k: ak = 'regular_spiking'
                elif 'fast spiking' in k: ak = 'fast_spiking'
                elif 'burst' in k: ak = 'intrinsic_bursting'
                elif 'chatter' in k: ak = 'chattering'
            if ak and ak in adex_map:
                model_type = 'adex'
                model_params = adex_map[ak]
        cfg = SpikingNeuronConfig(
            neuron_type=neuron_type,
            structure=primary,
            neurotransmitter=nt,
            percentage=(norm_pcts[idx] if norm_pcts[idx] is not None else pct),
            a=a, b=b, c=c, d=d, i_bg=Ibg,
            model_type=model_type, model_params=model_params,
        )
        configs.append(cfg)
    return configs
    """