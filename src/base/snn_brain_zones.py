import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.base.neuron import IzhikevichNeuron, AdExNeuron, VectorizedLIFNeuron
from src.base.snn_layers import BaseLayer
from src.maths.addition_linear import AdditionLinear
from src.base.events import EventBus

from enum import Enum
class BrainZoneType(Enum):
    PREFRONTAL_CORTEX = "prefrontal_cortex"; TEMPORAL_CORTEX = "temporal_cortex"
    HIPPOCAMPUS = "hippocampus"; CEREBELLUM = "cerebellum"; THALAMUS = "thalamus"
    AMYGDALA = "amygdala"; BASAL_GANGLIA = "basal_ganglia"; BRAINSTEM = "brainstem"
    OCCIPITAL_CORTEX = "occipital_cortex"; PARIETAL_CORTEX = "parietal_cortex"; INSULAR_CORTEX = "insular_cortex"

@dataclass
class SpikingNeuronConfig:
    neuron_type: str; structure: str; neurotransmitter: str; percentage: float
    threshold: float = 0.6; membrane_time_constant: float = 10.0
    init_surrogate_slope: float = 15.0; beta_decay: float = 0.95
    a: float = None; b: float = None; c: float = None; d: float = None; dt: float = 0.2
    model_type: str = None; model_params: Dict = None

@dataclass
class BrainZoneConfig:
    name: str = ""; max_neurons: int = 1024; min_neurons: int = 256
    neuron_type: str = "liquid"; gated: bool = False; num_layers: int = 2
    base_layer_container_config: Any = None; zone_type: BrainZoneType = None
    d_model: int = 1024; use_spiking: bool = True
    spiking_configs: List[SpikingNeuronConfig] = None; event_bus: EventBus = None

class EnhancedSpikingNeuron(nn.Module):
    def __init__(self, config, d_model, event_bus=None, zone_name=None):
        super().__init__()
        self.config = config
        self.d_model = d_model
        
        if config.a is not None: 
            self.core = IzhikevichNeuron(
                a=config.a, b=config.b, c=config.c, d=config.d, dt=config.dt
            )
            self.mode = 'izh'
        elif config.model_type == 'adex':
            p = config.model_params or {}
            self.core = AdExNeuron(**p)
            self.mode = 'adex'
        else: 
            self.core = VectorizedLIFNeuron(
                size=d_model, beta=config.beta_decay, threshold=config.threshold, 
                init_slope=config.init_surrogate_slope, event_bus=event_bus, name=zone_name
            )
            self.mode = 'lif'
            
        self.register_buffer('homeo_i', torch.tensor(0.0))

    def forward(self, x):
        is_seq = x.dim() == 3
        x_eff = x + self.homeo_i
        
        if self.mode in ('izh', 'adex'):
            if is_seq:
                spikes = self.core.forward_sequence(x_eff)
                return spikes, {}, {}
            else:
                spikes = self.core.forward_sequence(x_eff.unsqueeze(1)).squeeze(1)
                return spikes, {}, {}
        else:
            if is_seq:
                spikes_list = []
                # LIF sequence processing (could be optimized further)
                for t in range(x.shape[1]):
                    s, _ = self.core(x_eff[:, t])
                    spikes_list.append(s)
                return torch.stack(spikes_list, dim=1), None, {}
            else:
                spikes, mem = self.core(x_eff)
                return spikes, mem, {}

class NeuromorphicBrainZone(nn.Module):
    def __init__(self, config: BrainZoneConfig):
        super().__init__()
        self.config = config
        self.neuron_groups = nn.ModuleDict()
        self.neuron_counts = {}
        
        # --- FIXED: Robust Initialization ---
        total_neurons = max(1, config.max_neurons)
        remaining = total_neurons
        
        # Use provided configs OR default if empty
        configs = config.spiking_configs
        if not configs:
            # Default configuration if none provided
            configs = [
                SpikingNeuronConfig(
                    neuron_type="pyramidal_default", 
                    structure="standard", 
                    neurotransmitter="glutamate", 
                    percentage=100.0,
                    threshold=0.5
                )
            ]
            
        for i, cfg in enumerate(configs):
            if i == len(configs) - 1:
                count = remaining
            else:
                count = int(total_neurons * cfg.percentage / 100.0)
                count = max(1, count) # Ensure at least 1 neuron
            
            # Prevent over-allocation
            if count > remaining: count = remaining
            if count <= 0: continue
            
            remaining -= count
            self.neuron_counts[cfg.neuron_type] = count
            self.neuron_groups[cfg.neuron_type] = EnhancedSpikingNeuron(
                cfg, count, config.event_bus, config.name
            )
            
        # Ensure we didn't end up empty due to logic errors
        if len(self.neuron_groups) == 0:
             self.neuron_counts["fallback"] = total_neurons
             self.neuron_groups["fallback"] = EnhancedSpikingNeuron(
                 configs[0], total_neurons, config.event_bus, config.name
             )
            
        # Projections
        self.input_projection = AdditionLinear(config.d_model, total_neurons, bias=False)
        self.output_projection = AdditionLinear(total_neurons, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, context: Optional[Dict] = None):
        # 1. Project Input
        zone_input = self.input_projection(x)
        
        # 2. Split and Process
        outputs = []
        start_idx = 0
        
        for name, module in self.neuron_groups.items():
            count = self.neuron_counts[name]
            if count <= 0: continue
            
            group_input = zone_input[..., start_idx : start_idx+count]
            spikes, _, _ = module(group_input)
            outputs.append(spikes)
            start_idx += count
            
        # 3. Combine
        if not outputs:
            # Emergency fallback if something went wrong (shouldn't happen with fix above)
            return torch.zeros_like(x), {'zone_name': self.config.name, 'error': 'no_output'}
            
        combined_spikes = torch.cat(outputs, dim=-1)
        
        # 4. Project Out
        output = self.output_projection(combined_spikes)
        
        with torch.no_grad():
            avg_rate = combined_spikes.float().mean().item()
            
        return output, {
            'zone_name': self.config.name, 
            'avg_firing_rate': avg_rate
        }