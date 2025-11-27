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
from base.neuron import AdaptiveLIFNeuron, LearnableSurrogateGradient, NeuronalState, IzhikevichNeuron, AdExNeuron
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
    """
    Wrapper connecting Config to JIT Neurons.
    Supports sequence processing natively.
    """
    def __init__(self, config, d_model, event_bus=None, zone_name=None):
        super().__init__()
        self.config = config
        self.d_model = d_model
        
        # Select JIT Kernel
        if config.a is not None: # Izhikevich
            self.core = IzhikevichNeuron(
                a=config.a, b=config.b, c=config.c, d=config.d, dt=config.dt
            )
            self.mode = 'izh'
        elif config.model_type == 'adex': # AdEx
            p = config.model_params or {}
            self.core = AdExNeuron(**p)
            self.mode = 'adex'
        else: # LIF
            self.core = VectorizedLIFNeuron(
                size=d_model, beta=config.beta_decay, threshold=config.threshold, 
                init_slope=config.init_surrogate_slope
            )
            self.mode = 'lif'
            
        self.register_buffer('homeo_i', torch.tensor(0.0)) # Homeostasis bias

    def forward(self, x):
        # x: [Batch, Time, Dim] or [Batch, Dim]
        is_seq = x.dim() == 3
        
        # Add homeostatic bias
        x_eff = x + self.homeo_i
        
        if self.mode in ('izh', 'adex'):
            # These support forward_sequence natively
            if is_seq:
                spikes = self.core.forward_sequence(x_eff)
                return spikes, None, {}
            else:
                # Single step
                # Expand dims to [B, 1, D] -> process -> squeeze
                spikes = self.core.forward_sequence(x_eff.unsqueeze(1)).squeeze(1)
                return spikes, None, {}
        else:
            # LIF
            if is_seq:
                # LIF needs manual loop or scan. 
                # For now, simplistic loop (LIF is fast enough or use JIT LIF if implemented)
                spikes_list = []
                for t in range(x.shape[1]):
                    s, _ = self.core(x_eff[:, t])
                    spikes_list.append(s)
                return torch.stack(spikes_list, dim=1), None, {}
            else:
                spikes, mem = self.core(x_eff)
                return spikes, mem, {}

class NeuromorphicBrainZone(nn.Module):
    """
    Optimized Brain Zone.
    Replaces explicit time-loop with tensor operations.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.neuron_groups = nn.ModuleDict()
        self.neuron_counts = {}
        
        # [Initialization logic: determine counts per group]
        # ... (Same as original, mapping config to EnhancedSpikingNeuron)
        
        # Projections
        self.input_projection = AdditionLinear(config.d_model, config.max_neurons)
        self.output_projection = AdditionLinear(config.max_neurons, config.d_model)

    def forward(self, x: torch.Tensor, context: Optional[Dict] = None):
        # x: [Batch, Time, D_model] or [Batch, D_model]
        
        # 1. Project Input (Vectorized)
        # [B, T, D] -> [B, T, Total_Neurons]
        zone_input = self.input_projection(x)
        
        # 2. Split and Process Groups in Parallel
        # Since we removed the time loop, we just split the tensor along the Neuron dim
        outputs = []
        start_idx = 0
        
        for name, module in self.neuron_groups.items():
            count = self.neuron_counts[name]
            # Slice the relevant features for this group
            # Works for both [B, N] and [B, T, N]
            group_input = zone_input[..., start_idx : start_idx+count]
            
            # Process sequence fully on GPU
            spikes, _, _ = module(group_input)
            outputs.append(spikes)
            
            start_idx += count
            
        # 3. Combine and Project Out
        # Concatenate along neuron dimension
        combined_spikes = torch.cat(outputs, dim=-1)
        
        # Project back to D_model
        output = self.output_projection(combined_spikes)
        
        return output, {'zone': self.config.name}

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