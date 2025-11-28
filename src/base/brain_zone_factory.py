from typing import Dict, Any, List, Optional, Tuple
import os
import csv
import json
import torch

from src.base.layers import BaseLayer, BaseLayerConfig, BaseLayerImplementation
from src.base.events import EventBus
from src.base.snn_brain_zones import NeuromorphicBrainZone as BrainZone, BrainZoneConfig, BrainZoneType, SpikingNeuronConfig

# -----------------------------
# Pattern CSV integration helpers
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
    """Load compact map of izhikevich key_patterns from the comprehensive JSON.
    Returns keys like 'regular_spiking','fast_spiking','bursting','chattering'.
    """
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
    """Load AdEx precise patterns from precise_patterns_params.json if present."""
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
    """Create SpikingNeuronConfig list for a zone using rows from pattern.csv that match the region.

    Percentages are divided evenly among matched rows; thresholds and other detailed parameters
    keep defaults; neurotransmitter is inferred from neuron type name.
    """
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

    configs: List[SpikingNeuronConfig] = []
    for idx, r in enumerate(rows):
        neuron_type = r.get("Neuron_Type") or "unknown"
        primary = r.get("Primary_Pattern") or "unspecified"
        nt = _guess_neurotransmitter(neuron_type)
        
        # Parse optional Izhi params if present in CSV row
        a=b=c=d=None
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
        if (a is None or b is None or c is None or d is None) and adex_map:
            k = primary.lower().strip()
            # map typical names to adex keys
            alias = {
                'regular spiking (rs)': 'regular_spiking',
                'fast spiking (fs)': 'fast_spiking',
                'intrinsically bursting (ib)': 'intrinsic_bursting',
                'chattering (ch)': 'chattering',
            }
            mapped = alias.get(k, k)
            if mapped in adex_map:
                # AdEx doesn't map directly to Izhikevich a,b,c,d but we can use them as placeholders or ignore
                # For now, we just don't set a,b,c,d if using AdEx, or we could try to approximate
                pass

        pct = norm_pcts[idx] if norm_pcts[idx] is not None else (100.0 / len(rows))
        
        configs.append(SpikingNeuronConfig(
            neuron_type=neuron_type,
            structure=r.get("Dendritic_Structure") or "unspecified",
            neurotransmitter=nt,
            percentage=pct,
            a=a, b=b, c=c, d=d
        ))
        
    return configs


class BrainZoneFactory:
    """Enhanced factory for creating brain zones"""
    
    def create_brain_zone(self, config: BrainZoneConfig, layers: Dict[int, BaseLayer]) -> BrainZone:
        """Create a brain zone with enhanced neuromorphic capabilities"""
        # Ensure all neuromorphic properties are propagated
        new_config = BrainZoneConfig(
            name=config.name, 
            max_neurons=config.max_neurons, 
            min_neurons=config.min_neurons, 
            neuron_type=config.neuron_type, 
            gated=config.gated, 
            num_layers=config.num_layers, 
            base_layer_container_config=config.base_layer_container_config,
            # Propagate enhanced properties
            d_model=config.d_model,
            zone_type=config.zone_type,
            use_spiking=config.use_spiking,
            event_bus=config.event_bus,
            spiking_configs=config.spiking_configs
        )
        return BrainZone(config=new_config, layers=layers)

    def create_neuromorphic_zone(self, zone_name: str, zone_type: BrainZoneType, 
                                d_model: int = 512, max_neurons: int = 512,
                                event_bus: Optional[EventBus] = None,
                                pattern_csv_path: Optional[str] = None,
                                pattern_json_path: Optional[str] = None) -> BrainZone:
        """Create a fully configured neuromorphic brain zone"""
        
        # Optionally derive spiking configs from pattern CSV
        spiking_configs: Optional[List[SpikingNeuronConfig]] = None
        if pattern_csv_path:
            try:
                izh_map = None
                if pattern_json_path:
                    izh_map = load_izhikevich_keypatterns_map(pattern_json_path)
                precise_map = load_precise_adex_map_default()
                spiking_configs = build_spiking_configs_for_zone(zone_type, pattern_csv_path, izh_map, precise_map)
            except Exception:
                spiking_configs = None

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


# Helper functions for easy zone creation
def create_prefrontal_cortex(d_model: int = 512, max_neurons: int = 512, 
                           event_bus: Optional[EventBus] = None,
                           pattern_csv_path: Optional[str] = None,
                           pattern_json_path: Optional[str] = None) -> BrainZone:
    """Create a prefrontal cortex zone optimized for reasoning"""
    factory = BrainZoneFactory()
    return factory.create_neuromorphic_zone(
        "prefrontal_cortex", BrainZoneType.PREFRONTAL_CORTEX, 
        d_model, max_neurons, event_bus, pattern_csv_path, pattern_json_path)

def create_temporal_cortex(d_model: int = 512, max_neurons: int = 512,
                         event_bus: Optional[EventBus] = None,
                         pattern_csv_path: Optional[str] = None,
                         pattern_json_path: Optional[str] = None) -> BrainZone:
    """Create a temporal cortex zone optimized for memory and semantics"""
    factory = BrainZoneFactory()
    return factory.create_neuromorphic_zone(
        "temporal_cortex", BrainZoneType.TEMPORAL_CORTEX,
        d_model, max_neurons, event_bus, pattern_csv_path, pattern_json_path)

def create_hippocampus(d_model: int = 512, max_neurons: int = 384,
                      event_bus: Optional[EventBus] = None,
                      pattern_csv_path: Optional[str] = None,
                      pattern_json_path: Optional[str] = None) -> BrainZone:
    """Create a hippocampus zone optimized for memory formation"""
    factory = BrainZoneFactory()
    return factory.create_neuromorphic_zone(
        "hippocampus", BrainZoneType.HIPPOCAMPUS,
        d_model, max_neurons, event_bus, pattern_csv_path, pattern_json_path)

def create_cerebellum(d_model: int = 512, max_neurons: int = 256,
                     event_bus: Optional[EventBus] = None,
                     pattern_csv_path: Optional[str] = None,
                     pattern_json_path: Optional[str] = None) -> BrainZone:
    """Create a cerebellum zone optimized for fine-tuning"""
    factory = BrainZoneFactory()
    return factory.create_neuromorphic_zone(
        "cerebellum", BrainZoneType.CEREBELLUM,
        d_model, max_neurons, event_bus, pattern_csv_path, pattern_json_path)
