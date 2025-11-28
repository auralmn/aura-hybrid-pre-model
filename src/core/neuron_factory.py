import uuid
import json
import os
from base import BaseNeuronConfig, NeuronalState, Synapse, MaturationStage, ActivityState

import numpy as np

from typing import Dict, Any, List, Optional, Union
from base.events import EventBus
import torch
import torch.nn as nn
from base.neuron import AdaptiveLIFNeuron, IzhikevichNeuron, AdExNeuron

class Neuron:

    id: str
    config: BaseNeuronConfig
    synapse: Synapse
    maturation_stage: MaturationStage
    activity_state: ActivityState
    membrane_potential: float
    gene_expression: Dict[str, float]
    cell_cycle: str
    maturation: str
    activity: str
    connections: List[str]
    environment: Dict[str, float]
    plasticity: Dict[str, float]
    fatigue: float
    W_hidden: np.ndarray
    W_input: np.ndarray
    W_tau: np.ndarray
    bias: np.ndarray
    tau_bias: np.ndarray
    state: np.ndarray

    def __init__(self, state: NeuronalState, hidden_dim: int=256, event_bus: Optional[EventBus]=None):
        # Fixed: Use proper ID assignment from state or generate new one
        self.id = state.id if hasattr(state, 'id') else str(uuid.uuid4())
        self._event_bus = event_bus
        self.config = state.config
        self.synapse = state.synapse
        self.maturation_stage = state.maturation_stage
        self.activity_state = state.activity_state
        self.membrane_potential = state.membrane_potential
        self.gene_expression = state.gene_expression
        self.cell_cycle = state.cell_cycle
        self.maturation = state.maturation
        self.activity = state.activity
        self.connections = state.connections
        self.environment = state.environment
        self.plasticity = state.plasticity
        self.fatigue = state.fatigue
        
        # Fixed: Removed redundant assignments and simplified initialization
        # Initialize weights with fallback values if state doesn't have them
        if hasattr(state, 'W_hidden') and state.W_hidden is not None:
            self.W_hidden = state.W_hidden
        else:
            self.W_hidden = np.random.normal(0, 0.1, (hidden_dim, hidden_dim))
            
        if hasattr(state, 'W_input') and state.W_input is not None:
            self.W_input = state.W_input
        else:
            self.W_input = np.random.normal(0, 0.1, (hidden_dim, state.config.input_dim))
            
        if hasattr(state, 'W_tau') and state.W_tau is not None:
            self.W_tau = state.W_tau
        else:
            self.W_tau = np.random.normal(0, 0.1, (hidden_dim, state.config.input_dim))
            
        if hasattr(state, 'bias') and state.bias is not None:
            self.bias = state.bias
        else:
            self.bias = np.zeros(hidden_dim)
            
        if hasattr(state, 'tau_bias') and state.tau_bias is not None:
            self.tau_bias = state.tau_bias
        else:
            self.tau_bias = np.zeros(hidden_dim)
            
        if hasattr(state, 'state') and state.state is not None:
            self.state = state.state
        else:
            self.state = np.zeros(hidden_dim)

    def fire(self, inputs: Optional[np.ndarray] = None, spike_value: float = 0.0) -> None:
        """Simulate a neuron firing and emit an event with details.

        This method does not alter network state beyond optional local state hints; it's
        intended for observability hooks.
        """
        if self._event_bus is None:
            return
        details: Dict[str, Any] = {
            'neuron_id': self.id,
            'maturation_stage': str(self.maturation_stage),
            'activity_state': str(self.activity_state),
            'spike_value': spike_value,
        }
        if inputs is not None:
            try:
                details['input_norm'] = float(np.linalg.norm(inputs))
                details['input_dim'] = int(inputs.shape[-1]) if hasattr(inputs, 'shape') else None
            except Exception:
                pass
        self._event_bus.broadcast_neuron_fired(details)

    def get_stats(self) -> Dict[str, Any]:
        # Fixed: Added missing closing brace
        return {
            'id': self.id,
            'config': self.config,
            'synapse': self.synapse,
            'maturation_stage': self.maturation_stage,
            'activity_state': self.activity_state
        }


class NeuronFactory:
    def __init__(self, input_dim: int=128, hidden_dim: int=256, output_dim: int=384, dt: float = 0.02, tau_min: float = 0.02, tau_max: float = 2.0, **kwargs: Any):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.neuron_config = BaseNeuronConfig(input_dim, hidden_dim, dt, tau_min, tau_max)
        self.synapse = Synapse(target=None, weight=0.0)
        self.maturation_stage = MaturationStage.PROGENITOR
        self.activity_state = ActivityState.RESTING
        
        # Load bio-plausible patterns
        self.neuron_patterns = self._load_patterns()

    def _load_patterns(self) -> Dict[str, Any]:
        """Load neuron firing patterns from configuration file"""
        # Try to find the patterns file in project root
        possible_paths = [
            "all_neurons_patterns.json",
            "../all_neurons_patterns.json",
            os.path.join(os.path.dirname(__file__), "../../all_neurons_patterns.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        return data.get("comprehensive_neuron_models", {}).get("models", {})
                except Exception as e:
                    print(f"Warning: Failed to load neuron patterns from {path}: {e}")
        
        return {}

    def _generate_neuron_id(self) -> str:
        return str(uuid.uuid4())

    def create_neuron(self, neuron_state: NeuronalState = None, event_bus: Optional[EventBus]=None) -> Neuron:
        # Fixed: Create proper NeuronalState object if not provided
        if neuron_state is None:
            # Create a default NeuronalState with the factory's configuration
            neuron_state = NeuronalState(
                id=self._generate_neuron_id(),
                config=self.neuron_config,
                synapse=self.synapse,
                maturation_stage=self.maturation_stage,
                activity_state=self.activity_state,
                membrane_potential=0.0,
                gene_expression={},
                cell_cycle="G1",
                maturation="immature",
                activity="resting",
                connections=[],
                environment={},
                plasticity={},
                fatigue=0.0,
                # Initialize these as None - Neuron will create defaults
                W_hidden=None,
                W_input=None,
                W_tau=None,
                bias=None,
                tau_bias=None,
                state=None
            )
        
        if event_bus is None:
            return Neuron(neuron_state, self.hidden_dim)
        return Neuron(neuron_state, self.hidden_dim, event_bus=event_bus)

    # ---------------------------------------------------------------------
    # Spiking neuron creation (normalized defaults; Option B)
    # ---------------------------------------------------------------------
    def create_spiking_neuron(
        self,
        model: str = "lif",
        d_model: int = 1,
        event_bus: Optional[EventBus] = None,
        *,
        threshold: float = 0.5,
        beta: float = 0.8,
        init_slope: float = 15.0,
        name: Optional[str] = None,
        # Enhanced: Allow specifying biological pattern
        pattern: str = "regular_spiking",
        params: Optional[Dict[str, float]] = None,
    ) -> nn.Module:
        """Create a simple spiking neuron module for ad-hoc use.

        Args:
            model: "lif", "izhikevich", or "adex"
            d_model: Input dimensionality (will be reduced to current if >1)
            pattern: Firing pattern name (e.g. "bursting", "fast_spiking") for Izhikevich/AdEx
            params: Manual parameter overrides
        """

        class _SpikingWrapper(nn.Module):
            def __init__(self, cell: nn.Module, label: str = "spiking_unit"):
                super().__init__()
                self.cell = cell
                self.label = label

            def forward(self, x: torch.Tensor):
                # Reduce inputs to a current per-step (mean over feature dims)
                if isinstance(x, torch.Tensor):
                    if x.dim() == 3:  # [B, T, D]
                        cur = x.mean(dim=-1)
                    elif x.dim() == 2:  # [B, D]
                        cur = x.mean(dim=-1, keepdim=False)
                    else:  # [T] or [B]
                        cur = x
                else:
                    # Fallback random low drive
                    cur = torch.zeros(1)
                # Ensure shape [T] or [B, T]
                if cur.dim() == 1:
                    return self.cell(cur)
                # Merge batch as steps for simple invocation
                b, t = cur.shape[0], (cur.shape[1] if cur.dim() == 2 else 1)
                out_spk, out_mem = [], []
                if cur.dim() == 2:
                    for i in range(b):
                        spk = self.cell(cur[i])
                        out_spk.append(spk[0] if isinstance(spk, tuple) else spk)
                        out_mem.append((spk[1] if isinstance(spk, tuple) else spk))
                    return torch.stack(out_spk, dim=0), torch.stack(out_mem, dim=0)
                return self.cell(cur)

        model_key = (model or "lif").lower()
        
        # 1. Izhikevich Model
        if model_key == "izhikevich":
            # Default parameters
            p = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 6.0, "dt": 0.2}
            
            # Load from patterns if available
            izh_model_data = self.neuron_patterns.get("1_izhikevich", {})
            key_patterns = izh_model_data.get("key_patterns", {})
            
            if pattern in key_patterns:
                pat_params = key_patterns[pattern]
                p.update(pat_params)
            elif pattern == "regular_spiking": # Fallback explicit check
                pass
            else:
                # Fallback to regular spiking if pattern not found
                if key_patterns:
                    # Default to first available or regular
                    pat_params = key_patterns.get("regular_spiking", next(iter(key_patterns.values())))
                    p.update(pat_params)
            
            # Manual overrides
            if params:
                p.update(params)
                
            cell = IzhikevichNeuron(
                a=float(p.get("a", 0.02)), b=float(p.get("b", 0.2)),
                c=float(p.get("c", -65.0)), d=float(p.get("d", 6.0)),
                dt=float(p.get("dt", 0.2))
            )
            return _SpikingWrapper(cell, label=name or f"izh_{pattern}")

        # 2. AdEx Model (Adaptive Exponential)
        elif model_key == "adex":
            # Default parameters (regular spikingish)
            p = {
                "C": 200.0, "g_L": 10.0, "E_L": -70.0, "V_T": -50.0, 
                "Delta_T": 2.0, "tau_w": 120.0, "a": 0.0, "b": 5.0, 
                "R": 1.0, "V_reset": -58.0, "dt": 0.1
            }
            
            # Load from patterns
            adex_data = self.neuron_patterns.get("11_adaptive_exponential", {})
            firing_patterns = adex_data.get("firing_patterns", {})
            
            # Note: AdEx JSON structure is a bit different (nested descriptions),
            # we might need to map descriptive keys to numeric values or use presets.
            # The JSON provides ranges/descriptions more than exact sets for some patterns.
            # We'll implement some presets based on standard AdEx types.
            
            if pattern == "bursting":
                p.update({"a": 2.0, "b": 10.0, "tau_w": 100.0, "V_reset": -46.0})
            elif pattern == "fast_spiking":
                p.update({"a": 0.0, "b": 0.0, "tau_w": 30.0, "C": 100.0}) # Low adaptation
            elif pattern == "adapting":
                p.update({"a": 4.0, "b": 20.0, "tau_w": 150.0})
            
            # Manual overrides
            if params:
                p.update(params)
                
            cell = AdExNeuron(
                C=float(p.get("C", 200.0)),
                g_L=float(p.get("g_L", 10.0)),
                E_L=float(p.get("E_L", -70.0)),
                V_T=float(p.get("V_T", -50.0)),
                Delta_T=float(p.get("Delta_T", 2.0)),
                tau_w=float(p.get("tau_w", 120.0)),
                a=float(p.get("a", 0.0)),
                b=float(p.get("b", 0.0)),
                R=float(p.get("R", 1.0)),
                V_reset=float(p.get("V_reset", -65.0)),
                dt=float(p.get("dt", 0.1))
            )
            return _SpikingWrapper(cell, label=name or f"adex_{pattern}")

        # 3. Default: LIF with learnable surrogate
        cell = AdaptiveLIFNeuron(
            beta=float(beta), threshold=float(threshold), init_slope=float(init_slope),
            event_bus=event_bus, name=name or "lif_neuron",
        )
        return _SpikingWrapper(cell, label=name or "lif_neuron")
