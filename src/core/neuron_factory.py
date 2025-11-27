import uuid
import json
import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union

# Import the optimized JIT-ready neurons
from base.neuron import (
    BaseNeuronConfig, 
    NeuronalState, 
    Synapse, 
    MaturationStage, 
    ActivityState,
    VectorizedLIFNeuron,  # Optimized LIF
    IzhikevichNeuron,     # JIT Izhikevich
    AdExNeuron            # JIT AdEx
)
from base.events import EventBus

class Neuron:
    """
    Bio-mimetic Neuron Data Container (GPU-Native).
    Holds state as PyTorch Tensors to ensure compatibility with the computational graph.
    """
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
    
    # Weights as Tensors
    W_hidden: torch.Tensor
    W_input: torch.Tensor
    W_tau: torch.Tensor
    bias: torch.Tensor
    tau_bias: torch.Tensor
    state: torch.Tensor

    def __init__(self, state: NeuronalState, hidden_dim: int = 256, event_bus: Optional[EventBus] = None):
        self.id = state.id if hasattr(state, 'id') else str(uuid.uuid4())
        self._event_bus = event_bus
        self.config = state.config
        
        # Copy biological metadata
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
        
        # Determine device (default to CUDA if available for speed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Weights as Tensors directly on Device
        # ----------------------------------------------------
        
        # 1. Hidden Weights
        if hasattr(state, 'W_hidden') and state.W_hidden is not None:
            self.W_hidden = self._to_tensor(state.W_hidden, device)
        else:
            self.W_hidden = torch.randn(hidden_dim, hidden_dim, device=device) * 0.1
            
        # 2. Input Weights
        if hasattr(state, 'W_input') and state.W_input is not None:
            self.W_input = self._to_tensor(state.W_input, device)
        else:
            self.W_input = torch.randn(hidden_dim, state.config.input_dim, device=device) * 0.1
            
        # 3. Tau Weights
        if hasattr(state, 'W_tau') and state.W_tau is not None:
            self.W_tau = self._to_tensor(state.W_tau, device)
        else:
            self.W_tau = torch.randn(hidden_dim, state.config.input_dim, device=device) * 0.1
            
        # 4. Biases
        if hasattr(state, 'bias') and state.bias is not None:
            self.bias = self._to_tensor(state.bias, device)
        else:
            self.bias = torch.zeros(hidden_dim, device=device)
            
        if hasattr(state, 'tau_bias') and state.tau_bias is not None:
            self.tau_bias = self._to_tensor(state.tau_bias, device)
        else:
            self.tau_bias = torch.zeros(hidden_dim, device=device)
            
        # 5. Dynamic State
        if hasattr(state, 'state') and state.state is not None:
            self.state = self._to_tensor(state.state, device)
        else:
            self.state = torch.zeros(hidden_dim, device=device)

    def _to_tensor(self, data, device):
        """Helper to safely convert numpy/lists to tensor on correct device"""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        return torch.tensor(data, dtype=torch.float32, device=device)

    def fire(self, inputs: Optional[torch.Tensor] = None, spike_value: float = 0.0) -> None:
        """Simulate firing event."""
        if self._event_bus is None:
            return
            
        details: Dict[str, Any] = {
            'neuron_id': self.id,
            'maturation_stage': str(self.maturation_stage),
            'activity_state': str(self.activity_state),
            'spike_value': spike_value,
        }
        
        if inputs is not None:
            # Handle tensor inputs for logging
            try:
                if isinstance(inputs, torch.Tensor):
                    details['input_norm'] = float(torch.norm(inputs).item())
                    details['input_dim'] = int(inputs.shape[-1])
                else:
                    # Fallback for legacy numpy inputs
                    import numpy as np
                    details['input_norm'] = float(np.linalg.norm(inputs))
            except Exception:
                pass
                
        self._event_bus.broadcast_neuron_fired(details)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'config': self.config,
            'synapse': self.synapse,
            'maturation_stage': self.maturation_stage,
            'activity_state': self.activity_state
        }


class NeuronFactory:
    """
    Factory for creating GPU-native Neurons and JIT-compiled Spiking Modules.
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 384, 
                 dt: float = 0.02, tau_min: float = 0.02, tau_max: float = 2.0, **kwargs: Any):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Base config for state container
        self.neuron_config = BaseNeuronConfig(input_dim, hidden_dim, dt, tau_min, tau_max)
        self.synapse = Synapse(target=None, weight=0.0)
        self.maturation_stage = MaturationStage.PROGENITOR
        self.activity_state = ActivityState.RESTING
        
        # Load biological patterns
        self.neuron_patterns = self._load_patterns()

    def _load_patterns(self) -> Dict[str, Any]:
        """Load neuron firing patterns from configuration file"""
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

    def create_neuron(self, neuron_state: NeuronalState = None, event_bus: Optional[EventBus] = None) -> Neuron:
        """Create a data-container Neuron (for state tracking, not computation)."""
        if neuron_state is None:
            neuron_state = NeuronalState(
                kind="generic",
                position=(0,0,0),
                W_hidden=None, # Will be init by Neuron class
                W_input=None,
                W_tau=None,
                bias=None,
                tau_bias=None,
                state=None
            )
            # Inject config
            neuron_state.config = self.neuron_config
            neuron_state.id = self._generate_neuron_id()
            
            # Defaults
            neuron_state.synapse = self.synapse
            neuron_state.maturation_stage = self.maturation_stage
            neuron_state.activity_state = self.activity_state
        
        return Neuron(neuron_state, self.hidden_dim, event_bus=event_bus)

    # ---------------------------------------------------------------------
    # Optimized Spiking Neuron Creation (JIT Modules)
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
        pattern: str = "regular_spiking",
        params: Optional[Dict[str, float]] = None,
    ) -> nn.Module:
        """
        Create a JIT-compiled spiking neuron module.
        
        Args:
            model: "lif", "izhikevich", or "adex"
            d_model: Input dimensionality 
            pattern: Firing pattern name (for Izhikevich/AdEx)
        """

        class _SpikingWrapper(nn.Module):
            """Wraps JIT neurons to handle dimensionality reduction if needed."""
            def __init__(self, cell: nn.Module, label: str = "spiking_unit"):
                super().__init__()
                self.cell = cell
                self.label = label

            def forward(self, x: torch.Tensor):
                # Ensure input is [Batch, Time] or [Batch, Time, Dim]
                # If input has feature dim > 1 but cell expects scalar current, reduce it
                # Izhikevich/AdEx usually driven by scalar current I per neuron
                
                # Case 1: x is [Batch, Time, Features] -> Reduce to [Batch, Time]
                if x.dim() == 3 and not hasattr(self.cell, 'size'): 
                    # If cell is scalar (single neuron params), mean over features
                    # If using VectorizedLIF, it handles features natively
                    pass
                
                # Check for VectorizedLIF (handles dim natively)
                if isinstance(self.cell, VectorizedLIFNeuron):
                    # Expects [Batch, Features] (single step) or handle loop externally?
                    # The optimized VectorizedLIF in src/base/neuron.py is single-step.
                    # If we have sequence, we need to loop or use a JIT sequence wrapper.
                    pass

                return self.cell(x)

        model_key = (model or "lif").lower()
        
        # 1. Izhikevich Model (JIT)
        if model_key == "izhikevich":
            p = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 6.0, "dt": 0.2}
            
            # Pattern loading logic...
            izh_model_data = self.neuron_patterns.get("1_izhikevich", {})
            key_patterns = izh_model_data.get("key_patterns", {})
            
            if pattern in key_patterns:
                p.update(key_patterns[pattern])
            elif key_patterns:
                # Fallback
                p.update(key_patterns.get("regular_spiking", next(iter(key_patterns.values()))))
            
            if params: p.update(params)
                
            cell = IzhikevichNeuron(
                a=float(p.get("a")), b=float(p.get("b")),
                c=float(p.get("c")), d=float(p.get("d")),
                dt=float(p.get("dt"))
            )
            # Wrapper not strictly needed if calling forward_sequence directly, 
            # but maintains API compatibility
            return cell

        # 2. AdEx Model (JIT)
        elif model_key == "adex":
            p = {
                "C": 200.0, "g_L": 10.0, "E_L": -70.0, "V_T": -50.0, 
                "Delta_T": 2.0, "tau_w": 120.0, "a": 0.0, "b": 5.0, 
                "R": 1.0, "V_reset": -58.0, "dt": 0.1
            }
            
            if pattern == "bursting":
                p.update({"a": 2.0, "b": 10.0, "tau_w": 100.0, "V_reset": -46.0})
            elif pattern == "fast_spiking":
                p.update({"a": 0.0, "b": 0.0, "tau_w": 30.0, "C": 100.0})
            
            if params: p.update(params)
                
            cell = AdExNeuron(
                C=float(p.get("C")), g_L=float(p.get("g_L")), E_L=float(p.get("E_L")),
                V_T=float(p.get("V_T")), Delta_T=float(p.get("Delta_T")), 
                tau_w=float(p.get("tau_w")), a=float(p.get("a")), b=float(p.get("b")),
                R=float(p.get("R")), V_reset=float(p.get("V_reset")), dt=float(p.get("dt"))
            )
            return cell

        # 3. Default: Vectorized LIF
        # Note: VectorizedLIFNeuron expects 'size' (d_model)
        cell = VectorizedLIFNeuron(
            size=d_model,
            beta=float(beta), 
            threshold=float(threshold), 
            init_slope=float(init_slope),
            event_bus=event_bus, 
            name=name or "lif_neuron"
        )
        return cell