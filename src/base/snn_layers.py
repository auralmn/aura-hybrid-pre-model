#!/usr/bin/env python3
"""
Enhanced layers.py with neuromorphic spiking layer support
Builds on your existing BaseLayer architecture
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass
import torch
import torch.nn as nn

from base.neuron import AdaptiveLIFNeuron, LearnableSurrogateGradient, VectorizedLIFNeuron
from base.events import EventBus

@dataclass
class BaseLayerConfig:
    """Enhanced base layer configuration"""
    name: str
    input_dim: int
    output_dim: int
    dt: float = 0.02
    tau_min: float = 0.02
    tau_max: float = 2.0
    
    # Enhanced neuromorphic properties
    use_spiking: bool = False
    spike_threshold: float = 0.6
    beta_decay: float = 0.9
    surrogate_slope: float = 5.0
    neuron_type: str = "LIF"
    dropout_rate: float = 0.1

class BaseLayer(ABC):
    """Enhanced base layer interface"""
    config: BaseLayerConfig
    
    @abstractmethod
    def forward(self, x: Any) -> Any:
        pass
    
    @abstractmethod
    def get_config(self) -> BaseLayerConfig:
        pass
    
    def reset_state(self):
        """Reset layer state - used for spiking layers"""
        pass
    
    def get_activity_stats(self) -> Dict[str, Any]:
        """Get layer activity statistics"""
        return {}

@dataclass
class BaseLayerContainerConfig:
    """Enhanced layer container configuration"""
    num_layers: int = 2
    layer_type: str = "sparse"
    layer_config: BaseLayerConfig = None
    
    # Enhanced properties for neuromorphic layers
    use_neuromorphic: bool = False
    connection_sparsity: float = 0.1
    inter_layer_delays: bool = False
    adaptive_thresholds: bool = True

class SpikingLayer(BaseLayer, nn.Module):
    """Spiking neural network layer implementation"""
    
    def __init__(self, config: BaseLayerConfig, event_bus: Optional[EventBus] = None):
        super().__init__()
        self.config = config
        self.event_bus = event_bus
        
        # Linear transformation
        self.linear = nn.Linear(config.input_dim, config.output_dim, bias=True)
        
        # Spiking neurons
        if config.use_spiking:
            # Vectorized implementation for performance
            self.spiking_neurons = VectorizedLIFNeuron(
                size=config.output_dim,
                beta=max(0.5, float(config.beta_decay)),
                threshold=max(0.1, float(config.spike_threshold)),
                init_slope=config.surrogate_slope,
                event_bus=event_bus,
                name=f"{config.name}_neurons"
            )
        else:
            self.spiking_neurons = None
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Activity tracking
        self.register_buffer('spike_count', torch.zeros(config.output_dim))
        self.register_buffer('total_steps', torch.zeros(1))
        
        # Layer statistics
        self.layer_stats = {
            'total_activations': 0,
            'avg_firing_rate': 0.0,
            'spike_count_history': [],
            'membrane_potential_history': []
        }
    
    def get_config(self) -> BaseLayerConfig:
        return self.config
    
    def reset_state(self):
        """Reset spiking neuron states"""
        if self.spiking_neurons:
            self.spiking_neurons.reset_mem()
        
        # Reset activity tracking
        self.spike_count.zero_()
        self.total_steps.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional spiking dynamics"""
        
        # Linear transformation
        linear_output = self.linear(x)
        linear_output = self.dropout(linear_output)
        
        if not self.config.use_spiking or self.spiking_neurons is None:
            # Standard non-spiking operation
            return torch.relu(linear_output)
        
        # Spiking operation (Vectorized)
        combined_spikes, combined_membrane = self.spiking_neurons(linear_output)
        
        # Update activity tracking
        self._update_activity_stats(combined_spikes)
        
        return combined_spikes
    
    def _update_activity_stats(self, spikes: torch.Tensor):
        """Update layer activity statistics"""
        with torch.no_grad():
            # Update spike counts
            batch_spikes = spikes.sum(dim=(0, 1)) if spikes.dim() > 2 else spikes.sum(dim=0)
            self.spike_count += batch_spikes
            self.total_steps += 1
            
            # Calculate firing rates
            if self.total_steps > 0:
                firing_rates = self.spike_count / self.total_steps
                avg_firing_rate = firing_rates.mean().item()
                
                self.layer_stats['avg_firing_rate'] = avg_firing_rate
                self.layer_stats['total_activations'] = int(self.total_steps.item())
                
                # Keep history (limited to last 100 entries)
                self.layer_stats['spike_count_history'].append(batch_spikes.sum().item())
                if len(self.layer_stats['spike_count_history']) > 100:
                    self.layer_stats['spike_count_history'] = self.layer_stats['spike_count_history'][-100:]
    
    def get_activity_stats(self) -> Dict[str, Any]:
        """Get comprehensive layer activity statistics"""
        stats = self.layer_stats.copy()
        
        if self.spiking_neurons and self.total_steps > 0:
            # Aggregated statistics for vectorized layer
            firing_rates = (self.spike_count / self.total_steps)
            stats['neuron_stats'] = {
                'avg_firing_rate': firing_rates.mean().item(),
                'max_firing_rate': firing_rates.max().item(),
                'min_firing_rate': firing_rates.min().item(),
                'avg_threshold': self.spiking_neurons.threshold.mean().item(),
                'avg_slope': self.spiking_neurons.slope.mean().item()
            }
            stats['config'] = {
                'layer_name': self.config.name,
                'input_dim': self.config.input_dim,
                'output_dim': self.config.output_dim,
                'use_spiking': self.config.use_spiking
            }
        
        return stats

class AdaptiveSpikingLayer(SpikingLayer):
    """Advanced spiking layer with adaptive properties"""
    
    def __init__(self, config: BaseLayerConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        
        # Adaptive threshold mechanism
        if config.use_spiking:
            self.adaptive_threshold = nn.Parameter(torch.ones(config.output_dim) * config.spike_threshold)
            self.threshold_adaptation_rate = 0.01
        
        # Lateral inhibition
        self.lateral_inhibition = nn.Parameter(torch.zeros(config.output_dim, config.output_dim))
        nn.init.normal_(self.lateral_inhibition, mean=0, std=0.1)
        
        # Homeostatic plasticity
        self.target_firing_rate = 0.1  # Target 10% firing rate
        self.homeostasis_strength = 0.001
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive mechanisms"""
        
        # Standard forward pass
        output = super().forward(x)
        
        if not self.config.use_spiking:
            return output
        
        # Apply lateral inhibition
        inhibited_output = output - torch.matmul(output, self.lateral_inhibition.abs())
        inhibited_output = torch.clamp(inhibited_output, min=0)
        
        # Adaptive threshold adjustment (homeostatic plasticity)
        self._adapt_thresholds()
        
        return inhibited_output
    
    def _adapt_thresholds(self):
        """Adapt thresholds based on firing rates for homeostasis"""
        if not hasattr(self, 'spiking_neurons') or not self.spiking_neurons:
            return
        
        with torch.no_grad():
            current_rates = self.spike_count / max(1, self.total_steps.item())
            
            # Adjust thresholds to maintain target firing rate
            rate_error = current_rates - self.target_firing_rate
            threshold_adjustment = rate_error * self.homeostasis_strength
            
            # Update neuron thresholds
            if hasattr(self.spiking_neurons, 'threshold'):
                self.spiking_neurons.threshold.data += threshold_adjustment
                # Clamp to reasonable range
                self.spiking_neurons.threshold.data.clamp_(-2.0, 2.0)

class ReservoirLayer(BaseLayer, nn.Module):
    """Liquid State Machine / Echo State Network layer"""
    
    def __init__(self, config: BaseLayerConfig, 
                 reservoir_size: int = 1000,
                 connectivity: float = 0.1,
                 spectral_radius: float = 0.95):
        super().__init__()
        self.config = config
        self.reservoir_size = reservoir_size
        
        # Input weights
        self.W_in = nn.Parameter(torch.randn(reservoir_size, config.input_dim) * 0.5)
        
        # Reservoir weights (fixed, not trained)
        reservoir_weights = torch.randn(reservoir_size, reservoir_size)
        # Apply sparsity
        mask = torch.rand(reservoir_size, reservoir_size) < connectivity
        reservoir_weights = reservoir_weights * mask.float()
        # Scale to desired spectral radius
        eigenvalues = torch.linalg.eigvals(reservoir_weights).real
        current_radius = torch.max(torch.abs(eigenvalues))
        reservoir_weights = reservoir_weights * (spectral_radius / current_radius)
        
        self.register_buffer('W_reservoir', reservoir_weights)
        
        # Output weights (trainable)
        self.W_out = nn.Linear(reservoir_size, config.output_dim)
        
        # Reservoir state
        self.register_buffer('reservoir_state', torch.zeros(1, reservoir_size))
        
        # Leakage parameter
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def get_config(self) -> BaseLayerConfig:
        return self.config
    
    def reset_state(self):
        """Reset reservoir state"""
        self.reservoir_state.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reservoir computing forward pass"""
        batch_size, seq_len = x.shape[:2]
        
        # Expand reservoir state for batch
        if self.reservoir_state.size(0) != batch_size:
            self.reservoir_state = self.reservoir_state.expand(batch_size, -1).contiguous()
        
        outputs = []
        
        for t in range(seq_len):
            # Input transformation
            input_activation = torch.matmul(x[:, t], self.W_in.T)
            
            # Reservoir dynamics
            reservoir_activation = torch.matmul(self.reservoir_state, self.W_reservoir.T)
            
            # Update reservoir state (leaky integration)
            self.reservoir_state = ((1 - self.alpha) * self.reservoir_state + 
                                   self.alpha * torch.tanh(input_activation + reservoir_activation))
            
            # Output
            output = self.W_out(self.reservoir_state)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

class BaseLayerImplementation(BaseLayer):
    """Enhanced base layer implementation with optional spiking"""
    
    def __init__(self, config: BaseLayerConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.event_bus = event_bus
        
        if config.use_spiking:
            # Create spiking version
            self.layer_impl = SpikingLayer(config, event_bus)
        else:
            # Create standard version
            self.layer_impl = self._create_standard_layer()
    
    def _create_standard_layer(self):
        """Create standard non-spiking layer"""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.output_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate)
        )
    
    def get_config(self) -> BaseLayerConfig:
        return self.config
    
    def forward(self, x: Any) -> Any:
        if hasattr(self.layer_impl, 'forward'):
            return self.layer_impl.forward(x)
        else:
            return self.layer_impl(x)
    
    def reset_state(self):
        if hasattr(self.layer_impl, 'reset_state'):
            self.layer_impl.reset_state()
    
    def get_activity_stats(self) -> Dict[str, Any]:
        if hasattr(self.layer_impl, 'get_activity_stats'):
            return self.layer_impl.get_activity_stats()
        return {'layer_type': 'standard', 'config': self.config.name}

class BaseLayerContainer:
    """Enhanced layer container with neuromorphic capabilities"""
    config: BaseLayerContainerConfig
    layers: Dict[int, BaseLayer]
    
    def __init__(self, config: BaseLayerContainerConfig, layers: Dict[int, BaseLayer]):
        self.config = config
        self.layers = layers
        
        # Container-level statistics
        self.container_stats = {
            'total_layers': len(layers),
            'spiking_layers': 0,
            'total_parameters': 0,
            'avg_layer_activity': 0.0
        }
        
        # Count spiking layers
        for layer in layers.values():
            if hasattr(layer, 'config') and layer.config.use_spiking:
                self.container_stats['spiking_layers'] += 1
    
    def get_config(self) -> BaseLayerContainerConfig:
        return self.config
    
    def get_layers(self) -> Dict[int, BaseLayer]:
        return self.layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers"""
        for layer_id in sorted(self.layers.keys()):
            layer = self.layers[layer_id]
            x = layer.forward(x)
        return x
    
    def reset_all_states(self):
        """Reset states of all layers"""
        for layer in self.layers.values():
            if hasattr(layer, 'reset_state'):
                layer.reset_state()
    
    def get_container_stats(self) -> Dict[str, Any]:
        """Get comprehensive container statistics"""
        stats = self.container_stats.copy()
        
        # Collect layer statistics
        layer_activities = []
        layer_details = {}
        
        for layer_id, layer in self.layers.items():
            layer_stats = layer.get_activity_stats() if hasattr(layer, 'get_activity_stats') else {}
            layer_details[f'layer_{layer_id}'] = layer_stats
            
            # Extract activity measure
            if 'avg_firing_rate' in layer_stats:
                layer_activities.append(layer_stats['avg_firing_rate'])
            elif 'activity_measure' in layer_stats:
                layer_activities.append(layer_stats['activity_measure'])
        
        stats['layer_details'] = layer_details
        
        if layer_activities:
            stats['avg_layer_activity'] = sum(layer_activities) / len(layer_activities)
            stats['layer_activity_distribution'] = {
                'min': min(layer_activities),
                'max': max(layer_activities),
                'std': torch.std(torch.tensor(layer_activities)).item()
            }
        
        return stats

class BaseLayerFactory:
    """Enhanced factory for creating various layer types"""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus
    
    def create_layer(self, config: BaseLayerConfig) -> BaseLayer:
        """Create layer based on configuration"""
        if config.neuron_type.lower() == "spiking":
            return SpikingLayer(config, self.event_bus)
        elif config.neuron_type.lower() == "adaptive":
            return AdaptiveSpikingLayer(config, self.event_bus)
        elif config.neuron_type.lower() == "reservoir":
            return ReservoirLayer(config)
        else:
            return BaseLayerImplementation(config, self.event_bus)
    
    def create_spiking_layer(self, name: str, input_dim: int, output_dim: int,
                            threshold: float = 1.0, slope: float = 25.0) -> SpikingLayer:
        """Convenience method for creating spiking layers"""
        config = BaseLayerConfig(
            name=name,
            input_dim=input_dim,
            output_dim=output_dim,
            use_spiking=True,
            spike_threshold=threshold,
            surrogate_slope=slope
        )
        return SpikingLayer(config, self.event_bus)
    
    def create_reservoir_layer(self, name: str, input_dim: int, output_dim: int,
                              reservoir_size: int = 1000) -> ReservoirLayer:
        """Convenience method for creating reservoir layers"""
        config = BaseLayerConfig(
            name=name,
            input_dim=input_dim,
            output_dim=output_dim
        )
        return ReservoirLayer(config, reservoir_size)
    
    def create_layer_container(self, config: BaseLayerContainerConfig, 
                              layer_configs: List[BaseLayerConfig]) -> BaseLayerContainer:
        """Create a container with multiple layers"""
        layers = {}
        
        for i, layer_config in enumerate(layer_configs):
            layers[i] = self.create_layer(layer_config)
        
        return BaseLayerContainer(config, layers)

# Helper functions for easy layer creation
def create_neuromorphic_layer_stack(input_dim: int, hidden_dims: List[int], output_dim: int,
                                   use_spiking: bool = True, 
                                   event_bus: Optional[EventBus] = None) -> BaseLayerContainer:
    """Create a stack of neuromorphic layers"""
    
    factory = BaseLayerFactory(event_bus)
    layer_configs = []
    
    # Input layer
    layer_configs.append(BaseLayerConfig(
        name="input_layer",
        input_dim=input_dim,
        output_dim=hidden_dims[0],
        use_spiking=use_spiking,
        neuron_type="spiking" if use_spiking else "standard"
    ))
    
    # Hidden layers
    for i in range(len(hidden_dims) - 1):
        layer_configs.append(BaseLayerConfig(
            name=f"hidden_layer_{i}",
            input_dim=hidden_dims[i],
            output_dim=hidden_dims[i + 1],
            use_spiking=use_spiking,
            neuron_type="spiking" if use_spiking else "standard"
        ))
    
    # Output layer
    layer_configs.append(BaseLayerConfig(
        name="output_layer",
        input_dim=hidden_dims[-1],
        output_dim=output_dim,
        use_spiking=use_spiking,
        neuron_type="spiking" if use_spiking else "standard"
    ))
    
    container_config = BaseLayerContainerConfig(
        num_layers=len(layer_configs),
        layer_type="neuromorphic",
        use_neuromorphic=use_spiking
    )
    
    return factory.create_layer_container(container_config, layer_configs)