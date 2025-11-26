import numpy as np
import time
from typing import List, Tuple, Dict, Any
from .neuron import ActivityState, MaturationStage, Neuron
from ..relays.hippocampus_relay import HippocampusModule
from .spatiotemporal_awareness import (
    HippocampalFormation, SpatialLocation, TemporalEvent, EpisodicMemory
)


class Hippocampus:

    def __init__(self, neuron_count, features=384, input_dim=384, neurogenesis_rate=0.01,
                 enable_spatiotemporal=True, spatial_dimensions=2):
        self.neurons = [Neuron(neuron_id=i, specialization='place_cell', abilities={'memory': 0.95},
                              maturation=MaturationStage.DIFFERENTIATED,
                              activity=ActivityState.RESTING,
                              n_features=features,
                              n_outputs=1)
                        for i in range(neuron_count)]
        # Bound outputs and add mild regularization for stability
        for n in self.neurons:
            n.nlms_head.clamp = (0.0, 1.0)
            n.nlms_head.l2 = 1e-4
        self.neurogenesis_rate = neurogenesis_rate
        self.place_cells = self.neurons
        self.relay: HippocampusModule = HippocampusModule(input_dim=input_dim, memory_strength=0.85)
        
        # Spatio-temporal awareness system
        self.enable_spatiotemporal = enable_spatiotemporal
        if enable_spatiotemporal:
            self.formation = HippocampalFormation(
                spatial_dimensions=spatial_dimensions,
                n_place_cells=min(neuron_count, 100),
                n_time_cells=50,
                n_grid_cells=75
            )
            self.current_cognitive_location = np.zeros(spatial_dimensions)
            self.memory_counter = 0
        else:
            self.formation = None

    async def init_population(self):
        """Initialize all neurons with proper attach calls using trio"""
        import trio
        async with trio.open_nursery() as nursery:
            for neuron in self.neurons:
                nursery.start_soon(neuron.attach)

    def encode(self, x):
        return self.relay.encode(x)

    def encode_memory(self, input_pattern, time=0):
        memories = []
        for neuron in self.place_cells:
            neuron.update_activity(input_pattern, time)
            if neuron.spike_history:
                memories.append(neuron.spike_history[-1][1])
            else:
                memories.append(0)
        return memories

    async def process(self, input_data: np.ndarray) -> dict:
        """Process input through the hippocampus with spatio-temporal awareness"""
        # Encode the input as a memory pattern
        encoded = self.encode(input_data)
        memories = self.encode_memory(input_data)
        
        result = {
            'encoded_pattern': encoded,
            'memories': memories,
            'neurogenesis_rate': self.neurogenesis_rate,
            'neuron_count': len(self.neurons)
        }
        
        # Spatio-temporal processing
        if self.enable_spatiotemporal and self.formation:
            # Update cognitive location based on input features
            # Use first two features as spatial coordinates (normalized)
            if len(input_data) >= 2:
                self.current_cognitive_location = input_data[:2] * 10  # Scale to cognitive space
                self.formation.update_spatial_state(self.current_cognitive_location)
            
            # Process as temporal event
            event_id = f"event_{self.memory_counter}_{int(time.time())}"
            self.formation.process_temporal_event(event_id, input_data)
            
            # Create episodic memory
            memory_id = f"episodic_{self.memory_counter}"
            self.formation.create_episodic_memory(
                memory_id=memory_id,
                event_id=event_id,
                features=input_data,
                associated_experts=['hippocampus']
            )
            self.memory_counter += 1
            
            # Get spatial and temporal context
            spatial_context = self.formation.get_spatial_context()
            temporal_context = self.formation.get_temporal_context()
            
            # Add spatio-temporal information to result
            result.update({
                'spatial_context': spatial_context,
                'temporal_context': temporal_context,
                'episodic_memories_count': len(self.formation.episodic_memories),
                'cognitive_location': self.current_cognitive_location.copy(),
                'theta_phase': spatial_context.get('theta_phase', 0.0)
            })
        
        return result

    def stimulate_neurogenesis(self):
        new_count = int(len(self.neurons) * self.neurogenesis_rate)
        new_neurons = []
        for i in range(new_count):
            new_neuron = Neuron(neuron_id=len(self.neurons), specialization="newborn", abilities={'memory':0.8},
                                maturation=MaturationStage.PROGENITOR,
                                activity=ActivityState.RESTING,
                                n_features=10, n_outputs=1)
            self.neurons.append(new_neuron)
            new_neurons.append(new_neuron)
        return new_neurons

    async def init_weights(self):
        """Initialize weights for all neurons in the hippocampus"""
        for neuron in self.neurons:
            if hasattr(neuron, 'init_weights'):
                await neuron.init_weights()
            elif hasattr(neuron, 'nlms_head') and hasattr(neuron.nlms_head, 'init_weights'):
                await neuron.nlms_head.init_weights()
    
    def retrieve_episodic_memories(self, query_features: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve similar episodic memories based on features and location"""
        if not self.enable_spatiotemporal or not self.formation:
            return []
        
        return self.formation.retrieve_similar_memories(
            query_features=query_features,
            location=self.current_cognitive_location,
            k=k
        )
    
    def get_cognitive_map(self) -> Dict[str, Any]:
        """Get the current cognitive map state"""
        if not self.enable_spatiotemporal or not self.formation:
            return {}
        
        return {
            'spatial_map': dict(self.formation.cognitive_map),
            'temporal_map': dict(self.formation.temporal_map),
            'current_location': self.current_cognitive_location.copy(),
            'n_episodic_memories': len(self.formation.episodic_memories),
            'n_spatial_locations': len(self.formation.spatial_locations)
        }
    
    def decay_episodic_memories(self, decay_rate: float = 0.01) -> None:
        """Apply natural decay to episodic memories"""
        if self.enable_spatiotemporal and self.formation:
            self.formation.decay_memories(decay_rate)
    
    def get_place_cell_activity(self) -> np.ndarray:
        """Get current place cell firing rates"""
        if not self.enable_spatiotemporal or not self.formation:
            return np.array([])
        
        return np.array([cell.firing_rate for cell in self.formation.place_cells])
    
    def get_grid_cell_activity(self) -> np.ndarray:
        """Get current grid cell firing rates"""
        if not self.enable_spatiotemporal or not self.formation:
            return np.array([])
        
        return np.array([cell.firing_rate for cell in self.formation.grid_cells])
    
    def get_time_cell_activity(self) -> np.ndarray:
        """Get current time cell firing rates"""
        if not self.enable_spatiotemporal or not self.formation:
            return np.array([])
        
        return np.array([cell.firing_rate for cell in self.formation.time_cells])
