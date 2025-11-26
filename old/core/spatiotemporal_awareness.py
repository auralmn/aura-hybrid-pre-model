# AURA-MOE Hippocampus: Spatio-Temporal Awareness Module
# Handling space-time memory, cognitive maps, and episodic processing

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import deque, defaultdict

# ---------------------- Spatial-Temporal Representations ----------------------

@dataclass
class SpatialLocation:
    """Represents a location in cognitive space"""
    coordinates: np.ndarray  # Multi-dimensional spatial coordinates
    landmarks: List[str] = field(default_factory=list)
    context_features: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    visits: int = 1
    
    def distance_to(self, other: 'SpatialLocation') -> float:
        """Euclidean distance to another location"""
        return float(np.linalg.norm(self.coordinates - other.coordinates))

@dataclass 
class TemporalEvent:
    """Represents an event in temporal sequence"""
    timestamp: float
    event_id: str
    features: np.ndarray
    duration: float = 0.0
    context: Optional[Dict[str, Any]] = None
    
    def time_distance_to(self, other: 'TemporalEvent') -> float:
        """Temporal distance to another event"""
        return abs(self.timestamp - other.timestamp)

@dataclass
class EpisodicMemory:
    """Episodic memory binding spatial and temporal information"""
    memory_id: str
    spatial_location: SpatialLocation
    temporal_event: TemporalEvent
    associated_experts: List[str] = field(default_factory=list)
    strength: float = 1.0
    retrieval_count: int = 0
    
    def decay_strength(self, decay_rate: float = 0.01) -> None:
        """Natural memory decay over time"""
        current_time = time.time()
        elapsed = current_time - self.temporal_event.timestamp
        self.strength *= np.exp(-decay_rate * elapsed / 3600.0)  # Hourly decay

# ---------------------- Place Cells (Spatial Processing) ----------------------

class PlaceCell:
    """Individual place cell with spatial receptive field"""
    
    def __init__(self, center: np.ndarray, radius: float = 1.0):
        self.center = center  # Spatial coordinates of receptive field center
        self.radius = radius  # Size of receptive field
        self.firing_rate = 0.0
        self.theta_phase = 0.0  # Theta oscillation phase
        self.max_firing_rate = 20.0  # Hz
        
    def compute_firing_rate(self, location: np.ndarray) -> float:
        """Compute firing rate based on distance from center"""
        distance = np.linalg.norm(location - self.center)
        if distance <= self.radius:
            # Gaussian firing field
            firing_rate = self.max_firing_rate * np.exp(-(distance**2) / (2 * (self.radius/3)**2))
            self.firing_rate = firing_rate
            return firing_rate
        else:
            self.firing_rate = 0.0
            return 0.0
    
    def update_theta_phase(self, velocity: np.ndarray, dt: float = 0.1) -> None:
        """Update theta phase based on movement (phase precession)"""
        speed = np.linalg.norm(velocity)
        phase_velocity = 2 * np.pi * 8.0  # 8 Hz theta rhythm
        self.theta_phase += phase_velocity * dt + 0.1 * speed * dt
        self.theta_phase = self.theta_phase % (2 * np.pi)

# ---------------------- Time Cells (Temporal Processing) ----------------------

class TimeCell:
    """Individual time cell tracking temporal intervals"""
    
    def __init__(self, preferred_interval: float, width: float = 1.0):
        self.preferred_interval = preferred_interval  # Preferred time interval (seconds)
        self.width = width  # Temporal receptive field width
        self.firing_rate = 0.0
        self.last_event_time = 0.0
        self.max_firing_rate = 15.0  # Hz
        
    def compute_firing_rate(self, current_time: float, last_event_time: float) -> float:
        """Compute firing rate based on elapsed time since event"""
        elapsed_time = current_time - last_event_time
        
        if abs(elapsed_time - self.preferred_interval) <= self.width:
            # Gaussian temporal field
            firing_rate = self.max_firing_rate * np.exp(
                -((elapsed_time - self.preferred_interval)**2) / (2 * (self.width/3)**2)
            )
            self.firing_rate = firing_rate
            return firing_rate
        else:
            self.firing_rate = 0.0
            return 0.0

# ---------------------- Grid Cells (Spatial Navigation) ----------------------

class GridCell:
    """Grid cell providing hexagonal spatial coding"""
    
    def __init__(self, spacing: float = 1.0, orientation: float = 0.0, phase: np.ndarray = None):
        self.spacing = spacing  # Grid spacing
        self.orientation = orientation  # Grid orientation (radians)
        self.phase = phase if phase is not None else np.zeros(2)  # Spatial phase offset
        self.firing_rate = 0.0
        self.max_firing_rate = 25.0  # Hz
        
    def compute_firing_rate(self, location: np.ndarray) -> float:
        """Compute firing rate based on hexagonal grid pattern"""
        # Rotate location by grid orientation
        cos_o, sin_o = np.cos(self.orientation), np.sin(self.orientation)
        rotated_loc = np.array([
            cos_o * location[0] - sin_o * location[1],
            sin_o * location[0] + cos_o * location[1]
        ])
        
        # Apply phase offset
        shifted_loc = rotated_loc - self.phase
        
        # Hexagonal grid computation (simplified)
        # Using three cosine gratings at 60-degree angles
        k = 4 * np.pi / (self.spacing * np.sqrt(3))
        
        u1 = k * shifted_loc[0]
        u2 = k * (-0.5 * shifted_loc[0] + 0.866 * shifted_loc[1])  # 60 degrees
        u3 = k * (-0.5 * shifted_loc[0] - 0.866 * shifted_loc[1])  # -60 degrees
        
        grid_value = (np.cos(u1) + np.cos(u2) + np.cos(u3)) / 3.0 + 0.5
        
        self.firing_rate = self.max_firing_rate * max(0, grid_value)
        return self.firing_rate

# ---------------------- Hippocampal Formation ----------------------

class HippocampalFormation:
    """Complete hippocampal system for spatio-temporal processing"""
    
    def __init__(self, spatial_dimensions: int = 2, n_place_cells: int = 100, 
                 n_time_cells: int = 50, n_grid_cells: int = 75):
        
        # Neural populations
        self.spatial_dimensions = spatial_dimensions
        self.place_cells: List[PlaceCell] = []
        self.time_cells: List[TimeCell] = []
        self.grid_cells: List[GridCell] = []
        
        # Memory systems
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.spatial_locations: Dict[str, SpatialLocation] = {}
        self.temporal_events: deque = deque(maxlen=1000)  # Recent events
        
        # Current state
        self.current_location = np.zeros(spatial_dimensions)
        self.current_velocity = np.zeros(spatial_dimensions)
        self.last_event_time = time.time()
        
        # Cognitive map
        self.cognitive_map: Dict[Tuple[str, str], float] = {}  # Location relationships
        self.temporal_map: Dict[Tuple[str, str], float] = {}   # Event relationships
        
        # Theta rhythm (7-12 Hz oscillation)
        self.theta_frequency = 8.0  # Hz
        self.theta_phase = 0.0
        
        self._initialize_neural_populations(n_place_cells, n_time_cells, n_grid_cells)
        
    def _initialize_neural_populations(self, n_place: int, n_time: int, n_grid: int) -> None:
        """Initialize neural cell populations"""
        
        # Place cells with distributed receptive fields
        for _ in range(n_place):
            center = np.random.uniform(-10, 10, self.spatial_dimensions)
            radius = np.random.uniform(0.5, 2.0)
            self.place_cells.append(PlaceCell(center, radius))
        
        # Time cells with different preferred intervals
        intervals = np.logspace(0, 3, n_time)  # 1 second to 1000 seconds
        for interval in intervals:
            width = interval * 0.3  # Width proportional to interval
            self.time_cells.append(TimeCell(interval, width))
        
        # Grid cells with different spacings and orientations
        spacings = np.logspace(0, 2, n_grid)  # Different spatial scales
        for spacing in spacings:
            orientation = np.random.uniform(0, np.pi/3)  # 0-60 degrees
            phase = np.random.uniform(-spacing/2, spacing/2, 2)
            self.grid_cells.append(GridCell(spacing, orientation, phase))
    
    def update_spatial_state(self, new_location: np.ndarray, dt: float = 0.1) -> None:
        """Update current spatial state and neural activity"""
        # Calculate velocity
        self.current_velocity = (new_location - self.current_location) / dt
        self.current_location = new_location.copy()
        
        # Update theta rhythm
        self.theta_phase += 2 * np.pi * self.theta_frequency * dt
        self.theta_phase = self.theta_phase % (2 * np.pi)
        
        # Update place cells
        for place_cell in self.place_cells:
            place_cell.compute_firing_rate(new_location)
            place_cell.update_theta_phase(self.current_velocity, dt)
        
        # Update grid cells
        for grid_cell in self.grid_cells:
            grid_cell.compute_firing_rate(new_location)
    
    def process_temporal_event(self, event_id: str, features: np.ndarray, 
                              context: Optional[Dict[str, Any]] = None) -> None:
        """Process a temporal event and update time cells"""
        current_time = time.time()
        
        # Create temporal event
        event = TemporalEvent(
            timestamp=current_time,
            event_id=event_id,
            features=features,
            context=context
        )
        self.temporal_events.append(event)
        
        # Update time cells based on elapsed time since last event
        for time_cell in self.time_cells:
            time_cell.compute_firing_rate(current_time, self.last_event_time)
        
        self.last_event_time = current_time
    
    def create_episodic_memory(self, memory_id: str, event_id: str, features: np.ndarray,
                              associated_experts: List[str] = None) -> None:
        """Create new episodic memory binding space and time"""
        
        # Current spatial location
        location_id = f"loc_{hash(str(self.current_location))%10000}"
        if location_id not in self.spatial_locations:
            self.spatial_locations[location_id] = SpatialLocation(
                coordinates=self.current_location.copy(),
                context_features=features.copy() if features is not None else None
            )
        
        # Process temporal event
        self.process_temporal_event(event_id, features)
        current_event = self.temporal_events[-1]
        
        # Create episodic memory
        episodic_memory = EpisodicMemory(
            memory_id=memory_id,
            spatial_location=self.spatial_locations[location_id],
            temporal_event=current_event,
            associated_experts=associated_experts or []
        )
        
        self.episodic_memories[memory_id] = episodic_memory
        
        # Update cognitive maps
        self._update_cognitive_maps(episodic_memory)
    
    def _update_cognitive_maps(self, memory: EpisodicMemory) -> None:
        """Update spatial and temporal cognitive maps"""
        
        # Update spatial relationships
        for other_memory in self.episodic_memories.values():
            if other_memory.memory_id != memory.memory_id:
                spatial_distance = memory.spatial_location.distance_to(other_memory.spatial_location)
                key = (memory.memory_id, other_memory.memory_id)
                self.cognitive_map[key] = spatial_distance
                
                # Update temporal relationships
                temporal_distance = memory.temporal_event.time_distance_to(other_memory.temporal_event)
                self.temporal_map[key] = temporal_distance
    
    def retrieve_similar_memories(self, query_features: np.ndarray, 
                                 location: Optional[np.ndarray] = None,
                                 k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve episodic memories similar to query"""
        
        if location is not None:
            self.update_spatial_state(location)
        
        similarities = []
        
        for memory_id, memory in self.episodic_memories.items():
            # Feature similarity
            feature_sim = float(np.dot(query_features, memory.temporal_event.features))
            feature_sim /= (np.linalg.norm(query_features) * np.linalg.norm(memory.temporal_event.features) + 1e-8)
            
            # Spatial similarity (closer = more similar)
            spatial_dist = memory.spatial_location.distance_to(
                SpatialLocation(coordinates=self.current_location)
            )
            spatial_sim = 1.0 / (1.0 + spatial_dist)
            
            # Temporal recency (recent = more similar)
            age = time.time() - memory.temporal_event.timestamp
            temporal_sim = np.exp(-age / 3600.0)  # Hour-based decay
            
            # Combined similarity
            combined_sim = (0.5 * feature_sim + 0.3 * spatial_sim + 0.2 * temporal_sim) * memory.strength
            similarities.append((memory_id, combined_sim))
        
        # Return top-k most similar memories
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_spatial_context(self) -> Dict[str, Any]:
        """Get current spatial context representation"""
        
        # Place cell population vector
        place_activity = np.array([cell.firing_rate for cell in self.place_cells])
        
        # Grid cell population vector
        grid_activity = np.array([cell.firing_rate for cell in self.grid_cells])
        
        return {
            "current_location": self.current_location.copy(),
            "current_velocity": self.current_velocity.copy(),
            "place_cells": place_activity,
            "grid_cells": grid_activity,
            "theta_phase": self.theta_phase,
            "n_memories": len(self.episodic_memories)
        }
    
    def get_temporal_context(self) -> Dict[str, Any]:
        """Get current temporal context representation"""
        
        # Time cell population vector  
        time_activity = np.array([cell.firing_rate for cell in self.time_cells])
        
        # Recent events
        recent_events = list(self.temporal_events)[-10:]  # Last 10 events
        
        return {
            "time_cells": time_activity,
            "last_event_time": self.last_event_time,
            "recent_events": [e.event_id for e in recent_events],
            "temporal_sequence_length": len(self.temporal_events)
        }
    
    def decay_memories(self, decay_rate: float = 0.01) -> None:
        """Apply natural memory decay"""
        for memory in self.episodic_memories.values():
            memory.decay_strength(decay_rate)
        
        # Remove very weak memories
        to_remove = [mid for mid, mem in self.episodic_memories.items() if mem.strength < 0.01]
        for mid in to_remove:
            del self.episodic_memories[mid]

print("🧠 HIPPOCAMPAL FORMATION LOADED!")
print("=" * 40)
print("Components:")
print("• Place Cells: Spatial location encoding")
print("• Time Cells: Temporal interval coding") 
print("• Grid Cells: Spatial navigation system")
print("• Episodic Memory: Space-time binding")
print("• Cognitive Maps: Spatial & temporal relationships")
print("• Theta Rhythm: 8Hz oscillatory coordination")
print("• Memory Retrieval: Context-based recall")
