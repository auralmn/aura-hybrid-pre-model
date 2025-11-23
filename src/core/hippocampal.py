"""
Hippocampal Formation Module

Bio-inspired episodic memory system implementing place cells, grid cells, time cells,
and cognitive maps for spatial and temporal memory encoding.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
from collections import deque
from scipy.signal import hilbert
from scipy.linalg import expm


@dataclass
class SpatialLocation:
    """Spatial location with coordinates and context"""
    coordinates: np.ndarray
    landmarks: List[str] = field(default_factory=list)
    context_features: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    visits: int = 1
    
    def distance_to(self, other: 'SpatialLocation') -> float:
        """Euclidean distance to another location"""
        return float(np.linalg.norm(self.coordinates - other.coordinates))


@dataclass 
class TemporalEvent:
    """Temporal event with timestamp and features"""
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
        self.strength *= np.exp(-decay_rate * elapsed / 3600.0)


class TemporalMemoryInterpolator:
    """Interpolate between memory states using multiple modes"""
    
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon
    
    def interpolate(self, M0: np.ndarray, M1: np.ndarray, t: float,
                    mode: Literal['linear', 'fourier', 'hilbert', 'hamiltonian'] = 'hilbert'
                   ) -> np.ndarray:
        """
        Interpolate between two memory states.
        
        Args:
            M0: Initial memory state
            M1: Final memory state
            t: Interpolation parameter [0, 1]
            mode: Interpolation method
                - linear: Weighted average
                - fourier: Frequency domain interpolation
                - hilbert: Phase-preserving analytic signal blend
                - hamiltonian: Quantum-inspired matrix exponential
        
        Returns:
            Interpolated memory state
        """
        alpha = np.clip(t, 0.0, 1.0)
        
        if mode == 'linear':
            return (1.0 - alpha) * M0 + alpha * M1
        
        elif mode == 'fourier':
            F0 = np.fft.fft(M0)
            F1 = np.fft.fft(M1)
            F_interp = (1.0 - alpha) * F0 + alpha * F1
            return np.real(np.fft.ifft(F_interp))
        
        A0 = hilbert(M0, axis=0)
        A1 = hilbert(M1, axis=0)
        
        if mode == 'hilbert':
            A_interp = (1.0 - alpha) * A0 + alpha * A1
            return np.real(A_interp)
        
        elif mode == 'hamiltonian':
            A_diff = (A1 - A0).astype(np.complex128)
            H_num = np.outer(A_diff, A_diff.T.conj())
            H_den = np.linalg.norm(A_diff)**2 + self.epsilon
            H = H_num / H_den
            U = expm(-1j * H * alpha)
            A_interp = U @ A0
            return np.real(A_interp)
        
        else:
            raise ValueError(f"Unknown interpolation mode: {mode}")


class PlaceCell:
    """Place cell with spatial receptive field"""
    
    def __init__(self, center: np.ndarray, radius: float = 1.0):
        self.center = center
        self.radius = radius
        self.firing_rate = 0.0
        self.theta_phase = 0.0
        self.max_firing_rate = 20.0
        
    def compute_firing_rate(self, location: np.ndarray) -> float:
        """Compute firing rate based on distance from center"""
        distance = np.linalg.norm(location - self.center)
        if distance <= self.radius:
            firing_rate = self.max_firing_rate * np.exp(-(distance**2) / (2 * (self.radius/3)**2))
            self.firing_rate = firing_rate
            return firing_rate
        else:
            self.firing_rate = 0.0
            return 0.0
    
    def update_theta_phase(self, velocity: np.ndarray, dt: float = 0.1) -> None:
        """Update theta phase based on movement (phase precession)"""
        speed = np.linalg.norm(velocity)
        phase_velocity = 2 * np.pi * 8.0
        self.theta_phase += phase_velocity * dt + 0.1 * speed * dt
        self.theta_phase = self.theta_phase % (2 * np.pi)


class TimeCell:
    """Time cell tracking temporal intervals"""
    
    def __init__(self, preferred_interval: float, width: float = 1.0):
        self.preferred_interval = preferred_interval
        self.width = width
        self.firing_rate = 0.0
        self.last_event_time = 0.0
        self.max_firing_rate = 15.0
        
    def compute_firing_rate(self, current_time: float, last_event_time: float) -> float:
        """Compute firing rate based on elapsed time since event"""
        elapsed_time = current_time - last_event_time
        
        if abs(elapsed_time - self.preferred_interval) <= self.width:
            firing_rate = self.max_firing_rate * np.exp(
                -((elapsed_time - self.preferred_interval)**2) / (2 * (self.width/3)**2)
            )
            self.firing_rate = firing_rate
            return firing_rate
        else:
            self.firing_rate = 0.0
            return 0.0


class GridCell:
    """Grid cell providing hexagonal spatial coding"""
    
    def __init__(self, spacing: float = 1.0, orientation: float = 0.0, phase: np.ndarray = None):
        self.spacing = spacing
        self.orientation = orientation
        self.phase = phase if phase is not None else np.zeros(2)
        self.firing_rate = 0.0
        self.max_firing_rate = 25.0
        
    def compute_firing_rate(self, location: np.ndarray) -> float:
        """Compute firing rate based on hexagonal grid pattern"""
        cos_o, sin_o = np.cos(self.orientation), np.sin(self.orientation)
        rotated_loc = np.array([
            cos_o * location[0] - sin_o * location[1],
            sin_o * location[0] + cos_o * location[1]
        ])
        
        shifted_loc = rotated_loc - self.phase
        k = 4 * np.pi / (self.spacing * np.sqrt(3))
        
        u1 = k * shifted_loc[0]
        u2 = k * (-0.5 * shifted_loc[0] + 0.866 * shifted_loc[1])
        u3 = k * (-0.5 * shifted_loc[0] - 0.866 * shifted_loc[1])
        
        grid_value = (np.cos(u1) + np.cos(u2) + np.cos(u3)) / 3.0 + 0.5
        
        self.firing_rate = self.max_firing_rate * max(0, grid_value)
        return self.firing_rate


class HippocampalFormation:
    """Complete hippocampal system for spatio-temporal processing"""
    
    def __init__(self, spatial_dimensions: int = 2, n_place_cells: int = 100, 
                 n_time_cells: int = 50, n_grid_cells: int = 75):
        
        self.spatial_dimensions = spatial_dimensions
        self.place_cells: List[PlaceCell] = []
        self.time_cells: List[TimeCell] = []
        self.grid_cells: List[GridCell] = []
        
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.spatial_locations: Dict[str, SpatialLocation] = {}
        self.temporal_events: deque = deque(maxlen=1000)
        
        self.current_location = np.zeros(spatial_dimensions)
        self.current_velocity = np.zeros(spatial_dimensions)
        self.last_event_time = time.time()
        
        self.cognitive_map: Dict[Tuple[str, str], float] = {}
        self.temporal_map: Dict[Tuple[str, str], float] = {}
        
        self.theta_frequency = 8.0
        self.theta_phase = 0.0
        
        self._initialize_neural_populations(n_place_cells, n_time_cells, n_grid_cells)
        
    def _initialize_neural_populations(self, n_place: int, n_time: int, n_grid: int) -> None:
        """Initialize neural cell populations"""
        
        for _ in range(n_place):
            center = np.random.uniform(-10, 10, self.spatial_dimensions)
            radius = np.random.uniform(0.5, 2.0)
            self.place_cells.append(PlaceCell(center, radius))
        
        intervals = np.logspace(0, 3, n_time)
        for interval in intervals:
            width = interval * 0.3
            self.time_cells.append(TimeCell(interval, width))
        
        spacings = np.logspace(0, 2, n_grid)
        for spacing in spacings:
            orientation = np.random.uniform(0, np.pi/3)
            phase = np.random.uniform(-spacing/2, spacing/2, 2)
            self.grid_cells.append(GridCell(spacing, orientation, phase))
    
    def update_spatial_state(self, new_location: np.ndarray, dt: float = 0.1) -> None:
        """Update current spatial state and neural activity"""
        self.current_velocity = (new_location - self.current_location) / dt
        self.current_location = new_location.copy()
        
        self.theta_phase += 2 * np.pi * self.theta_frequency * dt
        self.theta_phase = self.theta_phase % (2 * np.pi)
        
        for place_cell in self.place_cells:
            place_cell.compute_firing_rate(new_location)
            place_cell.update_theta_phase(self.current_velocity, dt)
        
        for grid_cell in self.grid_cells:
            grid_cell.compute_firing_rate(new_location)
    
    def process_temporal_event(self, event_id: str, features: np.ndarray, 
                              context: Optional[Dict[str, Any]] = None) -> None:
        """Process a temporal event and update time cells"""
        current_time = time.time()
        
        event = TemporalEvent(
            timestamp=current_time,
            event_id=event_id,
            features=features,
            context=context
        )
        self.temporal_events.append(event)
        
        for time_cell in self.time_cells:
            time_cell.compute_firing_rate(current_time, self.last_event_time)
        
        self.last_event_time = current_time
    
    def create_episodic_memory(self, memory_id: str, event_id: str, features: np.ndarray,
                              associated_experts: List[str] = None) -> None:
        """Create new episodic memory binding space and time"""
        
        location_id = f"loc_{hash(str(self.current_location))%10000}"
        if location_id not in self.spatial_locations:
            self.spatial_locations[location_id] = SpatialLocation(
                coordinates=self.current_location.copy(),
                context_features=features.copy() if features is not None else None
            )
        
        self.process_temporal_event(event_id, features)
        current_event = self.temporal_events[-1]
        
        episodic_memory = EpisodicMemory(
            memory_id=memory_id,
            spatial_location=self.spatial_locations[location_id],
            temporal_event=current_event,
            associated_experts=associated_experts or []
        )
        
        self.episodic_memories[memory_id] = episodic_memory
        self._update_cognitive_maps(episodic_memory)
    
    def _update_cognitive_maps(self, memory: EpisodicMemory) -> None:
        """Update spatial and temporal cognitive maps.

        For each existing memory, we store the distance in both directions so the
        map is effectively bidirectional. This matches the expectation in the
        integration tests that a graph with *n* memories has *n*(n-1) edges.
        """
        for other_memory in self.episodic_memories.values():
            if other_memory.memory_id != memory.memory_id:
                spatial_distance = memory.spatial_location.distance_to(other_memory.spatial_location)
                key_fwd = (memory.memory_id, other_memory.memory_id)
                key_rev = (other_memory.memory_id, memory.memory_id)
                self.cognitive_map[key_fwd] = spatial_distance
                self.cognitive_map[key_rev] = spatial_distance
                temporal_distance = memory.temporal_event.time_distance_to(other_memory.temporal_event)
                self.temporal_map[key_fwd] = temporal_distance
                self.temporal_map[key_rev] = temporal_distance
                
            """Update spatial and temporal cognitive maps"""
            
            for other_memory in self.episodic_memories.values():
                if other_memory.memory_id != memory.memory_id:
                    spatial_distance = memory.spatial_location.distance_to(other_memory.spatial_location)
                    key = (memory.memory_id, other_memory.memory_id)
                    self.cognitive_map[key] = spatial_distance
                    
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
            feature_sim = float(np.dot(query_features, memory.temporal_event.features))
            feature_sim /= (np.linalg.norm(query_features) * np.linalg.norm(memory.temporal_event.features) + 1e-8)
            
            spatial_dist = memory.spatial_location.distance_to(
                SpatialLocation(coordinates=self.current_location)
            )
            spatial_sim = 1.0 / (1.0 + spatial_dist)
            
            age = time.time() - memory.temporal_event.timestamp
            temporal_sim = np.exp(-age / 3600.0)
            
            combined_sim = (0.5 * feature_sim + 0.3 * spatial_sim + 0.2 * temporal_sim) * memory.strength
            similarities.append((memory_id, combined_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_spatial_context(self) -> Dict[str, Any]:
        """Get current spatial context representation"""
        
        place_activity = np.array([cell.firing_rate for cell in self.place_cells])
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
        
        time_activity = np.array([cell.firing_rate for cell in self.time_cells])
        recent_events = list(self.temporal_events)[-10:]
        
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
        
        to_remove = [mid for mid, mem in self.episodic_memories.items() if mem.strength < 0.01]
        for mid in to_remove:
            del self.episodic_memories[mid]
