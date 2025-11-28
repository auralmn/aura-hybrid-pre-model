"""
Hippocampal Formation Module (GPU-Native / Vectorized)

Bio-inspired episodic memory system implementing place cells, grid cells, time cells,
and cognitive maps using efficient PyTorch tensor operations for L4/A100 acceleration.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
from collections import deque

# Keep data classes for metadata storage, but logic moves to tensors
@dataclass
class SpatialLocation:
    """Spatial location metadata"""
    coordinates: torch.Tensor
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class EpisodicMemory:
    """Episodic memory metadata (tensor indices managed by formation)"""
    memory_id: str
    feature_idx: int  # Index in the GPU memory bank
    timestamp: float
    strength: float = 1.0

class HippocampalFormation(nn.Module):
    """
    GPU-Accelerated Hippocampal System.
    
    Replaces list-based cells with batched tensors:
    - Place Cells: (N, D) centers
    - Grid Cells: (N, D) phases/spacings
    - Memory Bank: (M, Feature_Dim) pre-allocated buffer
    """
    
    def __init__(self, 
                 spatial_dimensions: int = 2, 
                 n_place_cells: int = 2000, 
                 n_time_cells: int = 100, 
                 n_grid_cells: int = 200,
                 max_memories: int = 100000,
                 feature_dim: int = 768,
                 device: str = 'cuda'):
        
        super().__init__()
        self.spatial_dims = spatial_dimensions
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # --- 1. Vectorized Place Cells ---
        # Randomly distributed centers: (N_place, D)
        self.register_buffer('place_centers', 
                           torch.rand(n_place_cells, spatial_dimensions, device=self.device) * 20 - 10)
        # Random radii: (N_place, 1)
        self.register_buffer('place_radii', 
                           torch.rand(n_place_cells, 1, device=self.device) * 1.5 + 0.5)
        self.place_max_rate = 20.0
        
        # --- 2. Vectorized Grid Cells ---
        # Spacings log-uniform: (N_grid, 1)
        spacings = torch.logspace(0, 2, n_grid_cells, base=2.0, device=self.device).unsqueeze(1)
        self.register_buffer('grid_spacings', spacings)
        
        # Orientations: (N_grid, 1)
        self.register_buffer('grid_orientations', 
                           torch.rand(n_grid_cells, 1, device=self.device) * (torch.pi / 3))
        
        # Phases: (N_grid, D)
        self.register_buffer('grid_phases', 
                           torch.rand(n_grid_cells, spatial_dimensions, device=self.device) * spacings)
        self.grid_max_rate = 25.0

        # --- 3. Vectorized Time Cells ---
        # Intervals log-space: (N_time, 1)
        intervals = torch.logspace(0, 3, n_time_cells, base=10.0, device=self.device).unsqueeze(1)
        self.register_buffer('time_intervals', intervals)
        self.register_buffer('time_widths', intervals * 0.3)
        
        # --- 4. GPU Memory Bank ---
        # Pre-allocate large buffer for episodic memories to avoid reallocation
        self.max_memories = max_memories
        self.memory_count = 0
        
        # Stores [features] for fast similarity search
        self.register_buffer('memory_features', 
                           torch.zeros(max_memories, feature_dim, device=self.device))
        
        # Stores [x, y, ... coords] for spatial lookup
        self.register_buffer('memory_locations',
                           torch.zeros(max_memories, spatial_dimensions, device=self.device))
                           
        # Stores [strength, timestamp, 0, 0]
        self.register_buffer('memory_metadata',
                           torch.zeros(max_memories, 4, device=self.device))
        
        # Python-side metadata mapping
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.id_to_idx: Dict[str, int] = {}
        
        # State
        self.current_location = torch.zeros(spatial_dimensions, device=self.device)
        self.last_event_time = time.time()
        
        # Pre-compute hex grid vectors for grid cells (constants)
        self.register_buffer('k_const', 4 * torch.pi / torch.sqrt(torch.tensor(3.0, device=self.device)))

    def update_spatial_state(self, new_location: torch.Tensor, dt: float = 0.1) -> None:
        """
        Update current spatial state.
        Args:
            new_location: Tensor (Batch, D) or (D,)
        """
        if isinstance(new_location, np.ndarray):
            new_location = torch.from_numpy(new_location).to(self.device, dtype=torch.float32)
        
        if new_location.dim() == 1:
            self.current_location = new_location
        else:
            self.current_location = new_location[0] # Take first of batch for global state

    def get_spatial_context(self) -> Dict[str, Any]:
        """
        Compute place and grid cell activities using vectorized GPU ops.
        """
        loc = self.current_location.unsqueeze(0) # (1, D)
        
        # --- Place Cells (Vectorized) ---
        # Dist sq: (1, D) - (N, D) -> (N, D) -> norm -> (N,)
        dists = torch.norm(loc - self.place_centers, dim=1, keepdim=True) # (N, 1)
        
        # Gaussian activation
        # Rate = Max * exp( - dist^2 / (2 * sigma^2) ) where sigma = radius/3
        sigmas = self.place_radii / 3.0
        place_rates = self.place_max_rate * torch.exp(-(dists**2) / (2 * sigmas**2))
        
        # Mask out-of-radius (optional, but Gaussian handles it mostly)
        place_rates = place_rates * (dists <= self.place_radii).float()
        
        # --- Grid Cells (Vectorized) ---
        # 3 waves plane calculation
        cos_o = torch.cos(self.grid_orientations)
        sin_o = torch.sin(self.grid_orientations)
        
        # Rotate location: x' = x cos - y sin, y' = x sin + y cos
        # This part assumes D=2 for rotation logic, simplifying for general case:
        x, y = loc[0, 0], loc[0, 1]
        rot_x = cos_o * x - sin_o * y
        rot_y = sin_o * x + cos_o * y
        rotated = torch.cat([rot_x, rot_y], dim=1) # (N, 2)
        
        shifted = rotated - self.grid_phases
        k = self.k_const / self.grid_spacings
        
        u1 = k * shifted[:, 0:1]
        u2 = k * (-0.5 * shifted[:, 0:1] + 0.866 * shifted[:, 1:2])
        u3 = k * (-0.5 * shifted[:, 0:1] - 0.866 * shifted[:, 1:2])
        
        grid_val = (torch.cos(u1) + torch.cos(u2) + torch.cos(u3)) / 3.0 + 0.5
        grid_rates = self.grid_max_rate * torch.relu(grid_val)
        
        return {
            "current_location": self.current_location,
            "place_cells": place_rates.flatten(), # Return tensor, not numpy
            "grid_cells": grid_rates.flatten(),
            "n_memories": self.memory_count
        }

    def get_temporal_context(self) -> Dict[str, Any]:
        """Vectorized time cell activity."""
        elapsed = time.time() - self.last_event_time
        
        # Gaussian temporal receptive fields
        # exp( - (t - preferred)^2 / (2 * width^2) )
        diff = elapsed - self.time_intervals
        time_rates = 15.0 * torch.exp(-(diff**2) / (2 * (self.time_widths/3)**2))
        
        return {
            "time_cells": time_rates.flatten(),
            "elapsed": elapsed
        }

    def create_episodic_memory(self, memory_id: str, event_id: str, features: torch.Tensor,
                              associated_experts: List[str] = None) -> None:
        """
        Store memory in the pre-allocated GPU bank.
        """
        if self.memory_count >= self.max_memories:
            # Simple FIFO overwrite if full (could be improved to LRU)
            idx = self.memory_count % self.max_memories 
        else:
            idx = self.memory_count
            self.memory_count += 1
            
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self.device, dtype=torch.float32)
            
        # Write to GPU tensors
        self.memory_features[idx] = features.detach()
        self.memory_locations[idx] = self.current_location
        
        # Metadata: [strength, timestamp, reserved, reserved]
        self.memory_metadata[idx] = torch.tensor([1.0, time.time(), 0.0, 0.0], device=self.device)
        
        # CPU-side tracking
        self.episodic_memories[memory_id] = EpisodicMemory(
            memory_id=memory_id,
            feature_idx=idx,
            timestamp=time.time()
        )
        self.id_to_idx[memory_id] = idx

    def retrieve_similar_memories(self, query_features: torch.Tensor, 
                                 location: Optional[torch.Tensor] = None,
                                 k: int = 5) -> List[Tuple[str, float]]:
        """
        Massively accelerated retrieval using matrix multiplication.
        """
        if self.memory_count == 0:
            return []
            
        if isinstance(query_features, np.ndarray):
            query_features = torch.from_numpy(query_features).to(self.device, dtype=torch.float32)
            
        # 1. Feature Similarity (Cosine)
        # Normalize query
        q_norm = torch.nn.functional.normalize(query_features.unsqueeze(0), dim=1)
        
        # Slice active memories
        active_feats = self.memory_features[:self.memory_count]
        m_norm = torch.nn.functional.normalize(active_feats, dim=1)
        
        # Dot product (Vectorized)
        sim_scores = torch.mm(q_norm, m_norm.t()).squeeze(0) # (M,)
        
        # 2. Spatial Similarity (if location provided)
        spatial_scores = torch.zeros_like(sim_scores)
        if location is not None:
            if isinstance(location, np.ndarray):
                location = torch.from_numpy(location).to(self.device, dtype=torch.float32)
            
            active_locs = self.memory_locations[:self.memory_count]
            dists = torch.norm(active_locs - location, dim=1)
            spatial_scores = 1.0 / (1.0 + dists)
            
        # 3. Temporal Similarity (Exponential decay)
        active_meta = self.memory_metadata[:self.memory_count]
        timestamps = active_meta[:, 1]
        strengths = active_meta[:, 0]
        
        ages = time.time() - timestamps
        temporal_scores = torch.exp(-ages / 3600.0)
        
        # 4. Combined Score
        # Weights: 0.5 Feature, 0.3 Spatial, 0.2 Temporal
        combined = (0.5 * sim_scores + 
                   0.3 * spatial_scores + 
                   0.2 * temporal_scores) * strengths
                   
        # 5. Top-K
        k = min(k, self.memory_count)
        top_scores, top_indices = torch.topk(combined, k)
        
        # Map back to IDs (this part is CPU but K is small, e.g. 5)
        results = []
        # Invert the mapping for lookup (could be optimized)
        idx_to_id = {v: k for k, v in self.id_to_idx.items()}
        
        for score, idx in zip(top_scores, top_indices):
            idx_val = idx.item()
            if idx_val in idx_to_id:
                results.append((idx_to_id[idx_val], score.item()))
                
        return results

    def decay_memories(self, decay_rate: float = 0.01) -> None:
        """
        Vectorized memory decay.
        """
        if self.memory_count == 0:
            return
            
        # Update strength in-place on GPU
        # strength *= exp(-decay * elapsed)
        # Note: simplistic decay applied every call step, or use timestamp
        # Ideally, strength = initial * exp(-rate * (now - creation))
        
        # Here we apply a multiplicative step decay for active memories
        self.memory_metadata[:self.memory_count, 0] *= (1.0 - decay_rate)
        
        # Pruning (Optional: zero out very weak memories to free slots?)
        # For fixed bank, we usually just overwrite via circular buffer (FIFO)
        # or implement a "free list" for strength < threshold.
        pass