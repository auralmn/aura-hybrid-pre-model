import unittest
import numpy as np
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
from collections import deque
from scipy.signal import hilbert
from scipy.linalg import expm

# Import hippocampal formation from production module
from src.core.hippocampal import (
    SpatialLocation,
    TemporalEvent,
    EpisodicMemory,
    TemporalMemoryInterpolator,
    PlaceCell,
    TimeCell,
    GridCell,
    HippocampalFormation
)


class TestHippocampalFormation(unittest.TestCase):
    
    def test_place_cell_firing(self):
        cell = PlaceCell(center=np.array([0.0, 0.0]), radius=1.0)
        
        # At center, should have max firing
        rate_center = cell.compute_firing_rate(np.array([0.0, 0.0]))
        self.assertAlmostEqual(rate_center, 20.0, places=1)
        
        # Near center, should have high firing
        rate_near = cell.compute_firing_rate(np.array([0.3, 0.3]))
        self.assertGreater(rate_near, 5.0)
        self.assertLess(rate_near, 20.0)
        
        # Far from center, should have no firing
        rate_far = cell.compute_firing_rate(np.array([5.0, 5.0]))
        self.assertEqual(rate_far, 0.0)
    
    def test_time_cell_firing(self):
        cell = TimeCell(preferred_interval=2.0, width=0.5)
        
        start_time = time.time()
        
        # At preferred interval, should have max firing
        rate_preferred = cell.compute_firing_rate(start_time + 2.0, start_time)
        self.assertGreater(rate_preferred, 10.0)
        
        # Far from preferred interval, should have no firing
        rate_far = cell.compute_firing_rate(start_time + 10.0, start_time)
        self.assertEqual(rate_far, 0.0)
    
    def test_grid_cell_hexagonal_pattern(self):
        cell = GridCell(spacing=1.0, orientation=0.0)
        
        # Grid cells should show periodic firing
        locations = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.5, 0.866]),
        ]
        
        rates = [cell.compute_firing_rate(loc) for loc in locations]
        
        # Should have some variation in firing rates
        self.assertGreater(max(rates), 0.0)
        self.assertGreater(np.std(rates), 0.0)
    
    def test_hippocampal_initialization(self):
        hippo = HippocampalFormation(
            spatial_dimensions=2,
            n_place_cells=50,
            n_time_cells=25,
            n_grid_cells=30
        )
        
        self.assertEqual(len(hippo.place_cells), 50)
        self.assertEqual(len(hippo.time_cells), 25)
        self.assertEqual(len(hippo.grid_cells), 30)
        self.assertEqual(hippo.spatial_dimensions, 2)
    
    def test_spatial_state_update(self):
        hippo = HippocampalFormation(n_place_cells=100, n_time_cells=5, n_grid_cells=10)
        
        # Manually add a place cell at the test location to guarantee activity
        test_location = np.array([1.0, 2.0])
        hippo.place_cells.append(PlaceCell(center=test_location.copy(), radius=2.0))
        
        hippo.update_spatial_state(test_location, dt=0.1)
        
        np.testing.assert_array_almost_equal(hippo.current_location, test_location)
        self.assertGreater(np.linalg.norm(hippo.current_velocity), 0.0)
        
        # At least our manually added place cell should be active
        active_place_cells = sum(1 for cell in hippo.place_cells if cell.firing_rate > 0.0)
        self.assertGreater(active_place_cells, 0)
    
    def test_episodic_memory_creation(self):
        hippo = HippocampalFormation(n_place_cells=10, n_time_cells=5, n_grid_cells=10)
        
        hippo.update_spatial_state(np.array([1.0, 1.0]))
        features = np.random.randn(64)
        
        hippo.create_episodic_memory(
            memory_id="mem_1",
            event_id="event_1",
            features=features,
            associated_experts=["expert_a", "expert_b"]
        )
        
        self.assertIn("mem_1", hippo.episodic_memories)
        memory = hippo.episodic_memories["mem_1"]
        self.assertEqual(memory.associated_experts, ["expert_a", "expert_b"])
        self.assertAlmostEqual(memory.strength, 1.0)
    
    def test_memory_retrieval(self):
        hippo = HippocampalFormation(n_place_cells=10, n_time_cells=5, n_grid_cells=10)
        
        # Create multiple memories at different locations
        for i in range(5):
            location = np.array([float(i), float(i)])
            hippo.update_spatial_state(location)
            features = np.random.randn(64)
            features[0] = float(i)
            
            hippo.create_episodic_memory(
                memory_id=f"mem_{i}",
                event_id=f"event_{i}",
                features=features
            )
            time.sleep(0.01)
        
        # Query for similar memories
        query_features = np.random.randn(64)
        query_features[0] = 2.0
        
        similar = hippo.retrieve_similar_memories(query_features, k=3)
        
        self.assertEqual(len(similar), 3)
        self.assertTrue(all(isinstance(sim, tuple) for sim in similar))
        self.assertTrue(all(len(sim) == 2 for sim in similar))
    
    def test_cognitive_map_creation(self):
        hippo = HippocampalFormation(n_place_cells=10, n_time_cells=5, n_grid_cells=10)
        
        # Create first memory
        hippo.update_spatial_state(np.array([0.0, 0.0]))
        hippo.create_episodic_memory("mem_1", "event_1", np.random.randn(64))
        
        # Create second memory after a delay
        
        initial_strength = hippo.episodic_memories["mem_1"].strength
        self.assertAlmostEqual(initial_strength, 1.0)
        
        # Manually set timestamp to past for decay testing
        hippo.episodic_memories["mem_1"].temporal_event.timestamp = time.time() - 7200  # 2 hours ago
        
        # Decay with realistic rate
        hippo.decay_memories(decay_rate=0.1)
        
        decayed_strength = hippo.episodic_memories["mem_1"].strength
        self.assertLess(decayed_strength, initial_strength)
    
    def test_spatial_and_temporal_context(self):
        hippo = HippocampalFormation(n_place_cells=10, n_time_cells=5, n_grid_cells=10)
        
        hippo.update_spatial_state(np.array([1.0, 2.0]))
        hippo.process_temporal_event("event_1", np.random.randn(64))
        
        spatial_ctx = hippo.get_spatial_context()
        temporal_ctx = hippo.get_temporal_context()
        
        self.assertIn("place_cells", spatial_ctx)
        self.assertIn("grid_cells", spatial_ctx)
        self.assertIn("theta_phase", spatial_ctx)
        
        self.assertIn("time_cells", temporal_ctx)
        self.assertIn("recent_events", temporal_ctx)
        
        self.assertEqual(len(spatial_ctx["place_cells"]), 10)
        self.assertEqual(len(temporal_ctx["time_cells"]), 5)
    
    def test_spatial_memory_interpolation(self):
        hippo = HippocampalFormation(n_place_cells=10, n_time_cells=5, n_grid_cells=10)
        
        # Create memories at corners of a square
        corners = [
            (np.array([0.0, 0.0]), "corner_00"),
            (np.array([10.0, 0.0]), "corner_10"),
            (np.array([0.0, 10.0]), "corner_01"),
            (np.array([10.0, 10.0]), "corner_11"),
        ]
        
        for location, mem_id in corners:
            hippo.update_spatial_state(location)
            features = np.random.randn(64)
            features[0] = location[0]
            features[1] = location[1]
            hippo.create_episodic_memory(mem_id, f"event_{mem_id}", features)
            time.sleep(0.01)
        
        # Query at center point (5.0, 5.0)
        center = np.array([5.0, 5.0])
        query_features = np.random.randn(64)
        
        similar = hippo.retrieve_similar_memories(query_features, location=center, k=4)
        
        # Should retrieve all 4 corners with relatively equal weights
        self.assertEqual(len(similar), 4)
        
        # All corners should be in the retrieved set
        retrieved_ids = [mem_id for mem_id, _ in similar]
        for _, mem_id in corners:
            self.assertIn(mem_id, retrieved_ids)
    
    def test_temporal_memory_interpolation(self):
        hippo = HippocampalFormation(n_place_cells=10, n_time_cells=5, n_grid_cells=10)
        
        # Create memories at different times
        base_time = time.time()
        time_points = [0, 1, 2, 3]
        
        for i, t_offset in enumerate(time_points):
            features = np.random.randn(64)
            features[0] = float(i)
            hippo.create_episodic_memory(f"mem_t{i}", f"event_t{i}", features)
            
            # Manually set timestamp for testing
            hippo.episodic_memories[f"mem_t{i}"].temporal_event.timestamp = base_time + t_offset
            time.sleep(0.01)
        
        # Query shortly after - should retrieve recent memories with higher weight
        query_features = np.random.randn(64)
        similar = hippo.retrieve_similar_memories(query_features, k=3)
        
        # Should retrieve 3 memories
        self.assertEqual(len(similar), 3)
        
        # Most recent memory should have highest combined similarity (temporal recency)
        retrieved_scores = dict(similar)
        self.assertIn("mem_t3", retrieved_scores)
    
    def test_temporal_interpolator_linear(self):
        interpolator = TemporalMemoryInterpolator()
        
        M0 = np.array([1.0, 2.0, 3.0, 4.0])
        M1 = np.array([5.0, 6.0, 7.0, 8.0])
        
        # t=0 should return M0
        result_0 = interpolator.interpolate(M0, M1, t=0.0, mode='linear')
        np.testing.assert_array_almost_equal(result_0, M0)
        
        # t=1 should return M1
        result_1 = interpolator.interpolate(M0, M1, t=1.0, mode='linear')
        np.testing.assert_array_almost_equal(result_1, M1)
        
        # t=0.5 should return midpoint
        result_half = interpolator.interpolate(M0, M1, t=0.5, mode='linear')
        expected_half = (M0 + M1) / 2.0
        np.testing.assert_array_almost_equal(result_half, expected_half)
    
    def test_temporal_interpolator_fourier(self):
        interpolator = TemporalMemoryInterpolator()
        
        # Create signal with known frequency content
        t_points = np.linspace(0, 2*np.pi, 64)
        M0 = np.sin(t_points)
        M1 = np.sin(2 * t_points)
        
        # Interpolate in frequency domain
        result = interpolator.interpolate(M0, M1, t=0.5, mode='fourier')
        
        # Result should be real-valued
        self.assertTrue(np.all(np.isreal(result)))
        
        # Result should be between M0 and M1 in amplitude
        self.assertGreater(np.max(np.abs(result)), np.max(np.abs(M0)) * 0.8)
        self.assertLess(np.max(np.abs(result)), np.max(np.abs(M1)) * 1.2)
    
    def test_temporal_interpolator_hilbert(self):
        interpolator = TemporalMemoryInterpolator()
        
        # Create signals
        t_points = np.linspace(0, 2*np.pi, 64)
        M0 = np.sin(t_points)
        M1 = np.cos(t_points)
        
        # Hilbert interpolation preserves analytic signal
        result = interpolator.interpolate(M0, M1, t=0.5, mode='hilbert')
        
        # Result should be real-valued
        self.assertTrue(np.all(np.isreal(result)))
        
        # Result shape should match input
        self.assertEqual(result.shape, M0.shape)
    
    def test_temporal_interpolator_hamiltonian(self):
        interpolator = TemporalMemoryInterpolator()
        
        # Use smaller arrays for computational efficiency
        M0 = np.array([1.0, 0.0, 0.0, 0.0])
        M1 = np.array([0.0, 1.0, 0.0, 0.0])
        
        # Hamiltonian interpolation (quantum-inspired)
        result = interpolator.interpolate(M0, M1, t=0.5, mode='hamiltonian')
        
        # Result should be real-valued
        self.assertTrue(np.all(np.isreal(result)))
        
        # Result should preserve norm approximately
        norm_result = np.linalg.norm(result)
        self.assertGreater(norm_result, 0.5)
        self.assertLess(norm_result, 1.5)
    
    def test_temporal_interpolator_clipping(self):
        interpolator = TemporalMemoryInterpolator()
        
        M0 = np.array([1.0, 2.0, 3.0])
        M1 = np.array([4.0, 5.0, 6.0])
        
        # Test t values outside [0, 1] are clipped
        result_negative = interpolator.interpolate(M0, M1, t=-0.5, mode='linear')
        np.testing.assert_array_almost_equal(result_negative, M0)
        
        result_over = interpolator.interpolate(M0, M1, t=1.5, mode='linear')
        np.testing.assert_array_almost_equal(result_over, M1)
    
    def test_temporal_interpolator_with_memory_features(self):
        hippo = HippocampalFormation(n_place_cells=10, n_time_cells=5, n_grid_cells=10)
        interpolator = TemporalMemoryInterpolator()
        
        # Create two memories with different feature vectors
        feat0 = np.random.randn(64)
        feat1 = np.random.randn(64)
        
        hippo.create_episodic_memory("mem_0", "event_0", feat0)
        time.sleep(0.01)
        hippo.create_episodic_memory("mem_1", "event_1", feat1)
        
        # Interpolate between memory features
        interp_features = interpolator.interpolate(feat0, feat1, t=0.3, mode='hilbert')
        
        # Interpolated features should have correct shape
        self.assertEqual(interp_features.shape, feat0.shape)
        
        # Use interpolated features for retrieval
        similar = hippo.retrieve_similar_memories(interp_features, k=2)
        self.assertEqual(len(similar), 2)


if __name__ == '__main__':
    print("\nTesting Hippocampal Formation Module")
    print("=" * 60)
    unittest.main(verbosity=2)
