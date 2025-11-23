"""
Integration tests for Hippocampal Formation with Brain class.
"""

import unittest
import numpy as np
import sys
import os

# Ensure src is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core.hippocampal import HippocampalFormation
from core.brain import Brain
from cli.config import BrainConfig


class TestHippocampalBrainIntegration(unittest.TestCase):
    """Test how hippocampal formation integrates with Brain class"""

    def test_brain_with_hippocampus_initialization(self):
        config = BrainConfig()
        brain = Brain(config)
        brain.hippocampus = HippocampalFormation(
            n_place_cells=50,
            n_time_cells=20,
            n_grid_cells=30,
        )
        self.assertIsNotNone(brain.hippocampus)
        self.assertEqual(len(brain.hippocampus.place_cells), 50)
        self.assertEqual(len(brain.hippocampus.time_cells), 20)
        self.assertEqual(len(brain.hippocampus.grid_cells), 30)

    def test_brain_processes_input_with_memory(self):
        config = BrainConfig()
        brain = Brain(config)
        brain.hippocampus = HippocampalFormation(
            n_place_cells=30,
            n_time_cells=15,
            n_grid_cells=20,
        )
        input_data = np.random.randn(10)
        spatial_location = np.array([1.0, 2.0])
        brain.hippocampus.update_spatial_state(spatial_location)
        brain.hippocampus.create_episodic_memory(
            memory_id="proc_1",
            event_id="input_1",
            features=input_data,
            associated_experts=["visual"]
        )
        self.assertEqual(len(brain.hippocampus.episodic_memories), 1)
        self.assertIn("proc_1", brain.hippocampus.episodic_memories)

    def test_brain_retrieves_similar_memories(self):
        config = BrainConfig()
        brain = Brain(config)
        brain.hippocampus = HippocampalFormation(
            n_place_cells=30,
            n_time_cells=15,
            n_grid_cells=20,
        )
        for i in range(5):
            location = np.array([float(i), float(i)])
            brain.hippocampus.update_spatial_state(location)
            features = np.random.randn(64)
            features[0] = float(i)
            brain.hippocampus.create_episodic_memory(
                memory_id=f"mem_{i}",
                event_id=f"event_{i}",
                features=features
            )
        query_features = np.random.randn(64)
        query_features[0] = 2.0
        similar = brain.hippocampus.retrieve_similar_memories(query_features, k=3)
        self.assertEqual(len(similar), 3)
        self.assertTrue(all(isinstance(s, tuple) and len(s) == 2 for s in similar))

    def test_brain_spatial_context_tracking(self):
        config = BrainConfig()
        brain = Brain(config)
        brain.hippocampus = HippocampalFormation(
            n_place_cells=50,
            n_time_cells=20,
            n_grid_cells=30,
        )
        locations = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0])]
        for loc in locations:
            brain.hippocampus.update_spatial_state(loc)
            spatial_ctx = brain.hippocampus.get_spatial_context()
            self.assertIn('place_cells', spatial_ctx)
            self.assertIn('grid_cells', spatial_ctx)
            self.assertIn('current_location', spatial_ctx)
            np.testing.assert_array_almost_equal(spatial_ctx['current_location'], loc)

    def test_brain_temporal_context_tracking(self):
        config = BrainConfig()
        brain = Brain(config)
        brain.hippocampus = HippocampalFormation(
            n_place_cells=30,
            n_time_cells=20,
            n_grid_cells=25,
        )
        for i in range(3):
            features = np.random.randn(32)
            brain.hippocampus.process_temporal_event(f"event_{i}", features)
        temporal_ctx = brain.hippocampus.get_temporal_context()
        self.assertIn('time_cells', temporal_ctx)
        self.assertIn('recent_events', temporal_ctx)
        self.assertIn('temporal_sequence_length', temporal_ctx)
        self.assertEqual(temporal_ctx['temporal_sequence_length'], 3)

    def test_brain_cognitive_map_formation(self):
        config = BrainConfig()
        brain = Brain(config)
        brain.hippocampus = HippocampalFormation(
            n_place_cells=30,
            n_time_cells=15,
            n_grid_cells=20,
        )
        locations = [np.array([0.0, 0.0]), np.array([3.0, 4.0]), np.array([6.0, 8.0])]
        for i, loc in enumerate(locations):
            brain.hippocampus.update_spatial_state(loc)
            brain.hippocampus.create_episodic_memory(
                f"mem_{i}",
                f"event_{i}",
                np.random.randn(32)
            )
        self.assertGreater(len(brain.hippocampus.cognitive_map), 0)
        self.assertEqual(len(brain.hippocampus.cognitive_map), 6)

    def test_brain_memory_decay(self):
        config = BrainConfig()
        brain = Brain(config)
        brain.hippocampus = HippocampalFormation(
            n_place_cells=30,
            n_time_cells=15,
            n_grid_cells=20,
        )
        brain.hippocampus.create_episodic_memory("mem_1", "event_1", np.random.randn(32))
        initial_strength = brain.hippocampus.episodic_memories["mem_1"].strength
        self.assertAlmostEqual(initial_strength, 1.0)
        import time
        brain.hippocampus.episodic_memories["mem_1"].temporal_event.timestamp = time.time() - 3600
        brain.hippocampus.decay_memories(decay_rate=0.1)
        decayed_strength = brain.hippocampus.episodic_memories["mem_1"].strength
        self.assertLess(decayed_strength, initial_strength)

    def test_brain_without_hippocampus(self):
        config = BrainConfig()
        brain = Brain(config)
        self.assertIsInstance(brain, Brain)
        hippocampus = getattr(brain, 'hippocampus', None)
        self.assertIsNone(hippocampus)

    def test_brain_sequential_processing_with_memory(self):
        config = BrainConfig()
        brain = Brain(config)
        brain.hippocampus = HippocampalFormation(
            n_place_cells=40,
            n_time_cells=20,
            n_grid_cells=30,
        )
        sequence_length = 10
        for i in range(sequence_length):
            input_data = np.random.randn(16)
            location = np.array([float(i), np.sin(i)])
            brain.hippocampus.update_spatial_state(location)
            brain.hippocampus.create_episodic_memory(
                f"seq_{i}",
                f"step_{i}",
                input_data
            )
        self.assertEqual(len(brain.hippocampus.episodic_memories), sequence_length)
        self.assertGreater(len(brain.hippocampus.cognitive_map), 0)
        temporal_ctx = brain.hippocampus.get_temporal_context()
        self.assertEqual(temporal_ctx['temporal_sequence_length'], sequence_length)


if __name__ == '__main__':
    print("\nTesting Hippocampal-Brain Integration")
    print("=" * 60)
    unittest.main(verbosity=2)
