import unittest
import torch
import numpy as np
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.hippocampal import HippocampalFormation

class TestHippocampalFormation(unittest.TestCase):
    
    def setUp(self):
        # Use CPU for testing to avoid CI/CD issues if no GPU present
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # FIXED: feature_dim=64 matches the random vectors used in tests below
        self.hippo = HippocampalFormation(
            spatial_dimensions=2,
            n_place_cells=100,
            n_time_cells=20,
            n_grid_cells=50,
            feature_dim=64,
            device=self.device
        )

    def test_initialization(self):
        self.assertEqual(self.hippo.place_centers.shape[0], 100)
        self.assertEqual(self.hippo.grid_phases.shape[0], 50)
        self.assertEqual(self.hippo.time_intervals.shape[0], 20)
        
    def test_spatial_update(self):
        new_loc = torch.tensor([1.0, 2.0], device=self.device)
        
        # Force a place cell to be at this location to guarantee firing
        self.hippo.place_centers[0] = new_loc
        self.hippo.place_radii[0] = 2.0
        
        self.hippo.update_spatial_state(new_loc)
        
        self.assertTrue(torch.allclose(self.hippo.current_location, new_loc))
        
        ctx = self.hippo.get_spatial_context()
        place_activity = ctx['place_cells']
        
        self.assertEqual(place_activity.shape[0], 100)
        # Check cell 0 is active
        self.assertGreater(place_activity[0].item(), 5.0)

    def test_temporal_update(self):
        # 64-dim feature vector matches setUp
        self.hippo.create_episodic_memory("mem_0", "event_0", torch.randn(64, device=self.device))
        
        time.sleep(0.01)
        ctx = self.hippo.get_temporal_context()
        time_activity = ctx['time_cells']
        
        self.assertEqual(time_activity.shape[0], 20)

    def test_memory_creation_and_retrieval(self):
        features = torch.randn(5, 64, device=self.device)
        locations = torch.randn(5, 2, device=self.device) * 5.0
        
        for i in range(5):
            self.hippo.update_spatial_state(locations[i])
            self.hippo.create_episodic_memory(
                memory_id=f"mem_{i}",
                event_id=f"evt_{i}",
                features=features[i]
            )
            
        self.assertEqual(self.hippo.memory_count, 5)
        
        query = features[0]
        results = self.hippo.retrieve_similar_memories(query, k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "mem_0")

    def test_memory_decay(self):
        self.hippo.create_episodic_memory("mem_decay", "evt_decay", torch.randn(64, device=self.device))
        
        idx = self.hippo.id_to_idx["mem_decay"]
        initial_strength = self.hippo.memory_metadata[idx, 0].item()
        
        self.hippo.decay_memories(decay_rate=0.1)
        
        new_strength = self.hippo.memory_metadata[idx, 0].item()
        self.assertLess(new_strength, initial_strength)

if __name__ == '__main__':
    unittest.main()