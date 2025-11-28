import pytest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.brain import EnhancedBrain
from src.base.snn_brain_zones import BrainZoneConfig, BrainZoneType, SpikingNeuronConfig

class TestBrainIntegration:
    
    def setup_method(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.d_model = 64
        
        # Explicit configs to ensure neuron groups are created
        self.zones_config = {
            'prefrontal': BrainZoneConfig(
                name='prefrontal',
                max_neurons=128,
                min_neurons=64,
                zone_type=BrainZoneType.PREFRONTAL_CORTEX,
                use_spiking=True,
                d_model=self.d_model,
                spiking_configs=[
                    SpikingNeuronConfig("pyr", "std", "glu", 80.0),
                    SpikingNeuronConfig("int", "std", "gaba", 20.0)
                ]
            ),
            'hippocampus': BrainZoneConfig(
                name='hippocampus',
                max_neurons=128,
                min_neurons=64,
                zone_type=BrainZoneType.HIPPOCAMPUS,
                use_spiking=True,
                d_model=self.d_model,
                spiking_configs=[
                    SpikingNeuronConfig("place", "std", "glu", 100.0)
                ]
            )
        }

    def test_brain_initialization_real(self):
        brain = EnhancedBrain(
            d_model=self.d_model, 
            zones_config=self.zones_config, 
            device=self.device
        )
        
        assert len(brain.zones) == 2
        assert 'prefrontal' in brain.zones
        
        # Robust class check
        zone = brain.zones['prefrontal']
        assert zone.__class__.__name__ == 'NeuromorphicBrainZone'
        
        # Verify neuron groups exist
        assert len(zone.neuron_groups) > 0

    def test_brain_forward_pass_flow(self):
        brain = EnhancedBrain(
            d_model=self.d_model, 
            zones_config=self.zones_config, 
            device=self.device
        )
        
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.d_model, device=self.device)
        
        output, info = brain.process_input(x, text_context="integration test")
        
        assert output.shape == (batch_size, seq_len, self.d_model)
        assert 'zone_activities' in info

    def test_brain_statistics_collection(self):
        brain = EnhancedBrain(
            d_model=self.d_model, 
            zones_config=self.zones_config, 
            device=self.device
        )
        
        x = torch.randn(1, 5, self.d_model, device=self.device) + 1.0 
        brain.process_input(x)
        
        brain.stats_collector.update_from_brain(brain)
        stats = brain.get_brain_statistics()
        
        assert stats.num_zones == 2
        assert stats.num_neurons == 256

    def test_brain_dynamic_zone_addition(self):
        self.zones_config['temporal'] = BrainZoneConfig(
            name='temporal',
            max_neurons=64,
            zone_type=BrainZoneType.TEMPORAL_CORTEX,
            d_model=self.d_model
        )
        
        brain = EnhancedBrain(
            d_model=self.d_model, 
            zones_config=self.zones_config, 
            device=self.device
        )
        
        assert 'temporal' in brain.zones
        assert len(brain.zones) == 3

if __name__ == '__main__':
    pytest.main([__file__, '-v'])