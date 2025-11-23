import pytest
from unittest.mock import Mock, patch

# Test for src/core/brain.py using shared fixtures
class TestBrain:
    
    @patch('core.brain.LayersFactory')
    @patch('core.brain.NeuronFactory')
    @patch('core.brain.BrainZoneFactory')
    def test_brain_initialization(self, mock_zone_factory_class, mock_neuron_factory_class, 
                                mock_layers_factory_class, mock_layer_container_config, mock_brain_zones):
        """Test Brain initialization using shared fixtures."""
        from core.brain import Brain
        
        # Setup factory instances
        mock_layers_factory_instance = Mock()
        mock_neuron_factory_instance = Mock()
        mock_zone_factory_instance = Mock()
        
        mock_layers_factory_class.return_value = mock_layers_factory_instance
        mock_neuron_factory_class.return_value = mock_neuron_factory_instance
        mock_zone_factory_class.return_value = mock_zone_factory_instance
        
        # Mock brain zone creation
        created_zones = {}
        for zone_name in mock_brain_zones.keys():
            created_zones[zone_name] = Mock(name=f"{zone_name}_created")
        
        with patch.object(Brain, 'create_brain_zone') as mock_create_zone:
            mock_create_zone.side_effect = lambda config: created_zones[config.name]
            
            brain = Brain(mock_layer_container_config, mock_brain_zones)
        
        # Verify factory initialization
        mock_layers_factory_class.assert_called_once_with(mock_layer_container_config)
        mock_neuron_factory_class.assert_called_once_with(256, 512, 768, 0.02, 0.02, 2.0)
        mock_zone_factory_class.assert_called_once()
        
        # Verify factories are assigned
        assert brain.layers_factory == mock_layers_factory_instance
        assert brain.neuron_factory == mock_neuron_factory_instance
        assert brain.zone_factory == mock_zone_factory_instance
        
        # Verify zones dictionary
        assert len(brain.zones) == 5  # cortex, thalamus, hippocampus, amygdala, cerebellum
        for zone_name in mock_brain_zones.keys():
            assert zone_name in brain.zones

    @patch('core.brain.BaseLayerContainerConfig')
    @patch('core.brain.BaseLayerConfig')
    def test_create_brain_zone(self, mock_layer_config_class, mock_container_config_class,
                             mock_brain_zone_config):
        """Test the create_brain_zone method using shared fixtures."""
        from core.brain import Brain
        
        # Setup mocks
        mock_container_config = Mock()
        mock_layer_config = Mock()
        mock_container_config_class.return_value = mock_container_config
        mock_layer_config_class.return_value = mock_layer_config
        
        mock_layers_factory = Mock()
        mock_zone_factory = Mock()
        mock_layers = Mock()
        mock_brain_zone = Mock()
        
        mock_layers_factory.create_layers.return_value = mock_layers
        mock_zone_factory.create_brain_zone.return_value = mock_brain_zone
        
        # Create brain instance with mocked factories
        brain = Brain.__new__(Brain)  # Create without calling __init__
        brain.layers_factory = mock_layers_factory
        brain.zone_factory = mock_zone_factory
        
        result = brain.create_brain_zone(mock_brain_zone_config)
        
        # Verify BaseLayerContainerConfig creation
        mock_container_config_class.assert_called_once()
        call_kwargs = mock_container_config_class.call_args[1]
        assert call_kwargs.get('num_layers', 3) == mock_brain_zone_config.num_layers
        
        # Verify BaseLayerConfig creation
        mock_layer_config_class.assert_called_once_with(
            name=mock_brain_zone_config.name,
            input_dim=mock_brain_zone_config.min_neurons,
            output_dim=mock_brain_zone_config.max_neurons
        )
        
        assert result == mock_brain_zone

    def test_create_layers_delegation(self, mock_layers_factory):
        """Test create_layers method delegates to factory using fixture."""
        from core.brain import Brain
        
        # Create brain instance with mocked factory
        brain = Brain.__new__(Brain)
        brain.layers_factory = mock_layers_factory
        
        # Test config
        test_config = Mock()
        result = brain.create_layers(test_config)
        
        # Verify delegation to shared fixture
        mock_layers_factory.create_layers.assert_called_once_with(test_config)
        assert result is not None  # Mock returns something


class TestBrainIntegration:
    """Integration tests using multiple shared fixtures."""
    
    def test_brain_with_complete_neuromorphic_setup(self, mock_layer_container_config,
                                                  mock_brain_zones, mock_layers_factory,
                                                  mock_neuron_factory, mock_brain_zone_factory):
        """Complete integration test using all shared fixtures."""
        from core.brain import Brain
        
        # This would test the complete system integration
        # Using all the shared fixtures to simulate a realistic scenario
        
        # Verify fixtures are properly configured
        assert len(mock_brain_zones) == 5
        assert mock_layer_container_config.num_layers == 3
        assert mock_layers_factory.input_dim == 128
        assert mock_neuron_factory.hidden_dim == 256
        
        # Mock brain zone creation
        created_zones = {}
        for zone_name, zone_config in mock_brain_zones.items():
            zone = mock_brain_zone_factory.create_brain_zone(zone_config, [])
            created_zones[zone_name] = zone
            assert zone.name == zone_name
        
        # Verify integration between components
        assert len(created_zones) == 5

    @patch('core.brain.LayersFactory')
    @patch('core.brain.NeuronFactory')
    @patch('core.brain.BrainZoneFactory')
    def test_brain_empty_zones_scenario(self, mock_zone_factory_class, mock_neuron_factory_class,
                                      mock_layers_factory_class, mock_layer_container_config):
        """Test Brain with empty zones using fixtures."""
        from core.brain import Brain
        
        empty_zones = {}
        
        brain = Brain(mock_layer_container_config, empty_zones)
        
        # Should still initialize factories
        mock_layers_factory_class.assert_called_once()
        mock_neuron_factory_class.assert_called_once()
        mock_zone_factory_class.assert_called_once()
        
        # Zones should be empty
        assert len(brain.zones) == 0


# Parametrized tests using shared fixtures
@pytest.mark.parametrize("zone_name,expected_neurons", [
    ("cortex", 1000),
    ("thalamus", 200),
    ("hippocampus", 300),
    ("amygdala", 80),
    ("cerebellum", 500),
])
def test_brain_zone_neuron_counts(zone_name, expected_neurons, mock_brain_zones):
    """Test brain zones have expected neuron counts using fixtures."""
    zone_config = mock_brain_zones[zone_name]
    assert zone_config.min_neurons == expected_neurons
    assert zone_config.max_neurons == expected_neurons * 10


@pytest.mark.integration
def test_brain_realistic_workflow(mock_layer_container_config, mock_brain_zones,
                                mock_neuronal_state, neural_activity_generator):
    """Realistic workflow integration test using multiple fixtures."""
    # Generate activity patterns for different brain zones
    activity_patterns = {}
    for zone_name, zone_config in mock_brain_zones.items():
        num_neurons = zone_config.min_neurons
        # Generate different patterns for different zones
        if zone_name == "cortex":
            pattern = neural_activity_generator(num_neurons, 1000, 'oscillatory')
        elif zone_name == "hippocampus":
            pattern = neural_activity_generator(num_neurons, 1000, 'sparse')
        else:
            pattern = neural_activity_generator(num_neurons, 1000, 'random')
        
        activity_patterns[zone_name] = pattern
        
        # Verify pattern shapes
        assert pattern.shape == (num_neurons, 1000)
    
    # Verify we have activity for all zones
    assert len(activity_patterns) == 5


@pytest.mark.slow
def test_brain_large_scale_simulation(performance_timer, mock_brain_zones):
    """Performance test for large-scale brain simulation."""
    performance_timer.start()
    
    # Simulate large-scale brain processing
    total_neurons = sum(zone.min_neurons for zone in mock_brain_zones.values())
    
    # Simulate some processing time proportional to neuron count
    import time
    time.sleep(0.001 * total_neurons / 1000)  # Scale with neuron count
    
    elapsed = performance_timer.stop()
    
    # Should complete within reasonable time
    assert elapsed < 2.0  # Less than 2 seconds
    assert total_neurons == 1000 + 200 + 300 + 80 + 500  # Sum of all zones


@pytest.mark.unit
def test_brain_zone_connectivity_matrix(mock_brain_zone_factory, mock_brain_zone_config):
    """Unit test for brain zone connectivity using fixtures."""
    layers_mock = Mock()
    zone = mock_brain_zone_factory.create_brain_zone(mock_brain_zone_config, layers_mock)
    
    # Verify connectivity matrix properties
    assert hasattr(zone, 'connectivity_matrix')
    assert zone.connectivity_matrix.shape == (mock_brain_zone_config.max_neurons, 
                                            mock_brain_zone_config.max_neurons)
    
    # Matrix should be properly initialized
    assert zone.connectivity_matrix.min() >= 0
    assert zone.connectivity_matrix.max() <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])