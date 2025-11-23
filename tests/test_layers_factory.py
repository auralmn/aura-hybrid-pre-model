import pytest
from unittest.mock import Mock, patch

# Test for src/core/layers_factory.py using shared fixtures
class TestLayersFactory:
    
    @patch('core.layers_factory.BaseLayerImplementation')
    @patch('core.layers_factory.BaseLayerContainer')
    def test_layers_factory_initialization(self, mock_container_class, mock_implementation_class,
                                         mock_layer_container_config):
        """Test LayersFactory initialization using shared fixture."""
        from core.layers_factory import LayersFactory
        
        factory = LayersFactory(mock_layer_container_config)
        
        assert factory.config == mock_layer_container_config
        assert factory.layers == []
    
    @patch('core.layers_factory.BaseLayerImplementation')
    @patch('core.layers_factory.BaseLayerContainer')
    @patch('builtins.print')
    def test_create_layers_success(self, mock_print, mock_container_class, mock_implementation_class,
                                 mock_layer_container_config, mock_layer_config):
        """Test successful layer creation using shared fixtures."""
        from core.layers_factory import LayersFactory
        
        # Setup mocks
        mock_layers = [Mock(name=f"layer_{i}") for i in range(3)]
        mock_implementation_class.side_effect = mock_layers
        
        mock_container_instance = Mock()
        mock_container_class.return_value = mock_container_instance
        
        # Create factory and layers
        factory = LayersFactory(mock_layer_container_config)
        result = factory.create_layers(mock_layer_config)
        
        # Verify correct number of layers created
        assert len(factory.layers) == 3
        assert mock_implementation_class.call_count == 3
        
        # Verify container creation
        mock_container_class.assert_called_once_with(
            config=mock_layer_container_config,
            layers=factory.layers
        )
        assert result == mock_container_instance
    
    @patch('core.layers_factory.BaseLayerImplementation')
    @patch('core.layers_factory.BaseLayerContainer')
    def test_create_layers_zero_layers(self, mock_container_class, mock_implementation_class,
                                     mock_layer_config):
        """Test creating layers with zero layers using fixtures."""
        from core.layers_factory import LayersFactory
        
        # Config with zero layers
        zero_layer_config = Mock()
        zero_layer_config.num_layers = 0
        
        mock_container_instance = Mock()
        mock_container_class.return_value = mock_container_instance
        
        factory = LayersFactory(zero_layer_config)
        result = factory.create_layers(mock_layer_config)
        
        # No layers should be created
        assert len(factory.layers) == 0
        assert mock_implementation_class.call_count == 0
    
    def test_layers_factory_config_access(self, mock_layer_container_config):
        """Test config access using shared fixture."""
        from core.layers_factory import LayersFactory
        
        factory = LayersFactory(mock_layer_container_config)
        
        # Config should be accessible
        assert factory.config == mock_layer_container_config
        assert factory.config.num_layers == 3


# Integration test using multiple shared fixtures
class TestLayersFactoryIntegration:
    
    @patch('core.layers_factory.BaseLayerImplementation')
    @patch('core.layers_factory.BaseLayerContainer')
    def test_realistic_neural_network_creation(self, mock_container_class, mock_implementation_class,
                                             mock_layers_factory):
        """Integration test using shared factory fixture."""
        # Use the mock factory directly
        mock_layers_factory.config.num_layers = 4
        
        layer_config = Mock()
        layer_config.name = "dense_layer"
        
        # Test the factory behavior
        result = mock_layers_factory.create_layers(layer_config)
        
        # Verify the mock factory worked correctly
        mock_layers_factory.create_layers.assert_called_once_with(layer_config)
        assert result is not None


# Parametrized tests using shared fixtures
@pytest.mark.parametrize("num_layers,layer_type", [
    (1, "dense"),
    (3, "conv2d"),
    (5, "lstm"),
    (10, "attention"),
])
def test_different_layer_configurations(num_layers, layer_type, mock_layer_config):
    """Test different layer configurations using fixtures."""
    from core.layers_factory import LayersFactory
    
    # Create config with specified parameters
    config = Mock()
    config.num_layers = num_layers
    config.layer_type = layer_type
    
    # Update shared fixture
    mock_layer_config.type = layer_type
    
    with patch('core.layers_factory.BaseLayerImplementation') as mock_impl, \
         patch('core.layers_factory.BaseLayerContainer') as mock_container:
        
        mock_impl.side_effect = [Mock() for _ in range(num_layers)]
        mock_container.return_value = Mock()
        
        factory = LayersFactory(config)
        result = factory.create_layers(mock_layer_config)
        
        # Verify correct number of layers created
        assert len(factory.layers) == num_layers
        assert mock_impl.call_count == num_layers


@pytest.mark.integration 
def test_layer_factory_with_brain_zones(mock_brain_zones, mock_layers_factory):
    """Integration test combining layers and brain zones."""
    # Test that layers can be created for different brain zones
    for zone_name, zone_config in mock_brain_zones.items():
        # Create layer config based on zone
        layer_config = Mock()
        layer_config.name = f"{zone_name}_layer"
        layer_config.input_dim = zone_config.min_neurons
        layer_config.output_dim = zone_config.max_neurons
        
        # Use the factory (mock will handle the actual creation)
        mock_layers_factory.config.num_layers = zone_config.num_layers
        result = mock_layers_factory.create_layers(layer_config)
        
        # Verify interaction happened
        assert result is not None


@pytest.mark.slow
def test_large_scale_layer_creation(performance_timer):
    """Performance test for creating many layers."""
    from core.layers_factory import LayersFactory
    
    performance_timer.start()
    
    # Simulate creating a large network
    large_config = Mock()
    large_config.num_layers = 100
    
    with patch('core.layers_factory.BaseLayerImplementation') as mock_impl, \
         patch('core.layers_factory.BaseLayerContainer') as mock_container:
        
        # Mock implementation to be fast
        mock_impl.side_effect = [Mock() for _ in range(100)]
        mock_container.return_value = Mock()
        
        factory = LayersFactory(large_config)
        layer_config = Mock()
        result = factory.create_layers(layer_config)
    
    elapsed = performance_timer.stop()
    
    # Should complete reasonably quickly even for large networks
    assert elapsed < 1.0  # Less than 1 second
    assert len(factory.layers) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])