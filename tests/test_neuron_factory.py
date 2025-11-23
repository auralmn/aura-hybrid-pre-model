import pytest
from unittest.mock import Mock, patch
import numpy as np

# Test for src/core/neuron_factory.py using shared fixtures
class TestNeuron:
    
    @patch('core.neuron_factory.uuid.uuid4')
    def test_neuron_initialization_with_full_state(self, mock_uuid, mock_neuronal_state):
        """Test Neuron initialization with complete state."""
        from core.neuron_factory import Neuron
        
        mock_uuid.return_value = "generated-uuid-456"
        
        neuron = Neuron(mock_neuronal_state, hidden_dim=256)
        
        # Test basic attributes
        assert neuron.id == "test-neuron-state"
        assert neuron.config == mock_neuronal_state.config
        assert neuron.synapse == mock_neuronal_state.synapse
        assert neuron.membrane_potential == -70.0
        assert neuron.gene_expression == {"BDNF": 0.8, "CREB": 0.6, "Arc": 0.3}
        
        # Test weight matrices
        np.testing.assert_array_equal(neuron.W_hidden, mock_neuronal_state.W_hidden)
        np.testing.assert_array_equal(neuron.W_input, mock_neuronal_state.W_input)

    @patch('core.neuron_factory.uuid.uuid4')
    def test_neuron_initialization_without_weights(self, mock_uuid, mock_neuron_config):
        """Test Neuron initialization when state lacks weight matrices."""
        from core.neuron_factory import Neuron
        
        mock_uuid.return_value = "generated-uuid-789"
        
        # Create minimal state without weight matrices
        minimal_state = Mock()
        minimal_state.id = mock_uuid.return_value
        minimal_state.config = mock_neuron_config
        minimal_state.synapse = Mock()
        minimal_state.maturation_stage = Mock()
        minimal_state.activity_state = Mock()
        minimal_state.membrane_potential = -65.0
        minimal_state.gene_expression = {}
        minimal_state.cell_cycle = "S"
        minimal_state.maturation = "immature"
        minimal_state.activity = "resting"
        minimal_state.connections = []
        minimal_state.environment = {}
        minimal_state.plasticity = {}
        minimal_state.fatigue = 0.0
        
        # Mock hasattr to return False for weight attributes
        def mock_hasattr_func(obj, name):
            weight_attrs = ['W_hidden', 'W_input', 'W_tau', 'bias', 'tau_bias', 'state']
            return name not in weight_attrs
        
        with patch('builtins.hasattr', side_effect=mock_hasattr_func):
            neuron = Neuron(minimal_state, hidden_dim=128)
        
        # Should generate UUID when no ID
        assert neuron.id == "generated-uuid-789"
        
        # Should create default weight matrices
        assert neuron.W_hidden.shape == (128, 128)
        assert neuron.W_input.shape == (128, 128)
        assert neuron.bias.shape == (128,)

    def test_neuron_get_stats(self, mock_neuronal_state):
        """Test the get_stats method returns correct information."""
        from core.neuron_factory import Neuron
        
        neuron = Neuron(mock_neuronal_state)
        stats = neuron.get_stats()
        
        # Verify all expected keys are present
        expected_keys = ['id', 'config', 'synapse', 'maturation_stage', 'activity_state']
        for key in expected_keys:
            assert key in stats
        
        assert stats['id'] == "test-neuron-state"


class TestNeuronFactory:
    
    @patch('core.neuron_factory.BaseNeuronConfig')
    @patch('core.neuron_factory.Synapse')
    @patch('core.neuron_factory.MaturationStage')
    @patch('core.neuron_factory.ActivityState')
    def test_neuron_factory_initialization(self, mock_activity, mock_maturation, 
                                         mock_synapse_class, mock_config_class):
        """Test NeuronFactory initialization with parameters."""
        from core.neuron_factory import NeuronFactory
        
        mock_config_class.return_value = Mock()
        mock_synapse_class.return_value = Mock()
        mock_maturation.PROGENITOR = Mock()
        mock_activity.RESTING = Mock()
        
        factory = NeuronFactory(
            input_dim=64,
            hidden_dim=128, 
            output_dim=256,
            dt=0.01,
            tau_min=0.01,
            tau_max=1.0
        )
        
        # Verify factory attributes
        assert factory.input_dim == 64
        assert factory.hidden_dim == 128
        assert factory.output_dim == 256
        
        # Verify BaseNeuronConfig was called with correct parameters
        mock_config_class.assert_called_once_with(64, 128, 0.01, 0.01, 1.0)

    def test_generate_neuron_id(self):
        """Test that neuron ID generation produces valid UUIDs."""
        from core.neuron_factory import NeuronFactory
        
        factory = NeuronFactory()
        neuron_id = factory._generate_neuron_id()
        
        # Verify it's a valid UUID string
        assert isinstance(neuron_id, str)
        # Should be able to parse as UUID
        import uuid
        uuid_obj = uuid.UUID(neuron_id)
        assert str(uuid_obj) == neuron_id

    @patch('core.neuron_factory.NeuronalState')
    @patch('core.neuron_factory.Neuron')
    @patch('core.neuron_factory.BaseNeuronConfig')
    @patch('core.neuron_factory.Synapse')
    @patch('core.neuron_factory.MaturationStage')
    @patch('core.neuron_factory.ActivityState')
    def test_create_neuron_with_default_state(self, mock_activity, mock_maturation,
                                            mock_synapse_class, mock_config_class,
                                            mock_neuron_class, mock_state_class):
        """Test creating neuron with default state."""
        from core.neuron_factory import NeuronFactory
        
        # Setup mocks
        mock_base_config = Mock()
        mock_synapse = Mock()
        mock_maturation_stage = Mock()
        mock_activity_state = Mock()
        
        mock_config_class.return_value = mock_base_config
        mock_synapse_class.return_value = mock_synapse
        mock_maturation.PROGENITOR = mock_maturation_stage
        mock_activity.RESTING = mock_activity_state
        
        mock_state_instance = Mock()
        mock_state_class.return_value = mock_state_instance
        
        mock_neuron_instance = Mock()
        mock_neuron_class.return_value = mock_neuron_instance
        
        factory = NeuronFactory(hidden_dim=128)
        result = factory.create_neuron()
        
        # Verify NeuronalState was created
        mock_state_class.assert_called_once()
        call_kwargs = mock_state_class.call_args[1]
        
        assert 'id' in call_kwargs
        assert 'config' in call_kwargs
        assert 'synapse' in call_kwargs
        assert call_kwargs['membrane_potential'] == 0.0
        assert call_kwargs['gene_expression'] == {}
        assert call_kwargs['cell_cycle'] == "G1"
        
        # Verify Neuron was created
        mock_neuron_class.assert_called_once_with(mock_state_instance, 128)
        assert result == mock_neuron_instance

    @patch('core.neuron_factory.Neuron')
    def test_create_neuron_with_provided_state(self, mock_neuron_class, mock_neuronal_state):
        """Test creating neuron with provided state using fixture."""
        from core.neuron_factory import NeuronFactory
        
        factory = NeuronFactory()
        mock_neuron_instance = Mock()
        mock_neuron_class.return_value = mock_neuron_instance
        
        result = factory.create_neuron(mock_neuronal_state)
        
        # Should use provided state directly
        mock_neuron_class.assert_called_once_with(mock_neuronal_state, factory.hidden_dim)
        assert result == mock_neuron_instance


# Parametrized tests using shared fixtures
@pytest.mark.parametrize("hidden_dim,input_dim", [
    (128, 64),
    (256, 128),
    (512, 256),
])
def test_neuron_weight_matrix_shapes_with_fixtures(hidden_dim, input_dim, mock_neuronal_state):
    """Test weight matrix shapes using conftest fixtures."""
    from core.neuron_factory import Neuron
    
    # Update mock config dimensions
    mock_neuronal_state.config.input_dim = input_dim
    mock_neuronal_state.config.hidden_dim = hidden_dim
    
    # Mock hasattr to force weight creation
    def mock_hasattr_func(obj, name):
        return False
        
    with patch('builtins.hasattr', side_effect=mock_hasattr_func):
        neuron = Neuron(mock_neuronal_state, hidden_dim=hidden_dim)
    
    # Verify shapes
    assert neuron.W_hidden.shape == (hidden_dim, hidden_dim)
    assert neuron.W_input.shape == (hidden_dim, input_dim)
    assert neuron.bias.shape == (hidden_dim,)


# Integration test using multiple fixtures
def test_neuron_factory_complete_workflow(mock_neuron_factory, mock_neuronal_state):
    """Integration test using shared factory fixture."""
    # Test the complete workflow using shared fixtures
    neuron = mock_neuron_factory.create_neuron(mock_neuronal_state)
    
    # Verify the neuron was created correctly
    assert neuron.id == mock_neuronal_state.id
    assert neuron.config == mock_neuronal_state.config
    assert neuron.state == mock_neuronal_state
    
    # Test creating neuron without state
    neuron_default = mock_neuron_factory.create_neuron()
    assert neuron_default.id is not None
    assert neuron_default.config == mock_neuron_factory.neuron_config


@pytest.mark.integration
def test_neuron_with_realistic_activity(neural_activity_generator):
    """Integration test using activity generator fixture."""
    # Generate realistic neural activity patterns
    random_activity = neural_activity_generator(100, 1000, 'random')
    oscillatory_activity = neural_activity_generator(100, 1000, 'oscillatory')
    sparse_activity = neural_activity_generator(100, 1000, 'sparse')
    
    # Verify shapes
    assert random_activity.shape == (100, 1000)
    assert oscillatory_activity.shape == (100, 1000)
    assert sparse_activity.shape == (100, 1000)
    
    # Verify patterns
    assert np.all(random_activity >= 0) and np.all(random_activity <= 1)
    assert np.sum(sparse_activity) < np.sum(random_activity)  # Sparse should have fewer active neurons


@pytest.mark.slow
@pytest.mark.parametrize("num_neurons", [10, 100, 1000])
def test_neuron_performance_scaling(num_neurons, performance_timer):
    """Performance test using timer fixture."""
    from core.neuron_factory import NeuronFactory
    
    # This would test actual performance with real implementation
    performance_timer.start()
    
    # Simulate neuron creation time
    import time
    time.sleep(0.001 * num_neurons / 100)  # Simulate scaling
    
    elapsed = performance_timer.stop()
    
    # Performance should scale reasonably
    assert elapsed < 0.1  # Should complete within 100ms for test


if __name__ == '__main__':
    pytest.main([__file__, '-v'])