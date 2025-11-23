"""
Unified test configuration and fixtures for the neuromorphic brain simulation project.

This file provides shared test fixtures, mock factories, and configuration
that can be used across all test modules in the tests/ directory.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import uuid
import sys
import os
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))


# ============================================================================
# SHARED FIXTURES - Configuration Objects
# ============================================================================

@pytest.fixture
def mock_neuron_config():
    """Mock BaseNeuronConfig with realistic parameters."""
    config = Mock()
    config.input_dim = 128
    config.hidden_dim = 256
    config.dt = 0.02
    config.tau_min = 0.02
    config.tau_max = 2.0
    return config


@pytest.fixture
def mock_layer_config():
    """Mock BaseLayerConfig with typical neural network layer settings."""
    config = Mock()
    config.name = "test_layer"
    config.input_dim = 128
    config.output_dim = 256
    config.activation = "relu"
    config.dropout_rate = 0.1
    return config


@pytest.fixture
def mock_layer_container_config():
    """Mock BaseLayerContainerConfig for multi-layer structures."""
    config = Mock()
    config.num_layers = 3
    config.layer_type = "dense"
    config.regularization = "l2"
    return config


@pytest.fixture
def mock_brain_zone_config():
    """Mock BrainZoneConfig for brain region simulation."""
    config = Mock()
    config.name = "test_cortex"
    config.min_neurons = 100
    config.max_neurons = 1000
    config.num_layers = 4
    config.connectivity = "dense"
    config.plasticity_enabled = True
    return config


# ============================================================================
# SHARED FIXTURES - State Objects
# ============================================================================

@pytest.fixture
def mock_neuronal_state(mock_neuron_config):
    """Complete mock NeuronalState with all required attributes."""
    state = Mock()
    state.id = "test-neuron-state"
    state.config = mock_neuron_config
    state.synapse = Mock()
    state.maturation_stage = Mock()
    state.activity_state = Mock()
    state.membrane_potential = -70.0
    state.gene_expression = {"BDNF": 0.8, "CREB": 0.6, "Arc": 0.3}
    state.cell_cycle = "G1"
    state.maturation = "mature"
    state.activity = "active"
    state.connections = ["neuron_001", "neuron_002", "neuron_003"]
    state.environment = {"temperature": 37.0, "pH": 7.4, "oxygen": 95.0}
    state.plasticity = {"LTP": 0.9, "LTD": 0.3, "STDP": 0.7}
    state.fatigue = 0.1
    
    # Weight matrices with proper dimensions
    hidden_dim = mock_neuron_config.hidden_dim
    input_dim = mock_neuron_config.input_dim
    
    state.W_hidden = np.random.normal(0, 0.1, (hidden_dim, hidden_dim))
    state.W_input = np.random.normal(0, 0.1, (hidden_dim, input_dim))
    state.W_tau = np.random.normal(0, 0.1, (hidden_dim, input_dim))
    state.bias = np.zeros(hidden_dim)
    state.tau_bias = np.zeros(hidden_dim)
    state.state = np.zeros(hidden_dim)
    
    return state


@pytest.fixture
def mock_synapse():
    """Mock Synapse object with typical synaptic properties."""
    synapse = Mock()
    synapse.target = "postsynaptic_neuron_001"
    synapse.weight = 0.5
    synapse.delay = 2.0
    synapse.plasticity_rule = "STDP"
    synapse.last_spike_time = 0.0
    return synapse


# ============================================================================
# SHARED FIXTURES - Factory Objects
# ============================================================================

@pytest.fixture
def mock_layers_factory():
    """Mock LayersFactory with create_layers capability."""
    factory = Mock()
    factory.config = Mock()
    factory.config.num_layers = 3
    factory.layers = []
    # Provide input_dim used by integration assertions
    factory.input_dim = 128
    
    def create_layers_side_effect(config):
        # Simulate layer creation
        layers = []
        num_layers = factory.config.num_layers if isinstance(factory.config.num_layers, int) else 3
        for i in range(num_layers):
            layer = Mock()
            layer.name = f"layer_{i}"
            layer.config = config
            layers.append(layer)
            factory.layers.append(layer)
        
        container = Mock()
        container.config = factory.config
        container.layers = factory.layers
        return container
    
    factory.create_layers.side_effect = create_layers_side_effect
    return factory


@pytest.fixture
def mock_neuron_factory():
    """Mock NeuronFactory with create_neuron capability."""
    factory = Mock()
    factory.input_dim = 128
    factory.hidden_dim = 256
    factory.output_dim = 384
    factory.neuron_config = Mock()
    
    def create_neuron_side_effect(state=None):
        if state is None:
            # Create default state
            state = Mock()
            state.id = str(uuid.uuid4())
            state.config = factory.neuron_config
        
        neuron = Mock()
        neuron.id = state.id
        neuron.config = state.config
        neuron.state = state
        return neuron
    
    factory.create_neuron.side_effect = create_neuron_side_effect
    factory._generate_neuron_id.return_value = str(uuid.uuid4())
    
    return factory


@pytest.fixture
def mock_brain_zone_factory():
    """Mock BrainZoneFactory for creating brain zones."""
    factory = Mock()
    
    def create_brain_zone_side_effect(config, layers):
        zone = Mock()
        zone.name = config.name
        zone.config = config
        zone.layers = layers
        zone.neurons = []
        zone.connectivity_matrix = np.random.rand(config.max_neurons, config.max_neurons)
        return zone
    
    factory.create_brain_zone.side_effect = create_brain_zone_side_effect
    return factory


# ============================================================================
# SHARED FIXTURES - Complete Systems
# ============================================================================

@pytest.fixture
def mock_brain_zones():
    """Complete set of mock brain zones for testing."""
    zones = {
        'cortex': Mock(name='cortex', min_neurons=1000, max_neurons=10000, num_layers=6),
        'thalamus': Mock(name='thalamus', min_neurons=200, max_neurons=2000, num_layers=3),
        'hippocampus': Mock(name='hippocampus', min_neurons=300, max_neurons=3000, num_layers=4),
        'amygdala': Mock(name='amygdala', min_neurons=80, max_neurons=800, num_layers=2),
        'cerebellum': Mock(name='cerebellum', min_neurons=500, max_neurons=5000, num_layers=5)
    }
    
    # Set names for each zone config
    for name, config in zones.items():
        config.name = name
        config.connectivity = "sparse"
        config.plasticity_enabled = True
    
    return zones


# ============================================================================
# SHARED FIXTURES - Test Data Generators
# ============================================================================

@pytest.fixture
def weight_matrix_generator():
    """Generator for creating weight matrices with different shapes."""
    def _generate(input_dim, hidden_dim, initialization='normal'):
        if initialization == 'normal':
            return np.random.normal(0, 0.1, (hidden_dim, input_dim))
        elif initialization == 'xavier':
            limit = np.sqrt(6.0 / (input_dim + hidden_dim))
            return np.random.uniform(-limit, limit, (hidden_dim, input_dim))
        elif initialization == 'zeros':
            return np.zeros((hidden_dim, input_dim))
        else:
            raise ValueError(f"Unknown initialization: {initialization}")
    
    return _generate


@pytest.fixture
def neural_activity_generator():
    """Generator for creating realistic neural activity patterns."""
    def _generate(num_neurons, time_steps, pattern='random'):
        if pattern == 'random':
            return np.random.rand(num_neurons, time_steps)
        elif pattern == 'oscillatory':
            t = np.linspace(0, 10, time_steps)
            base = np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz oscillation
            return np.tile(base, (num_neurons, 1)) + 0.1 * np.random.rand(num_neurons, time_steps)
        elif pattern == 'sparse':
            activity = np.zeros((num_neurons, time_steps))
            # 5% of neurons active at any time
            for t in range(time_steps):
                active_neurons = np.random.choice(num_neurons, size=num_neurons//20, replace=False)
                activity[active_neurons, t] = 1.0
            return activity
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    return _generate


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "parametrize: marks parametrized tests"
    )


# ============================================================================
# SHARED TEST UTILITIES
# ============================================================================

class MockFactory:
    """Utility class for creating consistent mock objects."""
    
    @staticmethod
    def create_neuron_mock(neuron_id=None, config=None):
        """Create a consistent mock neuron."""
        neuron = Mock()
        neuron.id = neuron_id or str(uuid.uuid4())
        neuron.config = config or Mock()
        neuron.membrane_potential = -70.0
        neuron.connections = []
        neuron.get_stats.return_value = {
            'id': neuron.id,
            'config': neuron.config,
            'membrane_potential': neuron.membrane_potential
        }
        return neuron
    
    @staticmethod
    def create_layer_mock(name="test_layer", input_dim=128, output_dim=256):
        """Create a consistent mock layer."""
        layer = Mock()
        layer.name = name
        layer.input_dim = input_dim
        layer.output_dim = output_dim
        layer.weights = np.random.normal(0, 0.1, (output_dim, input_dim))
        layer.bias = np.zeros(output_dim)
        return layer
    
    @staticmethod
    def create_brain_zone_mock(name="test_zone", num_neurons=100):
        """Create a consistent mock brain zone."""
        zone = Mock()
        zone.name = name
        zone.num_neurons = num_neurons
        zone.neurons = [MockFactory.create_neuron_mock() for _ in range(num_neurons)]
        zone.connectivity_matrix = np.random.rand(num_neurons, num_neurons)
        return zone


# ============================================================================
# PERFORMANCE TESTING UTILITIES
# ============================================================================

@pytest.fixture
def performance_timer():
    """Fixture for timing test execution."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment once per session."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Set environment variables if needed
    os.environ['TESTING'] = 'true'
    
    yield
    
    # Cleanup after tests
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


# Print configuration info when conftest is loaded
print(f"🧪 Test configuration loaded")
print(f"📁 Project root: {project_root}")
print(f"📦 Source path: {src_path}")
print(f"🎯 Ready for pytest execution with shared fixtures and utilities")