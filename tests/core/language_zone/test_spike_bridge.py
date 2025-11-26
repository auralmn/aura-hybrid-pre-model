import pytest
import torch
import numpy as np
from src.core.language_zone.spike_bridge import SpikeToContinuousBridge, ContinuousToSpikeBridge


class TestSpikeToContinuousBridge:
    """Test spike to continuous conversion."""
    
    def test_rate_encoding(self):
        """Test rate coding conversion."""
        bridge = SpikeToContinuousBridge(
            spike_dim=100,
            output_dim=64,
            encoding='rate',
            time_window=10
        )
        
        spikes = torch.randn(4, 50, 100)
        features = bridge(spikes)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (4, 64)
    
    def test_temporal_encoding(self):
        """Test temporal coding with exponential weighting."""
        bridge = SpikeToContinuousBridge(
            spike_dim=100,
            output_dim=100,
            encoding='temporal',
            time_window=10
        )
        
        # Recent spikes should have more weight
        spikes = torch.zeros(2, 50, 100)
        spikes[:, -5:, :] = 1.0  # Only recent spikes
        
        features = bridge(spikes)
        assert features.mean() > 0
    
    def test_phase_encoding(self):
        """Test phase/frequency encoding."""
        bridge = SpikeToContinuousBridge(
            spike_dim=100,
            output_dim=100,
            encoding='phase',
            time_window=16  # Power of 2 for FFT efficiency
        )
        
        # Create simple spike pattern
        spikes = torch.randn(2, 32, 100)  # More timesteps for FFT
        
        features = bridge(spikes)
        assert features.shape == (2, 100)
        assert not np.isnan(features).any()
    
    def test_dimension_projection(self):
        """Test learnable projection when dims don't match."""
        bridge = SpikeToContinuousBridge(
            spike_dim=128,
            output_dim=64,  # Different dimension
            encoding='rate'
        )
        
        assert hasattr(bridge, 'projection')
        
        spikes = torch.randn(2, 50, 128)
        features = bridge(spikes)
        assert features.shape == (2, 64)
    
    def test_gradient_flow(self):
        """Ensure gradients flow through projection."""
        bridge = SpikeToContinuousBridge(128, 64, encoding='rate')
        spikes = torch.randn(2, 50, 128, requires_grad=True)
        
        # Forward through bridge
        features_np = bridge(spikes)
        
        # Convert back to torch for gradient check
        features_torch = bridge.projection(spikes.mean(dim=1))
        loss = features_torch.sum()
        loss.backward()
        
        assert spikes.grad is not None


class TestContinuousToSpikeBridge:
    """Test continuous to spike conversion."""
    
    def test_poisson_encoding(self):
        """Test Poisson spike encoding."""
        bridge = ContinuousToSpikeBridge(
            input_dim=64,
            spike_dim=100,
            encoding='poisson',
            num_timesteps=20
        )
        
        continuous = np.random.randn(4, 64)
        device = torch.device('cpu')
        spikes = bridge(continuous, device)
        
        assert spikes.shape == (4, 20, 100)
        assert spikes.dtype == torch.float32
        # Poisson spikes should be binary
        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})
    
    def test_threshold_encoding(self):
        """Test threshold-based encoding."""
        bridge = ContinuousToSpikeBridge(
            input_dim=64,
            spike_dim=64,
            encoding='threshold',
            num_timesteps=10
        )
        
        # Positive values should spike
        continuous = np.ones((2, 64))
        spikes = bridge(continuous, torch.device('cpu'))
        
        assert spikes.sum() > 0
    
    def test_temporal_encoding(self):
        """Test temporal spread encoding."""
        bridge = ContinuousToSpikeBridge(
            input_dim=64,
            spike_dim=64,
            encoding='temporal',
            num_timesteps=10
        )
        
        continuous = np.random.randn(2, 64)
        spikes = bridge(continuous, torch.device('cpu'))
        
        assert spikes.shape == (2, 10, 64)
    
    def test_dimension_projection(self):
        """Test projection when dimensions differ."""
        bridge = ContinuousToSpikeBridge(
            input_dim=64,
            spike_dim=128,  # Different dimension
            encoding='poisson'
        )
        
        continuous = np.random.randn(2, 64)
        spikes = bridge(continuous, torch.device('cpu'))
        
        assert spikes.shape[2] == 128
    
    def test_numerical_stability(self):
        """Test with extreme input values."""
        bridge = ContinuousToSpikeBridge(64, 64, encoding='poisson')
        
        # Very large values
        continuous_large = np.ones((2, 64)) * 1000.0
        spikes_large = bridge(continuous_large, torch.device('cpu'))
        assert not torch.isnan(spikes_large).any()
        
        # Very small values
        continuous_small = np.ones((2, 64)) * 1e-6
        spikes_small = bridge(continuous_small, torch.device('cpu'))
        assert not torch.isnan(spikes_small).any()


class TestBridgeRoundtrip:
    """Test spike → continuous → spike conversion."""
    
    def test_information_preservation(self):
        """Verify information is approximately preserved."""
        s2c = SpikeToContinuousBridge(100, 64, encoding='rate', time_window=20)
        c2s = ContinuousToSpikeBridge(64, 100, encoding='poisson', num_timesteps=20)
        
        # Original spikes
        original_spikes = torch.randint(0, 2, (2, 50, 100)).float()
        
        # Convert to continuous
        continuous = s2c(original_spikes)
        
        # Convert back to spikes
        reconstructed_spikes = c2s(continuous, torch.device('cpu'))
        
        # Spike rates should be similar
        original_rate = original_spikes.mean().item()
        reconstructed_rate = reconstructed_spikes.mean().item()
        
        # Allow 50% tolerance due to stochastic encoding
        assert abs(original_rate - reconstructed_rate) < original_rate * 0.5
