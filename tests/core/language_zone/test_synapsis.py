import torch
import pytest
import math
from src.core.language_zone.synapsis import Synapsis


class TestSynapsisInitialization:
    """Test proper initialization of Synapsis module."""
    
    def test_weight_shapes(self):
        """Verify weight matrix has correct dimensions."""
        syn = Synapsis(in_features=128, out_features=256)
        
        assert syn.weight.shape == (256, 128), f"Wrong weight shape: {syn.weight.shape}"
        assert syn.bias.shape == (256,), f"Wrong bias shape: {syn.bias.shape}"
    
    def test_plasticity_flag(self):
        """Verify plasticity state is properly initialized."""
        # Without plasticity
        syn_no_plast = Synapsis(128, 256, enable_plasticity=False)
        assert syn_no_plast.enable_plasticity == False
        
        # With plasticity
        syn_plast = Synapsis(128, 256, enable_plasticity=True)
        assert syn_plast.enable_plasticity == True
        assert hasattr(syn_plast, 'stdp_lr'), "Plasticity enabled but no STDP learning rate"
        assert hasattr(syn_plast, 'trace_decay'), "Plasticity enabled but no trace decay"
    
    def test_snn_aware_initialization(self):
        """Verify weights are initialized for spiking regime."""
        syn = Synapsis(100, 200, target_firing_rate=0.3)
        
        # Check weight statistics
        weight_std = syn.weight.std().item()
        expected_std = 1.0 / math.sqrt(100 * 0.3)  # Approximate
        
        # Should be in reasonable range (within 50% of expected)
        assert 0.5 * expected_std < weight_std < 1.5 * expected_std, \
            f"Weight std {weight_std} not in SNN-aware range"
        
        # Bias should be zero or small
        assert torch.allclose(syn.bias, torch.zeros_like(syn.bias), atol=1e-6), \
            "Bias should be initialized to zero"


class TestSynapsisForward:
    """Test forward pass computation."""
    
    def test_basic_forward_pass(self):
        """Verify basic spike → current transformation."""
        syn = Synapsis(10, 20)
        
        # Create binary spike input
        spikes = torch.randint(0, 2, (4, 50, 10), dtype=torch.float32)
        
        output, state = syn(spikes, state=None)
        
        # Check output shape
        assert output.shape == (4, 50, 20), f"Wrong output shape: {output.shape}"
        
        # Output should be real-valued (not just 0/1)
        assert output.dtype == torch.float32
        
        # State should be None if plasticity disabled
        assert state is None or isinstance(state, tuple)
    
    def test_output_values_reasonable(self):
        """Verify output magnitudes are in reasonable range."""
        syn = Synapsis(10, 20)
        
        # Dense spike input (high firing rate)
        spikes_dense = torch.ones(2, 100, 10)
        output_dense, _ = syn(spikes_dense)
        
        # Sparse spike input (low firing rate)
        spikes_sparse = torch.zeros(2, 100, 10)
        spikes_sparse[:, ::10, :] = 1.0  # 10% firing rate
        output_sparse, _ = syn(spikes_sparse)
        
        # Dense spikes should produce larger outputs
        assert output_dense.abs().mean() > output_sparse.abs().mean()
    
    def test_zero_spikes_zero_output(self):
        """Verify zero spikes produce zero output (assuming zero bias)."""
        syn = Synapsis(10, 20)
        syn.bias.data.zero_()  # Force zero bias
        
        spikes = torch.zeros(2, 50, 10)
        output, _ = syn(spikes)
        
        # Output should be all zeros (only bias contributes)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)


class TestTemporalAccumulation:
    """Test efficient temporal spike handling."""
    
    def test_temporal_consistency(self):
        """Verify output is consistent across time dimension."""
        syn = Synapsis(10, 20)
        
        # Create spikes with known pattern
        spikes = torch.zeros(2, 100, 10)
        spikes[:, 0, 0] = 1.0  # Spike at t=0
        spikes[:, 50, 5] = 1.0  # Spike at t=50
        
        output, _ = syn(spikes)
        
        # Outputs at t=0 and t=50 should be non-zero
        assert output[:, 0, :].abs().sum() > 0, "No output at t=0"
        assert output[:, 50, :].abs().sum() > 0, "No output at t=50"
    
    def test_batch_independence(self):
        """Verify different batch elements are processed independently."""
        syn = Synapsis(10, 20)
        
        # Create different spike patterns per batch
        spikes = torch.zeros(3, 50, 10)
        spikes[0, :, 0] = 1.0  # First batch: all spikes in channel 0
        spikes[1, :, 5] = 1.0  # Second batch: all spikes in channel 5
        spikes[2, :, :] = 0.0  # Third batch: no spikes
        
        output, _ = syn(spikes)
        
        # Outputs should be different across batches
        assert not torch.allclose(output[0], output[1], atol=1e-3)
        assert not torch.allclose(output[1], output[2], atol=1e-3)


class TestGradientFlow:
    """Test gradient propagation through Synapsis."""
    
    def test_gradient_to_input(self):
        """Ensure gradients flow back to input spikes."""
        syn = Synapsis(10, 20)
        
        spikes = torch.randn(2, 50, 10, requires_grad=True)
        output, _ = syn(spikes)
        
        loss = output.sum()
        loss.backward()
        
        # Check gradient exists
        assert spikes.grad is not None, "No gradient to input spikes"
        assert not torch.isnan(spikes.grad).any(), "NaN gradients"
        assert (spikes.grad.abs() > 0).any(), "All gradients are zero"
    
    def test_gradient_to_weights(self):
        """Ensure gradients flow to synaptic weights."""
        syn = Synapsis(10, 20)
        
        spikes = torch.randn(2, 50, 10)
        output, _ = syn(spikes)
        
        loss = output.sum()
        loss.backward()
        
        # Check weight gradients
        assert syn.weight.grad is not None, "No gradient to weights"
        assert not torch.isnan(syn.weight.grad).any(), "NaN weight gradients"
        assert (syn.weight.grad.abs() > 0).any(), "All weight gradients zero"
        
        # Check bias gradients
        assert syn.bias.grad is not None, "No gradient to bias"
        assert (syn.bias.grad.abs() > 0).any(), "All bias gradients zero"
    
    def test_gradient_magnitude(self):
        """Verify gradient magnitudes are reasonable (not exploding/vanishing)."""
        syn = Synapsis(10, 20)
        
        spikes = torch.randn(4, 100, 10, requires_grad=True)
        output, _ = syn(spikes)
        
        loss = output.mean()
        loss.backward()
        
        # Gradient magnitude should be in reasonable range
        grad_magnitude = spikes.grad.abs().mean().item()
        assert 1e-6 < grad_magnitude < 1e3, \
            f"Gradient magnitude {grad_magnitude} out of range"


class TestPlasticity:
    """Test STDP-like plasticity (if enabled)."""
    
    def test_plasticity_state_creation(self):
        """Verify plasticity state is created when enabled."""
        syn = Synapsis(10, 20, enable_plasticity=True)
        
        spikes = torch.randn(2, 50, 10)
        output, state = syn(spikes, state=None)
        
        # State should exist
        assert state is not None, "Plasticity enabled but no state returned"
        assert isinstance(state, tuple), "State should be tuple"
        
        # State should contain trace information
        pre_trace, post_trace = state
        assert pre_trace.shape == (2, 10), f"Wrong pre_trace shape: {pre_trace.shape}"
        assert post_trace.shape == (2, 20), f"Wrong post_trace shape: {post_trace.shape}"
    
    def test_plasticity_disabled(self):
        """Verify no plasticity computation when disabled."""
        syn = Synapsis(10, 20, enable_plasticity=False)
        
        spikes = torch.randn(2, 50, 10)
        output, state = syn(spikes, state=None)
        
        # No state should be returned
        assert state is None, "Plasticity disabled but state returned"


class TestNumericalStability:
    """Test numerical stability under edge cases."""
    
    def test_long_sequences(self):
        """Verify stability with very long sequences."""
        syn = Synapsis(10, 20)
        
        # 5000 timestep sequence
        spikes = torch.randn(2, 5000, 10)
        output, _ = syn(spikes)
        
        # Check for numerical issues
        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"
        assert output.abs().max() < 1e6, "Output magnitude too large"
    
    def test_large_batch(self):
        """Verify stability with large batch sizes."""
        syn = Synapsis(10, 20)
        
        # Large batch
        spikes = torch.randn(128, 50, 10)
        output, _ = syn(spikes)
        
        assert not torch.isnan(output).any()
        assert output.shape == (128, 50, 20)
    
    def test_extreme_spike_values(self):
        """Verify handling of extreme input values."""
        syn = Synapsis(10, 20)
        
        # Very large spikes
        spikes_large = torch.ones(2, 50, 10) * 100.0
        output_large, _ = syn(spikes_large)
        
        # Very small spikes
        spikes_small = torch.ones(2, 50, 10) * 0.001
        output_small, _ = syn(spikes_small)
        
        # Should not crash or produce NaN
        assert not torch.isnan(output_large).any()
        assert not torch.isnan(output_small).any()
