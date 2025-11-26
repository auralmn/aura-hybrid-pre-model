import pytest
import torch
import numpy as np
from src.core.language_zone.gif_neuron import GIFNeuron

class TestGIFNeuron:
    def test_initialization(self):
        """Test GIF Neuron initialization with SpikeLLM parameters."""
        neuron = GIFNeuron(input_dim=128, hidden_dim=256, L=16)
        assert neuron.input_dim == 128
        assert neuron.hidden_dim == 256
        assert neuron.L == 16

    def test_multi_bit_spiking(self):
        """Test that the neuron can emit multi-bit spikes (s > 1)."""
        input_dim = 1
        hidden_dim = 1
        L = 16
        neuron = GIFNeuron(input_dim=input_dim, hidden_dim=hidden_dim, L=L, threshold=1.0)
        
        # Manually set decay to 1.0 (no leak) for simple math
        neuron.decay = 1.0
        
        # Set weights to 1, bias to 0
        with torch.no_grad():
            neuron.linear.weight.fill_(1.0)
            neuron.linear.bias.fill_(0.0)
            
        # Input: 5.5. Should produce spike value 5 (floor(5.5 / 1.0))
        x = torch.tensor([[[5.5]]]) # (Batch=1, Time=1, Dim=1)
        
        output, (v, _) = neuron(x)
        
        assert output.item() == 5.0
        # Residual potential should be 0.5
        assert torch.isclose(v, torch.tensor([[[0.5]]]), atol=1e-5).all()

    def test_clipping_at_L(self):
        """Test that spikes are clipped at L."""
        input_dim = 1
        hidden_dim = 1
        L = 4
        neuron = GIFNeuron(input_dim=input_dim, hidden_dim=hidden_dim, L=L, threshold=1.0)
        neuron.decay = 1.0
        
        with torch.no_grad():
            neuron.linear.weight.fill_(1.0)
            neuron.linear.bias.fill_(0.0)
            
        # Input: 10.0. Should be clipped to L=4.
        x = torch.tensor([[[10.0]]])
        
        output, _ = neuron(x)
        
        assert output.item() == 4.0

    def test_accumulation_over_time(self):
        """Test that potential accumulates over time steps."""
        input_dim = 1
        hidden_dim = 1
        neuron = GIFNeuron(input_dim=input_dim, hidden_dim=hidden_dim, L=16, threshold=1.0)
        neuron.decay = 1.0
        
        with torch.no_grad():
            neuron.linear.weight.fill_(1.0)
            neuron.linear.bias.fill_(0.0)
            
        # Input: 0.6 followed by 0.6.
        # t=0: V=0.6, Spike=0
        # t=1: V=0.6+0.6=1.2, Spike=1, V_rem=0.2
        x = torch.tensor([[[0.6], [0.6]]]) # (Batch=1, Time=2, Dim=1)
        
        output, (v, _) = neuron(x)
        
        # Output shape: (1, 2, 1)
        assert output[0, 0, 0].item() == 0.0
        assert output[0, 1, 0].item() == 1.0
        assert torch.isclose(v, torch.tensor([[[0.2]]]), atol=1e-5).all()

    def test_quantization_error_metric(self):
        """
        Test the quantization error metric logic.
        This is more of a validation that the neuron *can* be used to measure this.
        Ideally, we want the output spikes * V_th to approximate the input current (integrated).
        """
        input_dim = 1
        hidden_dim = 1
        neuron = GIFNeuron(input_dim=input_dim, hidden_dim=hidden_dim, L=16, threshold=1.0)
        neuron.decay = 1.0
        
        with torch.no_grad():
            neuron.linear.weight.fill_(1.0)
            neuron.linear.bias.fill_(0.0)
            
        x = torch.randn(1, 100, 1)
        output, _ = neuron(x)
        
        # Reconstructed signal (approximate)
        reconstructed = output * neuron.threshold
        
        # Error should be bounded (roughly)
        # Note: This is a loose test, just checking shapes and types mostly
        error = (x - reconstructed).abs().mean()
        assert error < 2.0 # Arbitrary bound, just ensuring it runs

    def test_gradient_flow(self):
        """Ensure gradients propagate through surrogate."""
        neuron = GIFNeuron(10, 20, L=16)
        x = torch.randn(2, 50, 10, requires_grad=True)
        spikes, _ = neuron(x)
        loss = spikes.sum()
        loss.backward()
        
        assert x.grad is not None
        assert (x.grad.abs() > 1e-8).any(), "Gradients too small"
        assert not torch.isnan(x.grad).any(), "NaN gradients"

    def test_spike_distribution(self):
        """Verify multi-bit spikes use full range [0, L]."""
        neuron = GIFNeuron(10, 20, L=16)
        # Large input range to force spikes
        x = torch.randn(4, 100, 10) * 5.0  
        spikes, _ = neuron(x)
        
        unique_levels = torch.unique(spikes).detach().cpu().numpy()
        # We expect at least some variety in spike levels with random input
        assert len(unique_levels) > 1, f"Only {len(unique_levels)} spike levels used"
        assert spikes.max() <= 16, "Spikes exceed L"
        assert spikes.min() >= 0, "Negative spikes"

    def test_numerical_stability(self):
        """Ensure no overflow/underflow with long sequences."""
        neuron = GIFNeuron(10, 20, L=16)
        x = torch.randn(2, 2000, 10)  # Long sequence
        spikes, (v, theta) = neuron(x)
        
        assert not torch.isnan(spikes).any()
        assert not torch.isinf(v).any()
        assert (theta > 0).all(), "Threshold became negative"

    def test_state_independence(self):
        """Verify state doesn't leak between batches."""
        neuron = GIFNeuron(10, 20, L=16)
        x1 = torch.randn(2, 50, 10)
        x2 = torch.randn(2, 50, 10)
        
        # If we don't pass state, it should start fresh each time
        out1, _ = neuron(x1, state=None)
        out2, _ = neuron(x2, state=None)
        
        # With same random seed/input, outputs would be same. 
        # But here x1 and x2 are different random tensors.
        # We want to ensure that processing x1 doesn't affect x2 if we don't pass state.
        # Let's verify that passing state DOES affect output vs not passing it.
        
        # Case 1: Independent
        out2_independent, _ = neuron(x2, state=None)
        
        # Case 2: Dependent (chained)
        _, state1 = neuron(x1, state=None)
        out2_dependent, _ = neuron(x2, state=state1)
        
        # The outputs should be different because state1 carries history
        assert not torch.allclose(out2_independent, out2_dependent), "State did not affect output"
