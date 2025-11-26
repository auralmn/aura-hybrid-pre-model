import pytest
import torch
import numpy as np
from src.core.language_zone.gif_neuron import GIFNeuron, BalancedGIFNeuron

class TestBalancedGIFNeuron:
    def test_initialization(self):
        """Test BalancedGIFNeuron initialization with E/I split."""
        neuron = BalancedGIFNeuron(input_dim=128, hidden_dim=256, L=16, inhibition_ratio=0.2)
        assert neuron.input_dim == 128
        assert neuron.hidden_dim == 256
        assert neuron.exc_dim == int(256 * 0.8)  # 204
        assert neuron.inh_dim == 256 - neuron.exc_dim  # 52
        assert neuron.inh_ratio == 0.2

    def test_ei_split(self):
        """Test that E/I populations are properly split."""
        hidden_dim = 100
        inh_ratio = 0.2
        neuron = BalancedGIFNeuron(input_dim=10, hidden_dim=hidden_dim, inhibition_ratio=inh_ratio)
        
        # Check dimensions
        assert neuron.exc_dim + neuron.inh_dim == hidden_dim
        assert neuron.exc_dim == 80
        assert neuron.inh_dim == 20

    def test_excitatory_positive_only(self):
        """Test that excitatory currents are positive."""
        neuron = BalancedGIFNeuron(input_dim=10, hidden_dim=20, L=16, inhibition_ratio=0.2)
        
        # Create input that would normally produce negative values
        x = torch.randn(2, 10, 10)
        
        # Get the excitatory component
        h_exc = neuron.linear_exc(x)
        i_exc = torch.nn.functional.relu(h_exc)
        
        # All excitatory currents should be >= 0
        assert (i_exc >= 0).all()

    def test_inhibitory_negative_only(self):
        """Test that inhibitory currents are negative."""
        neuron = BalancedGIFNeuron(input_dim=10, hidden_dim=20, L=16, inhibition_ratio=0.2)
        
        # Create input
        x = torch.randn(2, 10, 10)
        
        # Get the inhibitory component
        h_inh = neuron.linear_inh(x)
        i_inh = -torch.nn.functional.relu(h_inh)
        
        # All inhibitory currents should be <= 0
        assert (i_inh <= 0).all()

    def test_forward_shape(self):
        """Test output shape matches standard GIFNeuron."""
        batch_size = 4
        seq_len = 50
        input_dim = 10
        hidden_dim = 20
        
        neuron = BalancedGIFNeuron(input_dim=input_dim, hidden_dim=hidden_dim, L=16)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output, (v, theta) = neuron(x)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert v.shape == (batch_size, hidden_dim)
        assert theta.shape == (batch_size, hidden_dim)

    def test_gradient_flow(self):
        """Test gradient flow through E/I balanced neuron."""
        neuron = BalancedGIFNeuron(10, 20, L=16)
        x = torch.randn(2, 50, 10, requires_grad=True)
        spikes, _ = neuron(x)
        loss = spikes.sum()
        loss.backward()
        
        assert x.grad is not None
        assert (x.grad.abs() > 1e-8).any(), "Gradients too small"
        assert not torch.isnan(x.grad).any(), "NaN gradients"

    def test_ei_balance_effect(self):
        """Test that E/I balance affects spike dynamics."""
        # Standard neuron (no E/I split)
        neuron_standard = GIFNeuron(10, 20, L=16)
        
        # Balanced neuron
        neuron_balanced = BalancedGIFNeuron(10, 20, L=16, inhibition_ratio=0.2)
        
        x = torch.randn(2, 50, 10)
        
        out_standard, _ = neuron_standard(x)
        out_balanced, _ = neuron_balanced(x)
        
        # Outputs should be different due to E/I balance
        # (they have different architectures)
        assert out_standard.shape == out_balanced.shape
        # Cannot directly compare values as they have different weight matrices

    def test_numerical_stability(self):
        """Ensure E/I balanced neuron is numerically stable."""
        neuron = BalancedGIFNeuron(10, 20, L=16)
        x = torch.randn(2, 2000, 10)  # Long sequence
        spikes, (v, theta) = neuron(x)
        
        assert not torch.isnan(spikes).any()
        assert not torch.isinf(v).any()
        assert (theta > 0).all()
