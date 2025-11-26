import pytest
import torch
import numpy as np
from src.core.language_zone.snn_ops import SNNMatmul, SNNSoftmax, SNNSiLU, SNNRMSNorm


class TestSNNMatmul:
    """Test spike-driven matrix multiplication."""
    
    def test_initialization(self):
        """Verify proper initialization."""
        matmul = SNNMatmul(in_features=128, out_features=256, scale=True)
        assert matmul.in_features == 128
        assert matmul.out_features == 256
        assert matmul.weight.shape == (256, 128)
        assert matmul.scale_factor > 0
    
    def test_forward_shape(self):
        """Test output shape."""
        matmul = SNNMatmul(100, 200)
        spikes = torch.randn(4, 50, 100)
        output = matmul(spikes)
        assert output.shape == (4, 50, 200)
    
    def test_scaling(self):
        """Verify scaling reduces magnitude."""
        matmul_scaled = SNNMatmul(100, 100, scale=True)
        matmul_unscaled = SNNMatmul(100, 100, scale=False)
        
        spikes = torch.randn(2, 50, 100) * 10.0
        
        out_scaled = matmul_scaled(spikes)
        out_unscaled = matmul_unscaled(spikes)
        
        # Scaled output should have smaller magnitude
        assert out_scaled.abs().mean() < out_unscaled.abs().mean()
    
    def test_gradient_flow(self):
        """Ensure gradients flow properly."""
        matmul = SNNMatmul(10, 20)
        spikes = torch.randn(2, 50, 10, requires_grad=True)
        output = matmul(spikes)
        loss = output.sum()
        loss.backward()
        
        assert spikes.grad is not None
        assert matmul.weight.grad is not None
        assert not torch.isnan(spikes.grad).any()


class TestSNNSoftmax:
    """Test spike-based softmax."""
    
    def test_normalization(self):
        """Verify outputs sum to 1 along dimension."""
        softmax = SNNSoftmax(dim=-1)
        spikes = torch.randn(4, 50, 100)
        output = softmax(spikes)
        
        # Check normalization
        sums = output.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_temperature_scaling(self):
        """Verify temperature controls sharpness."""
        spikes = torch.randn(2, 50, 100)
        
        softmax_low_temp = SNNSoftmax(temperature=0.5)
        softmax_high_temp = SNNSoftmax(temperature=2.0)
        
        out_low = softmax_low_temp(spikes)
        out_high = softmax_high_temp(spikes)
        
        # Lower temperature should be sharper (higher max values)
        assert out_low.max() > out_high.max()
    
    def test_numerical_stability(self):
        """Test with extreme values."""
        softmax = SNNSoftmax()
        spikes_large = torch.randn(2, 50, 100) * 100.0
        output = softmax(spikes_large)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestSNNSiLU:
    """Test piecewise SiLU approximation."""
    
    def test_forward_shape(self):
        """Test output shape preservation."""
        silu = SNNSiLU()
        spikes = torch.randn(4, 50, 100)
        output = silu(spikes)
        assert output.shape == spikes.shape
    
    def test_silu_properties(self):
        """Verify SiLU properties."""
        silu = SNNSiLU()
        
        # Test zero point
        assert torch.allclose(silu(torch.tensor([0.0])), torch.tensor([0.0]), atol=1e-6)
        
        # Test positive values are positive
        x_pos = torch.tensor([1.0, 2.0, 3.0])
        y_pos = silu(x_pos)
        assert (y_pos > 0).all()
    
    def test_gradient_flow(self):
        """Ensure gradients flow through SiLU."""
        silu = SNNSiLU()
        spikes = torch.randn(2, 50, 10, requires_grad=True)
        output = silu(spikes)
        loss = output.sum()
        loss.backward()
        
        assert spikes.grad is not None
        assert not torch.isnan(spikes.grad).any()


class TestSNNRMSNorm:
    """Test SNN-adapted RMS normalization."""
    
    def test_initialization(self):
        """Verify proper initialization."""
        norm = SNNRMSNorm(normalized_shape=128)
        assert norm.gamma.shape == (128,)
        assert torch.allclose(norm.gamma, torch.ones(128))
    
    def test_normalization_effect(self):
        """Verify RMS normalization effect."""
        norm = SNNRMSNorm(100)
        
        # Create spikes with varying magnitudes
        spikes = torch.randn(4, 50, 100) * torch.linspace(0.1, 10, 100)
        output = norm(spikes)
        
        # RMS should be more uniform after normalization
        input_rms = torch.sqrt(torch.mean(spikes ** 2, dim=-1))
        output_rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        
        # Output RMS variance should be lower
        assert output_rms.std() < input_rms.std()
    
    def test_learnable_gamma(self):
        """Verify gamma is learnable."""
        norm = SNNRMSNorm(10)
        spikes = torch.randn(2, 50, 10, requires_grad=True)
        output = norm(spikes)
        loss = output.sum()
        loss.backward()
        
        assert norm.gamma.grad is not None
        assert torch.any(norm.gamma.grad != 0)
    
    def test_numerical_stability(self):
        """Test with extreme values."""
        norm = SNNRMSNorm(100)
        
        # Very small values
        spikes_small = torch.randn(2, 50, 100) * 1e-6
        output_small = norm(spikes_small)
        assert not torch.isnan(output_small).any()
        
        # Very large values
        spikes_large = torch.randn(2, 50, 100) * 1e6
        output_large = norm(spikes_large)
        assert not torch.isinf(output_large).any()
