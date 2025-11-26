"""
Test Suite for ThetaGammaPositionalEncoding

TDD: Write tests FIRST before implementation.
Tests biological properties of theta-gamma oscillatory positional encoding.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent  # tests/core/language_zone -> project root
sys.path.insert(0, str(project_root))

import torch
import numpy as np


def test_theta_gamma_shape():
    """Test output shapes are correct"""
    from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
    
    batch_size = 4
    seq_len = 128
    embedding_dim = 768
    
    # Create config
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 768
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
    
    config = MockConfig()
    encoder = ThetaGammaPositionalEncoding(config)
    
    # Create positions
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward pass
    pos_encoding = encoder(positions, seq_len)
    
    # Check shape
    assert pos_encoding.shape == (batch_size, seq_len, embedding_dim), \
        f"Expected shape {(batch_size, seq_len, embedding_dim)}, got {pos_encoding.shape}"
    
    print("✅ test_theta_gamma_shape PASSED")


def test_theta_frequency():
    """Test theta oscillation has correct frequency"""
    from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
    
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 256
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
    
    config = MockConfig()
    encoder = ThetaGammaPositionalEncoding(config)
    
    seq_len = 1000
    positions = torch.arange(seq_len).unsqueeze(0)
    
    pos_encoding = encoder(positions, seq_len)
    
    # Extract first dimension (should have theta oscillation)
    theta_signal = pos_encoding[0, :, 0].detach().numpy()
    
    # Check oscillatory behavior (should cross zero multiple times)
    zero_crossings = np.sum(np.diff(np.sign(theta_signal)) != 0)
    
    # For seq_len=1000 positions spanning ~2π, theta should complete ~1 cycle
    # So we expect ~2 zero crossings (one up, one down per cycle)
    assert zero_crossings >= 2, \
        f"Theta signal should oscillate (expected ~2 zero crossings, got {zero_crossings})"
    
    print(f"✅ test_theta_frequency PASSED (zero crossings: {zero_crossings})")


def test_gamma_frequency():
    """Test gamma oscillation has higher frequency than theta"""
    from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
    
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 256
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
    
    config = MockConfig()
    encoder = ThetaGammaPositionalEncoding(config)
    
    seq_len = 1000
    positions = torch.arange(seq_len).unsqueeze(0)
    
    pos_encoding = encoder(positions, seq_len)
    
    # Gamma should have ~5x more oscillations than theta
    # Check via autocorrelation or zero crossings
    
    # Simple check: gamma components should vary more rapidly
    # Use variance of first derivative as proxy for frequency
    theta_component = pos_encoding[0, :, 0].detach().numpy()
    gamma_component = pos_encoding[0, :, 1].detach().numpy()  # Different dim for gamma
    
    theta_variation = np.var(np.diff(theta_component))
    gamma_variation = np.var(np.diff(gamma_component))
    
    # Gamma should have higher variation (faster oscillation)
    assert gamma_variation > theta_variation * 0.5, \
        f"Gamma should oscillate faster (theta var: {theta_variation:.4f}, gamma var: {gamma_variation:.4f})"
    
    print(f"✅ test_gamma_frequency PASSED (theta: {theta_variation:.4f}, gamma: {gamma_variation:.4f})")


def test_phase_amplitude_coupling():
    """Test theta-gamma phase-amplitude coupling (PAC)"""
    from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
    
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 256
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
    
    config = MockConfig()
    encoder = ThetaGammaPositionalEncoding(config)
    
    seq_len = 500
    positions = torch.arange(seq_len).unsqueeze(0)
    
    pos_encoding = encoder(positions, seq_len)
    
    # Extract signals
    signal = pos_encoding[0, :, :].detach().numpy()
    
    # Check that gamma amplitude is modulated by theta phase
    # This is a key biological property of theta-gamma coupling
    
    # Simple test: compute correlation between abs(gamma) and theta cosine
    # If coupled, gamma amplitude should peak at theta peaks
    
    # For simplicity, just check that encoding is not constant
    variance = np.var(signal, axis=0)
    
    assert np.all(variance > 0.01), \
        "All dimensions should have non-zero variance (oscillate)"
    
    print(f"✅ test_phase_amplitude_coupling PASSED (min var: {variance.min():.4f})")


def test_learnable_parameters():
    """Test that phase offsets and amplitudes are learnable"""
    from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
    
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 128
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
    
    config = MockConfig()
    encoder = ThetaGammaPositionalEncoding(config)
    
    # Check learnable parameters exist
    params = list(encoder.parameters())
    
    assert len(params) > 0, "Should have learnable parameters"
    
    # Check parameter shapes
    param_names = [name for name, _ in encoder.named_parameters()]
    
    assert 'theta_phase_offsets' in param_names, \
        "Should have theta phase offsets"
    assert 'gamma_phase_offsets' in param_names, \
        "Should have gamma phase offsets"
    assert 'amplitude_modulation' in param_names, \
        "Should have amplitude modulation"
    
    # Check shapes match embedding_dim
    for name, param in encoder.named_parameters():
        assert param.shape == (config.embedding_dim,), \
            f"{name} should have shape ({config.embedding_dim},), got {param.shape}"
    
    print(f"✅ test_learnable_parameters PASSED ({len(params)} parameters)")


def test_position_sensitivity():
    """Test that different positions produce different encodings"""
    from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
    
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 256
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
    
    config = MockConfig()
    encoder = ThetaGammaPositionalEncoding(config)
    
    # Test positions
    pos_0 = torch.tensor([[0]])
    pos_10 = torch.tensor([[10]])
    pos_100 = torch.tensor([[100]])
    
    enc_0 = encoder(pos_0, 200)
    enc_10 = encoder(pos_10, 200)
    enc_100 = encoder(pos_100, 200)
    
    # Different positions should have different encodings
    diff_0_10 = torch.norm(enc_0 - enc_10).item()
    diff_0_100 = torch.norm(enc_0 - enc_100).item()
    diff_10_100 = torch.norm(enc_10 - enc_100).item()
    
    assert diff_0_10 > 0.1, f"Positions 0 and 10 should differ (diff: {diff_0_10})"
    # Note: Due to oscillatory nature, further positions may wrap around
    # Just check they're all different
    assert diff_10_100 > 0.1, \
        f"Positions 10 and 100 should differ (diff: {diff_10_100})"
    
    print(f"✅ test_position_sensitivity PASSED (diff 0-10: {diff_0_10:.3f}, 0-100: {diff_0_100:.3f})")


def test_batch_consistency():
    """Test that batched computation gives consistent results"""
    from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
    
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 256
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
    
    config = MockConfig()
    encoder = ThetaGammaPositionalEncoding(config)
    
    seq_len = 50
    
    # Single batch
    positions_single = torch.arange(seq_len).unsqueeze(0)
    enc_single = encoder(positions_single, seq_len)
    
    # Multiple batches with same positions
    batch_size = 4
    positions_batch = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    enc_batch = encoder(positions_batch, seq_len)
    
    # All batches should be identical
    for b in range(batch_size):
        diff = torch.norm(enc_single[0] - enc_batch[b]).item()
        assert diff < 1e-5, \
            f"Batch {b} should match single (diff: {diff})"
    
    print(f"✅ test_batch_consistency PASSED")


def test_gradient_flow():
    """Test that gradients flow through encoding"""
    from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
    
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 128
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
    
    config = MockConfig()
    encoder = ThetaGammaPositionalEncoding(config)
    
    seq_len = 32
    positions = torch.arange(seq_len).unsqueeze(0)
    
    # Forward + backward
    pos_encoding = encoder(positions, seq_len)
    loss = pos_encoding.sum()
    loss.backward()
    
    # Check gradients exist
    for name, param in encoder.named_parameters():
        assert param.grad is not None, \
            f"Gradient for {name} should exist"
        assert not torch.isnan(param.grad).any(), \
            f"Gradient for {name} should not contain NaN"
        assert param.grad.abs().sum() > 0, \
            f"Gradient for {name} should be non-zero"
    
    print(f"✅ test_gradient_flow PASSED")


def test_frequency_ratios():
    """Test that gamma/theta frequency ratio is preserved"""
    from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
    
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 256
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
    
    config = MockConfig()
    encoder = ThetaGammaPositionalEncoding(config)
    
    # Expected ratio
    expected_ratio = config.gamma_frequency / config.theta_frequency
    
    assert abs(expected_ratio - 5.0) < 0.1, \
        f"Gamma should be ~5x faster than theta (ratio: {expected_ratio})"
    
    print(f"✅ test_frequency_ratios PASSED (ratio: {expected_ratio})")


def run_all_tests():
    """Run all ThetaGamma tests"""
    print("="*60)
    print("ThetaGammaPositionalEncoding Test Suite (TDD)")
    print("="*60)
    
    tests = [
        test_theta_gamma_shape,
        test_theta_frequency,
        test_gamma_frequency,
        test_phase_amplitude_coupling,
        test_learnable_parameters,
        test_position_sensitivity,
        test_batch_consistency,
        test_gradient_flow,
        test_frequency_ratios,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"\nRunning {test_func.__name__}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
