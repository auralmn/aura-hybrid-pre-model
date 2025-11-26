"""
Test Suite for HippocampalTransformerLayer

TDD: Write tests FIRST before implementation.
Tests the full transformer layer integration.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from src.core.hippocampal import HippocampalFormation

def test_layer_shape():
    """Test output shapes are correct"""
    from src.core.language_zone.hippocampal_layer import HippocampalTransformerLayer
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 768
        num_heads: int = 12
        dropout: float = 0.1
        max_seq_len: int = 1024
        intermediate_size: int = 3072
        
    config = MockConfig()
    hippocampus = HippocampalFormation(768, 2000, 100, 500)
    layer = HippocampalTransformerLayer(config, hippocampus)
    
    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, config.embedding_dim)
    prosody = torch.rand(batch_size, seq_len, 4)
    
    output = layer(hidden_states, prosody=prosody)
    
    assert output.shape == (batch_size, seq_len, config.embedding_dim), \
        f"Expected output shape {(batch_size, seq_len, config.embedding_dim)}, got {output.shape}"
    
    print("✅ test_layer_shape PASSED")


def test_residual_connections():
    """Test that gradients flow through residuals"""
    from src.core.language_zone.hippocampal_layer import HippocampalTransformerLayer
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 768
        num_heads: int = 12
        dropout: float = 0.0
        max_seq_len: int = 1024
        intermediate_size: int = 3072
        
    config = MockConfig()
    hippocampus = HippocampalFormation(768, 2000, 100, 500)
    layer = HippocampalTransformerLayer(config, hippocampus)
    
    hidden_states = torch.randn(2, 32, 768, requires_grad=True)
    prosody = torch.rand(2, 32, 4)
    
    output = layer(hidden_states, prosody=prosody)
    loss = output.sum()
    loss.backward()
    
    assert hidden_states.grad is not None, "Gradient should flow back to input"
    assert torch.norm(hidden_states.grad) > 0, "Gradient should be non-zero"
    
    print("✅ test_residual_connections PASSED")


def test_feedforward_processing():
    """Test that feedforward network transforms data"""
    from src.core.language_zone.hippocampal_layer import HippocampalTransformerLayer
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 768
        num_heads: int = 12
        dropout: float = 0.0
        max_seq_len: int = 1024
        intermediate_size: int = 3072
        
    config = MockConfig()
    hippocampus = HippocampalFormation(768, 2000, 100, 500)
    layer = HippocampalTransformerLayer(config, hippocampus)
    
    # Mock attention to be identity to isolate FF
    # (Hard to mock internal component without dependency injection, 
    # so we'll just check that output is different from input)
    
    hidden_states = torch.randn(2, 32, 768)
    output = layer(hidden_states)
    
    # Output should be different from input (transformation occurred)
    diff = torch.norm(output - hidden_states).item()
    assert diff > 0.1, f"Layer should transform input (diff: {diff})"
    
    print(f"✅ test_feedforward_processing PASSED (diff: {diff:.4f})")


def run_all_tests():
    print("="*60)
    print("HippocampalTransformerLayer Test Suite (TDD)")
    print("="*60)
    
    tests = [
        test_layer_shape,
        test_residual_connections,
        test_feedforward_processing,
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
