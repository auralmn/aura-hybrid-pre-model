"""
Test Suite for HippocampalProsodyAttention

TDD: Write tests FIRST before implementation.
Tests integration of prosody modulation and hippocampal memory retrieval in attention.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.core.hippocampal import HippocampalFormation

def test_attention_shape():
    """Test output shapes are correct"""
    from src.core.language_zone.hippocampal_attention import HippocampalProsodyAttention
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 768
        num_heads: int = 12
        dropout: float = 0.1
        max_seq_len: int = 1024
        
    config = MockConfig()
    hippocampus = HippocampalFormation(768, 2000, 100, 500)
    attention = HippocampalProsodyAttention(config, hippocampus)
    
    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, config.embedding_dim)
    prosody = torch.rand(batch_size, seq_len, 4)  # 4 prosody features
    
    output, attn_weights = attention(hidden_states, prosody=prosody)
    
    assert output.shape == (batch_size, seq_len, config.embedding_dim), \
        f"Expected output shape {(batch_size, seq_len, config.embedding_dim)}, got {output.shape}"
    
    # Weights might be [batch, heads, seq, seq] or similar
    assert attn_weights.shape[0] == batch_size, "Batch size mismatch in weights"
    assert attn_weights.shape[2] == seq_len, "Seq len mismatch in weights"
    
    print("✅ test_attention_shape PASSED")


def test_prosody_modulation():
    """Test that prosody affects attention weights"""
    from src.core.language_zone.hippocampal_attention import HippocampalProsodyAttention
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 768
        num_heads: int = 12
        dropout: float = 0.0  # No dropout for deterministic test
        max_seq_len: int = 1024
        
    config = MockConfig()
    hippocampus = HippocampalFormation(768, 2000, 100, 500)
    attention = HippocampalProsodyAttention(config, hippocampus)
    
    hidden_states = torch.randn(2, 32, config.embedding_dim)
    
    # Case 1: Low prosody (neutral)
    prosody_low = torch.zeros(2, 32, 4)
    _, weights_low = attention(hidden_states, prosody=prosody_low)
    
    # Case 2: High prosody (emotional/emphatic)
    prosody_high = torch.ones(2, 32, 4)
    _, weights_high = attention(hidden_states, prosody=prosody_high)
    
    # Weights should be different
    diff = torch.norm(weights_low - weights_high).item()
    assert diff > 0.01, f"Prosody should modulate attention weights (diff: {diff})"
    
    # High prosody should typically lead to sharper attention (higher max weight)
    # or different routing. Just checking difference for now.
    
    print(f"✅ test_prosody_modulation PASSED (diff: {diff:.4f})")


def test_memory_retrieval_integration():
    """Test that attention retrieves and uses hippocampal memories"""
    from src.core.language_zone.hippocampal_attention import HippocampalProsodyAttention
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 768
        num_heads: int = 12
        dropout: float = 0.0
        max_seq_len: int = 1024
        
    config = MockConfig()
    hippocampus = HippocampalFormation(768, 2000, 100, 500)
    attention = HippocampalProsodyAttention(config, hippocampus)
    
    # Pre-seed a memory in hippocampus
    memory_key = torch.randn(768)
    memory_key = memory_key / memory_key.norm()
    # Use create_episodic_memory instead of store_memory
    hippocampus.create_episodic_memory(
        memory_id="test_mem_1",
        event_id="test_event",
        features=memory_key.numpy()
    )
    
    # Input with variation so attention weights are non-uniform
    # If inputs are identical, attention is uniform (1/N) regardless of scaling
    torch.manual_seed(42)
    hidden_states = torch.randn(1, 10, 768)
    prosody = torch.zeros(1, 10, 4)
    
    # Forward pass with memory retrieval enabled
    output, _ = attention(hidden_states, prosody=prosody, use_memory=True)
    
    # Output should be influenced by memory
    # (This is hard to test deterministically without implementation details, 
    # but we can check if the flag runs without error and affects output)
    
    output_no_mem, _ = attention(hidden_states, prosody=prosody, use_memory=False)
    
    diff = torch.norm(output - output_no_mem).item()
    assert diff > 1e-6, f"Memory retrieval should affect output (diff: {diff})"
    
    print(f"✅ test_memory_retrieval_integration PASSED (diff: {diff:.6f})")


def test_causal_masking():
    """Test that future tokens are masked (causal attention)"""
    from src.core.language_zone.hippocampal_attention import HippocampalProsodyAttention
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        embedding_dim: int = 768
        num_heads: int = 12
        dropout: float = 0.0
        max_seq_len: int = 1024
        
    config = MockConfig()
    hippocampus = HippocampalFormation(768, 2000, 100, 500)
    attention = HippocampalProsodyAttention(config, hippocampus)
    
    batch_size = 1
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.embedding_dim)
    prosody = torch.zeros(batch_size, seq_len, 4)
    
    _, weights = attention(hidden_states, prosody=prosody)
    
    # Weights shape: [batch, heads, query_len, key_len]
    # Upper triangle (excluding diagonal) should be -inf or 0 (after softmax)
    # Here we check the raw weights or post-softmax weights. 
    # Usually attention returns post-softmax weights.
    
    # Check that position i cannot attend to j > i
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            val = weights[0, 0, i, j].item()
            assert val == 0.0, f"Future token {j} visible at step {i} (val: {val})"
            
    print("✅ test_causal_masking PASSED")


def run_all_tests():
    print("="*60)
    print("HippocampalProsodyAttention Test Suite (TDD)")
    print("="*60)
    
    tests = [
        test_attention_shape,
        test_prosody_modulation,
        test_memory_retrieval_integration,
        test_causal_masking,
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
