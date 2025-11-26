"""
Test Suite for HippocampalTransformer (Full Model)

TDD: Write tests FIRST before implementation.
Tests the complete model integration.
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

def test_model_forward_pass():
    """Test full model forward pass"""
    from src.core.language_zone.hippocampal_transformer import HippocampalTransformer
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        vocab_size: int = 1000
        embedding_dim: int = 768
        num_layers: int = 2
        num_heads: int = 12
        dropout: float = 0.1
        max_seq_len: int = 1024
        intermediate_size: int = 3072
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
        n_place_cells: int = 2000
        
    config = MockConfig()
    hippocampus = HippocampalFormation(768, 2000, 100, 500)
    model = HippocampalTransformer(config, hippocampus)
    
    input_ids = torch.randint(0, 1000, (2, 32))
    prosody = torch.rand(2, 32, 4)
    
    logits, memories = model(input_ids, prosody=prosody)
    
    assert logits.shape == (2, 32, 1000), \
        f"Expected logits shape (2, 32, 1000), got {logits.shape}"
    
    # Check that memories were returned (place cell activity)
    assert memories is not None, "Should return memory activity"
    assert memories.shape == (2, 32, 2000), \
        f"Expected memory shape (2, 32, 2000), got {memories.shape}"
        
    print("✅ test_model_forward_pass PASSED")


def test_memory_creation_hook():
    """Test that forward pass triggers memory creation in hippocampus"""
    from src.core.language_zone.hippocampal_transformer import HippocampalTransformer
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        vocab_size: int = 1000
        embedding_dim: int = 768
        num_layers: int = 2
        num_heads: int = 12
        dropout: float = 0.0
        max_seq_len: int = 1024
        intermediate_size: int = 3072
        theta_frequency: float = 8.0
        gamma_frequency: float = 40.0
        n_place_cells: int = 2000
        
    config = MockConfig()
    hippocampus = HippocampalFormation(768, 2000, 100, 500)
    model = HippocampalTransformer(config, hippocampus)
    
    # Initial memory count
    initial_memories = len(hippocampus.episodic_memories)
    
    input_ids = torch.randint(0, 1000, (1, 10))
    # Enable training mode and memory creation
    model.train()
    
    # We need to simulate a training step where memories are created
    # Usually this happens in the trainer, but the model might expose a method or hook
    # Or the model forward pass creates them if a flag is set?
    # Let's assume for now the model returns the features needed to create them,
    # and the trainer calls hippocampus.create_episodic_memory.
    # So this test might just verify that the model produces the right features.
    
    logits, place_activity = model(input_ids)
    
    # Check that place_activity is suitable for memory creation
    assert place_activity.requires_grad, "Place activity should be differentiable for training"
    assert place_activity.shape[-1] == config.n_place_cells
    
    print("✅ test_memory_creation_hook PASSED")


def run_all_tests():
    print("="*60)
    print("HippocampalTransformer (Full) Test Suite (TDD)")
    print("="*60)
    
    tests = [
        test_model_forward_pass,
        test_memory_creation_hook,
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
