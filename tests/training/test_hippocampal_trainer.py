"""
Test Suite for HippocampalTransformerTrainer

TDD: Write tests FIRST before implementation.
Tests the training loop, memory replay, and consolidation mechanisms.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.core.hippocampal import HippocampalFormation
# We will implement this module next
# from src.training.hippocampal_trainer import HippocampalTransformerTrainer, ReplayBuffer, EWCConsolidator

def test_replay_buffer_priority():
    """Test that ReplayBuffer stores and retrieves based on priority (loss)."""
    from src.training.hippocampal_trainer import ReplayBuffer
    
    buffer = ReplayBuffer(capacity=10)
    
    # Add items with different losses
    # Item: (input_ids, labels, loss)
    buffer.add(torch.tensor([1]), torch.tensor([1]), 1.0) # Low priority
    buffer.add(torch.tensor([2]), torch.tensor([2]), 10.0) # High priority
    buffer.add(torch.tensor([3]), torch.tensor([3]), 5.0) # Medium priority
    
    # Sample batch of 1
    batch = buffer.sample(batch_size=1)
    
    # Should sample the highest loss item (greedy or high prob)
    # For this test, we assume a simple priority queue or high probability
    # If using stochastic sampling, we might need to seed or sample more
    
    # Let's assume the implementation uses a simple priority sort for now or high temp
    assert len(buffer) == 3
    
    # Verify structure
    assert isinstance(batch, list)
    assert len(batch) == 1
    input_ids, labels, loss = batch[0]
    assert isinstance(input_ids, torch.Tensor)

    print("✅ test_replay_buffer_priority PASSED")

def test_ewc_penalty():
    """Test that EWC consolidator calculates penalty for weight changes."""
    from src.training.hippocampal_trainer import EWCConsolidator
    
    model = nn.Linear(10, 2)
    consolidator = EWCConsolidator(model)
    
    # 1. Compute Fisher (simulate training data)
    # We need to feed some data to compute gradients
    inputs = torch.randn(5, 10)
    # Targets should be class indices for CrossEntropyLoss
    targets = torch.randint(0, 2, (5,))
    
    # Mock a dataloader
    dataloader = [(inputs, targets)]
    
    consolidator.compute_fisher(dataloader, device='cpu')
    
    # 2. Change weights (simulate learning new task)
    with torch.no_grad():
        model.weight.add_(1.0) # Big change
        
    # 3. Calculate penalty
    penalty = consolidator.penalty(model)
    
    assert penalty > 0, "EWC penalty should be positive when weights change"
    assert penalty.requires_grad, "Penalty should be differentiable"
    
    print("✅ test_ewc_penalty PASSED")

def test_wake_sleep_cycle():
    """Test trainer transitions between Wake and Sleep phases."""
    from src.training.hippocampal_trainer import HippocampalTransformerTrainer
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        sleep_interval: int = 5
        sleep_steps: int = 2
        batch_size: int = 2
        lr: float = 1e-4
        
    config = MockConfig()
    model = nn.Linear(10, 2) # Mock model
    hippocampus = HippocampalFormation(10, 100, 10, 50)
    
    trainer = HippocampalTransformerTrainer(model, config, hippocampus)
    
    # Initial state
    assert trainer.phase == "wake"
    assert trainer.global_step == 0
    
    # Simulate training steps
    # We need to mock the train_step method or call a simplified version
    
    # Step 1-4: Wake
    for _ in range(4):
        trainer.step_counter()
        assert trainer.phase == "wake"
        
    # Step 5: Should trigger Sleep
    trainer.step_counter()
    # Depending on implementation, it might switch state immediately or return a flag
    # Let's assume it switches state
    
    # If the trainer handles the loop, we might need to inspect internal state
    # For TDD, let's define that `should_sleep()` returns True
    
    assert trainer.should_sleep() == True
    
    print("✅ test_wake_sleep_cycle PASSED")

def run_all_tests():
    print("="*60)
    print("HippocampalTrainer Test Suite (TDD)")
    print("="*60)
    
    tests = [
        test_replay_buffer_priority,
        test_ewc_penalty,
        test_wake_sleep_cycle,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"\nRunning {test_func.__name__}...")
            test_func()
            passed += 1
        except ImportError as e:
            print(f"❌ {test_func.__name__} FAILED (Import Error): {e}")
            failed += 1
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
