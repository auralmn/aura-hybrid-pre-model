"""
Test Suite for HippocampalTransformerTrainer (GPU-Native Compatible)
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.hippocampal import HippocampalFormation
from src.training.hippocampal_trainer import HippocampalTransformerTrainer, ReplayBuffer, EWCConsolidator

class MockModel(nn.Module):
    """Mock model mimicking HippocampalTransformer signature."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        
    def forward(self, input_ids, prosody=None, use_memory=True):
        # Return tuple (logits, place_cell_activity)
        logits = self.linear(input_ids.float())
        # Mock place activity (batch, seq, 1)
        place_activity = torch.zeros(input_ids.shape[0], 1, 1)
        return logits, place_activity

def test_replay_buffer_sampling():
    """Test ReplayBuffer stores and retrieves data correctly."""
    print("\nRunning test_replay_buffer_sampling...")
    
    buffer = ReplayBuffer(capacity=10)
    
    # Add items (input_ids, labels, loss)
    # Note: input_ids must be 2D [Batch, Seq] or 1D [Seq] depending on usage
    # The buffer expects [Batch, Seq] and splits it into [Seq] items.
    
    # Create a batch of 2 items
    inputs = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32) # [2, 2]
    labels = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)    # [2, 2]
    loss = 1.5
    
    buffer.add(inputs, labels, loss)
    
    # Should have 2 items (unpacked batch)
    assert len(buffer) == 2
    
    # Sample batch of 1
    batch = buffer.sample(batch_size=1)
    
    assert len(batch) == 1
    item = batch[0]
    # Item structure: (input_ids, labels, loss)
    assert len(item) == 3
    assert isinstance(item[0], torch.Tensor)
    assert isinstance(item[1], torch.Tensor)
    assert isinstance(item[2], float)
    
    print("✅ test_replay_buffer_sampling PASSED")

def test_ewc_penalty():
    """Test EWC penalty calculation."""
    print("\nRunning test_ewc_penalty...")
    
    model = MockModel()
    consolidator = EWCConsolidator(model)
    
    # 1. Compute Fisher
    inputs = torch.randn(5, 10) # [Batch, Dim]
    targets = torch.randint(0, 2, (5,))
    
    # Mock dataloader: list of (inputs, labels)
    dataloader = [(inputs, targets)]
    
    consolidator.compute_fisher(dataloader, device='cpu')
    
    # Check fisher populated
    assert len(consolidator.fisher) > 0
    
    # 2. Change weights
    with torch.no_grad():
        model.linear.weight.add_(1.0)
        
    # 3. Calculate penalty
    penalty = consolidator.penalty(model)
    
    assert penalty > 0.0
    assert penalty.requires_grad
    
    print("✅ test_ewc_penalty PASSED")

def test_wake_sleep_cycle():
    """Test Wake/Sleep phase transition."""
    print("\nRunning test_wake_sleep_cycle...")
    
    @dataclass
    class MockConfig:
        sleep_interval: int = 5
        batch_size: int = 2
        lr: float = 1e-4
        # Loss params
        label_smoothing: float = 0.0
        entropy_lambda: float = 0.0
        sparsity_lambda: float = 0.0
        target_sparsity: float = 0.03
        
    config = MockConfig()
    model = MockModel()
    # Minimal hippocampus
    hippo = HippocampalFormation(spatial_dimensions=2, n_place_cells=10, n_time_cells=5, n_grid_cells=5, device='cpu')
    
    trainer = HippocampalTransformerTrainer(model, config, hippo)
    
    # Initial state
    assert trainer.phase == "wake"
    assert trainer.global_step == 0
    
    # Steps 1-4: Wake
    for _ in range(4):
        trainer.step_counter()
        assert trainer.phase == "wake"
        
    # Step 5: Trigger Sleep
    trainer.step_counter()
    assert trainer.phase == "sleep"
    
    print("✅ test_wake_sleep_cycle PASSED")

def run_all_tests():
    print("="*60)
    print("HippocampalTrainer Test Suite")
    print("="*60)
    
    tests = [
        test_replay_buffer_sampling,
        test_ewc_penalty,
        test_wake_sleep_cycle,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)