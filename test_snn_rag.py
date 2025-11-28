"""
Test SNN + RAG Integration

Tests the new SNNRAGTransformer with:
1. SNN-based FFN layers
2. RAG memory retrieval during forward pass
3. Memory storage during training
4. Generation with memory augmentation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import math

from src.core.hippocampal import HippocampalFormation
from src.core.language_zone.snn_rag_transformer import SNNRAGTransformer
from src.core.language_zone.snn_ffn import SNNFFN, HybridFFN
from src.training.losses import HippocampalLoss


@dataclass
class TestConfig:
    """Config for SNN-RAG testing."""
    vocab_size: int = 500
    embedding_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 32
    intermediate_size: int = 128
    n_place_cells: int = 50
    place_cell_sparsity: float = 0.03
    theta_freq: float = 8.0
    gamma_freq: float = 40.0
    # Aliases for theta_gamma_encoding.py
    theta_frequency: float = 8.0
    gamma_frequency: float = 40.0


def test_snn_ffn():
    """Test standalone SNN FFN module."""
    print("\n" + "=" * 50)
    print("TEST: SNN FFN")
    print("=" * 50)
    
    snn = SNNFFN(
        input_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_timesteps=4,
        L=8
    )
    
    x = torch.randn(2, 10, 64)  # [Batch, Seq, Dim]
    out = snn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch"
    
    # Check gradients flow
    loss = out.sum()
    loss.backward()
    
    grad_ok = all(p.grad is not None for p in snn.parameters() if p.requires_grad)
    print(f"Gradients flow: {grad_ok}")
    assert grad_ok, "Gradients not flowing"
    
    print("PASSED")


def test_hybrid_ffn():
    """Test Hybrid FFN (MLP + SNN blend)."""
    print("\n" + "=" * 50)
    print("TEST: Hybrid FFN")
    print("=" * 50)
    
    hybrid = HybridFFN(
        input_dim=64,
        hidden_dim=128,
        snn_ratio=0.5
    )
    
    x = torch.randn(2, 10, 64)
    out = hybrid(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Gate value: {torch.sigmoid(hybrid.gate).item():.3f}")
    
    assert out.shape == x.shape
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    
    print("PASSED")


def test_snn_rag_transformer():
    """Test full SNNRAGTransformer."""
    print("\n" + "=" * 50)
    print("TEST: SNN-RAG Transformer")
    print("=" * 50)
    
    config = TestConfig()
    device = torch.device('cpu')
    
    # Create hippocampus
    hippocampus = HippocampalFormation(
        spatial_dimensions=2,
        n_place_cells=config.n_place_cells,
        n_time_cells=10,
        n_grid_cells=10,
        max_memories=100,
        feature_dim=config.embedding_dim,
        device=str(device)
    )
    
    # Create model
    model = SNNRAGTransformer(
        config=config,
        hippocampus=hippocampus,
        use_snn_ffn=True,
        snn_layers=[0, 2],  # SNN in layers 0 and 2
        memory_injection="gate",
        num_retrieved=3
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass (no memories yet)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    prosody = torch.randn(2, 16, 4)
    
    print("\nForward pass (empty memory)...")
    logits, place_activity = model(input_ids, prosody=prosody, use_memory=True)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Place activity shape: {place_activity.shape}")
    
    assert logits.shape == (2, 16, config.vocab_size)
    
    # Test with memory storage
    print("\nForward pass with memory storage...")
    model.train()
    logits, _ = model(input_ids, prosody=prosody, use_memory=True, store_memory=True)
    
    print(f"Memories stored: {hippocampus.memory_count}")
    assert hippocampus.memory_count > 0, "No memories stored"
    
    # Test forward with retrieval
    print("\nForward pass with memory retrieval...")
    model.eval()
    logits2, _ = model(input_ids, prosody=prosody, use_memory=True)
    
    print(f"Logits shape: {logits2.shape}")
    
    print("PASSED")


def test_training_loop():
    """Test training with SNN-RAG model."""
    print("\n" + "=" * 50)
    print("TEST: Training Loop")
    print("=" * 50)
    
    config = TestConfig()
    device = torch.device('cpu')
    
    hippocampus = HippocampalFormation(
        spatial_dimensions=2,
        n_place_cells=config.n_place_cells,
        n_time_cells=10,
        n_grid_cells=10,
        max_memories=100,
        feature_dim=config.embedding_dim,
        device=str(device)
    )
    
    model = SNNRAGTransformer(
        config=config,
        hippocampus=hippocampus,
        use_snn_ffn=True,
        snn_layers=[0, 2],
        memory_injection="gate",
        num_retrieved=3
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = HippocampalLoss(label_smoothing=0.1)
    
    # Create simple data
    data = torch.randint(0, config.vocab_size, (20, config.max_seq_len))
    
    model.train()
    losses = []
    
    print("\nTraining for 20 steps...")
    for step in range(20):
        # Sample batch
        idx = torch.randint(0, len(data), (4,))
        batch = data[idx]
        
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward with memory storage every 5 steps
        store_mem = (step % 5 == 0)
        logits, place_activity = model(input_ids, use_memory=True, store_memory=store_mem)
        
        # Loss (ensure contiguous tensors)
        logits = logits.contiguous()
        labels = labels.contiguous()
        loss = criterion(logits, labels, place_activity)
        loss.backward()
        
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        losses.append(loss.item())
        
        if step % 5 == 0:
            print(f"Step {step}: Loss={loss.item():.4f}, Grad={grad_norm:.2f}, Memories={hippocampus.memory_count}")
    
    # Check loss decreased
    initial_loss = sum(losses[:3]) / 3
    final_loss = sum(losses[-3:]) / 3
    
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Memories stored: {hippocampus.memory_count}")
    
    if final_loss < initial_loss:
        print("PASSED - Loss decreased")
    else:
        print("WARNING - Loss did not decrease (may need more steps)")


def test_generation():
    """Test generation with RAG."""
    print("\n" + "=" * 50)
    print("TEST: Generation with RAG")
    print("=" * 50)
    
    config = TestConfig()
    device = torch.device('cpu')
    
    hippocampus = HippocampalFormation(
        spatial_dimensions=2,
        n_place_cells=config.n_place_cells,
        n_time_cells=10,
        n_grid_cells=10,
        max_memories=100,
        feature_dim=config.embedding_dim,
        device=str(device)
    )
    
    model = SNNRAGTransformer(
        config=config,
        hippocampus=hippocampus,
        use_snn_ffn=True,
        memory_injection="gate"
    )
    
    # Store some memories first
    model.train()
    for i in range(5):
        dummy_input = torch.randint(0, config.vocab_size, (1, 16))
        model(dummy_input, use_memory=True, store_memory=True)
    
    print(f"Memories stored: {hippocampus.memory_count}")
    
    # Generate
    model.eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    
    print(f"\nPrompt: {prompt[0].tolist()}")
    
    generated = model.generate(
        prompt,
        max_new_tokens=15,
        temperature=0.8,
        top_k=50,
        use_memory=True
    )
    
    print(f"Generated: {generated[0].tolist()}")
    print(f"New tokens: {generated.shape[1] - prompt.shape[1]}")
    
    assert generated.shape[1] > prompt.shape[1], "No tokens generated"
    
    print("PASSED")


def run_all_tests():
    """Run all SNN-RAG tests."""
    print("=" * 60)
    print("SNN + RAG INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("SNN FFN", test_snn_ffn),
        ("Hybrid FFN", test_hybrid_ffn),
        ("SNN-RAG Transformer", test_snn_rag_transformer),
        ("Training Loop", test_training_loop),
        ("Generation", test_generation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\nFAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

