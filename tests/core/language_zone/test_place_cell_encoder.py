"""
Test Suite for PlaceCellSemanticEncoder

TDD: Write tests FIRST before implementation.
Tests sparse population coding for semantic embeddings using place cells.
Uses mock hippocampus to avoid import dependencies.
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


# Import real HippocampalFormation
from src.core.hippocampal import HippocampalFormation


def test_place_cell_encoder_shape():
    """Test output shapes are correct"""
    from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
    from dataclasses import dataclass
    
    # Create real hippocampus
    hippocampus = HippocampalFormation(
        spatial_dimensions=768,
        n_place_cells=2000,
        n_time_cells=100,
        n_grid_cells=500
    )
    
    @dataclass
    class MockConfig:
        vocab_size: int = 50257
        embedding_dim: int = 768
        n_place_cells: int = 2000
    
    config = MockConfig()
    encoder = PlaceCellSemanticEncoder(config, hippocampus)
    
    # Test input
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    semantic_embedding, place_cell_activity = encoder(input_ids)
    
    # Check shapes
    assert semantic_embedding.shape == (batch_size, seq_len, config.embedding_dim), \
        f"Expected semantic shape {(batch_size, seq_len, config.embedding_dim)}, got {semantic_embedding.shape}"
    
    assert place_cell_activity.shape == (batch_size, seq_len, config.n_place_cells), \
        f"Expected place cell shape {(batch_size, seq_len, config.n_place_cells)}, got {place_cell_activity.shape}"
    
    print("✅ test_place_cell_encoder_shape PASSED")


def test_sparse_activation():
    """Test that place cell activation is sparse (~3%)"""
    from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
    from dataclasses import dataclass
    
    hippocampus = HippocampalFormation(
        spatial_dimensions=768,
        n_place_cells=2000,
        n_time_cells=100,
        n_grid_cells=500
    )
    
    @dataclass
    class MockConfig:
        vocab_size: int = 50257
        embedding_dim: int = 768
        n_place_cells: int = 2000
    
    config = MockConfig()
    encoder = PlaceCellSemanticEncoder(config, hippocampus)
    
    # Test input
    input_ids = torch.randint(0, config.vocab_size, (4, 32))
    
    # Forward pass
    _, place_cell_activity = encoder(input_ids)
    
    # Check sparsity (should be ~3% active)
    active_cells = (place_cell_activity > 0).float().mean()
    
    # Should be between 1% and 10% (roughly 3% target)
    assert 0.01 < active_cells < 0.10, \
        f"Place cell sparsity should be ~3%, got {active_cells:.2%}"
    
    print(f"✅ test_sparse_activation PASSED (sparsity: {active_cells:.2%})")


def test_reconstruction_quality():
    """Test that semantic embedding can be reconstructed from place cells"""
    from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
    from dataclasses import dataclass
    
    hippocampus = HippocampalFormation(
        spatial_dimensions=768,
        n_place_cells=2000,
        n_time_cells=100,
        n_grid_cells=500
    )
    
    @dataclass
    class MockConfig:
        vocab_size: int = 50257
        embedding_dim: int = 768
        n_place_cells: int = 2000
    
    config = MockConfig()
    encoder = PlaceCellSemanticEncoder(config, hippocampus)
    
    # Test input
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    
    # Get original embeddings
    with torch.no_grad():
        original_embeds = encoder.token_embedding(input_ids)
    
    # Forward pass
    reconstructed_embeds, place_activity = encoder(input_ids)
    
    # Reconstruction should be similar to original (with residual connection)
    similarity = torch.nn.functional.cosine_similarity(
        original_embeds.reshape(-1, config.embedding_dim),
        reconstructed_embeds.reshape(-1, config.embedding_dim)
    ).mean()
    
    # Should have reasonable similarity (>0.5) due to residual connection
    assert similarity > 0.5, \
        f"Reconstructed embeddings should be similar to originals (sim: {similarity:.3f})"
    
    print(f"✅ test_reconstruction_quality PASSED (similarity: {similarity:.3f})")


def test_different_tokens_different_patterns():
    """Test that different tokens activate different place cell patterns"""
    from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
    from dataclasses import dataclass
    
    hippocampus = HippocampalFormation(
        spatial_dimensions=768,
        n_place_cells=2000,
        n_time_cells=100,
        n_grid_cells=500
    )
    
    @dataclass
    class MockConfig:
        vocab_size: int = 50257
        embedding_dim: int = 768
        n_place_cells: int = 2000
    
    config = MockConfig()
    encoder = PlaceCellSemanticEncoder(config, hippocampus)
    
    # Different tokens
    token_a = torch.tensor([[100]])
    token_b = torch.tensor([[200]])
    token_c = torch.tensor([[300]])
    
    # Get place cell patterns
    _, pattern_a = encoder(token_a)
    _, pattern_b = encoder(token_b)
    _, pattern_c = encoder(token_c)
    
    # Patterns should be different
    diff_ab = torch.norm(pattern_a - pattern_b).item()
    diff_bc = torch.norm(pattern_b - pattern_c).item()
    diff_ac = torch.norm(pattern_a - pattern_c).item()
    
    # All should have substantial differences
    assert diff_ab > 1.0, f"Tokens A and B should have different patterns (diff: {diff_ab})"
    assert diff_bc > 1.0, f"Tokens B and C should have different patterns (diff: {diff_bc})"
    assert diff_ac > 1.0, f"Tokens A and C should have different patterns (diff: {diff_ac})"
    
    print(f"✅ test_different_tokens_different_patterns PASSED (avg diff: {(diff_ab+diff_bc+diff_ac)/3:.2f})")


def test_same_token_same_pattern():
    """Test that same token produces consistent place cell pattern"""
    from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
    from dataclasses import dataclass
    
    hippocampus = HippocampalFormation(
        spatial_dimensions=768,
        n_place_cells=2000,
        n_time_cells=100,
        n_grid_cells=500
    )
    
    @dataclass
    class MockConfig:
        vocab_size: int = 50257
        embedding_dim: int = 768
        n_place_cells: int = 2000
    
    config = MockConfig()
    encoder = PlaceCellSemanticEncoder(config, hippocampus)
    
    # Same token, multiple times
    token = torch.tensor([[42, 42, 42, 42]])
    
    # Get place cell patterns
    _, patterns = encoder(token)
    
    # All patterns for same token should be identical
    pattern_0 = patterns[0, 0]
    pattern_1 = patterns[0, 1]
    pattern_2 = patterns[0, 2]
    
    diff_01 = torch.norm(pattern_0 - pattern_1).item()
    diff_12 = torch.norm(pattern_1 - pattern_2).item()
    
    assert diff_01 < 1e-5, f"Same token should produce same pattern (diff: {diff_01})"
    assert diff_12 < 1e-5, f"Same token should produce same pattern (diff: {diff_12})"
    
    print(f"✅ test_same_token_same_pattern PASSED")


def test_batch_consistency():
    """Test that batched computation is consistent"""
    from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
    from dataclasses import dataclass
    
    hippocampus = HippocampalFormation(
        spatial_dimensions=768,
        n_place_cells=2000,
        n_time_cells=100,
        n_grid_cells=500
    )
    
    @dataclass
    class MockConfig:
        vocab_size: int = 50257
        embedding_dim: int = 768
        n_place_cells: int = 2000
    
    config = MockConfig()
    encoder = PlaceCellSemanticEncoder(config, hippocampus)
    
    # Single sample
    token_single = torch.tensor([[100, 200, 300]])
    _, pattern_single = encoder(token_single)
    
    # Batched (same sequence repeated)
    token_batch = token_single.expand(4, -1)
    _, pattern_batch = encoder(token_batch)
    
    # All batches should match single
    for b in range(4):
        diff = torch.norm(pattern_single[0] - pattern_batch[b]).item()
        assert diff < 1e-5, f"Batch {b} should match single (diff: {diff})"
    
    print(f"✅ test_batch_consistency PASSED")


def test_gradient_flow():
    """Test that gradients flow through encoder"""
    from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
    from dataclasses import dataclass
    
    hippocampus = HippocampalFormation(
        spatial_dimensions=768,
        n_place_cells=2000,
        n_time_cells=100,
        n_grid_cells=500
    )
    
    @dataclass
    class MockConfig:
        vocab_size: int = 50257
        embedding_dim: int = 768
        n_place_cells: int = 2000
    
    config = MockConfig()
    encoder = PlaceCellSemanticEncoder(config, hippocampus)
    
    # Test input
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    
    # Forward + backward
    semantic_embedding, _ = encoder(input_ids)
    loss = semantic_embedding.sum()
    loss.backward()
    
    # Check gradients exist
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"Gradient for {name} should exist"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} should not contain NaN"
    
    print(f"✅ test_gradient_flow PASSED")


def test_topk_selection():
    """Test that top-k selection produces correct sparsity"""
    from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
    from dataclasses import dataclass
    
    hippocampus = HippocampalFormation(
        spatial_dimensions=768,
        n_place_cells=2000,
        n_time_cells=100,
        n_grid_cells=500
    )
    
    @dataclass
    class MockConfig:
        vocab_size: int = 50257
        embedding_dim: int = 768
        n_place_cells: int = 2000
    
    config = MockConfig()
    encoder = PlaceCellSemanticEncoder(config, hippocampus)
    
    # Test input
    input_ids = torch.randint(0, config.vocab_size, (1, 1))
    
    # Forward pass
    _, place_activity = encoder(input_ids)
    
    # Count active cells per token
    active_per_token = (place_activity[0, 0] > 0).sum().item()
    
    # Should be exactly k active cells (where k = n_place_cells * sparsity)
    expected_k = int(config.n_place_cells * 0.03)  # 3% sparsity
    
    # Allow small tolerance
    assert abs(active_per_token - expected_k) <= 5, \
        f"Should have ~{expected_k} active cells, got {active_per_token}"
    
    print(f"✅ test_topk_selection PASSED (active: {active_per_token}/{config.n_place_cells})")


def test_residual_connection():
    """Test that residual connection is present"""
    from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
    from dataclasses import dataclass
    
    hippocampus = HippocampalFormation(
        spatial_dimensions=768,
        n_place_cells=2000,
        n_time_cells=100,
        n_grid_cells=500
    )
    
    @dataclass
    class MockConfig:
        vocab_size: int = 50257
        embedding_dim: int = 768
        n_place_cells: int = 2000
    
    config = MockConfig()
    encoder = PlaceCellSemanticEncoder(config, hippocampus)
    
    # Test input
    input_ids = torch.tensor([[42]])
    
    # Get token embedding directly
    with torch.no_grad():
        token_embed = encoder.token_embedding(input_ids)
    
    # Get output from encoder
    semantic_embed, _ = encoder(input_ids)
    
    # Output should contain information from token embedding (due to residual connection)
    similarity = torch.nn.functional.cosine_similarity(
        token_embed.reshape(-1),
        semantic_embed.reshape(-1),
        dim=0
    ).item()
    
    # Should have high similarity (>0.7) due to residual
    assert similarity > 0.7, \
        f"Residual connection should preserve token embedding (sim: {similarity:.3f})"
    
    print(f"✅ test_residual_connection PASSED (similarity: {similarity:.3f})")


def run_all_tests():
    """Run all PlaceCellSemanticEncoder tests"""
    print("="*60)
    print("PlaceCellSemanticEncoder Test Suite (TDD)")
    print("="*60)
    
    tests = [
        test_place_cell_encoder_shape,
        test_sparse_activation,
        test_reconstruction_quality,
        test_different_tokens_different_patterns,
        test_same_token_same_pattern,
        test_batch_consistency,
        test_gradient_flow,
        test_topk_selection,
        test_residual_connection,
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
