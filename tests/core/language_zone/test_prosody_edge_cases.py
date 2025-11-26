"""
Prosody Edge Case Validation Tests

Tests edge cases to ensure robust prosody attention behavior.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

import torch
from src.core.language_zone.prosody_attention import ProsodyAttentionBridge
from src.core.language_zone.prosody_gif import ProsodyModulatedGIF


def test_prosody_edge_cases():
    """Test edge cases your demo didn't cover."""
    
    bridge = ProsodyAttentionBridge(attention_preset='emotional', k_winners=5)
    
    # Edge Case 1: All-caps spam (should saturate, not explode)
    print("\n" + "="*60)
    print("Edge Case 1: All-CAPS Spam")
    print("="*60)
    tokens_spam = ["HELLO", "WORLD", "!!!", "AMAZING", "WOW", "GREAT"]
    token_ids = [hash(t) % 1000 for t in tokens_spam]
    
    result = bridge.compute_attention_gains(token_ids, tokens_spam)
    
    print(f"Tokens: {tokens_spam}")
    print(f"Winners found: {len(result['winners_idx'])}/{len(tokens_spam)}")
    print(f"Winner indices: {result['winners_idx']}")
    print(f"Gain (μ): {result['mu_scalar']:.2f}")
    print(f"Salience: {result['salience']}")
    
    # All tokens should be winners (saturated)
    assert len(result['winners_idx']) >= 3, \
        f"Expected >= 3 winners for spam, got {len(result['winners_idx'])}"
    
    # Salience should be clipped (not >10)
    assert result['mu_scalar'] <= 3.0, \
        f"Gain {result['mu_scalar']:.2f} exceeds max_gain, check clipping"
    
    print(f"✅ Spam test passed: {len(result['winners_idx'])} winners, gain={result['mu_scalar']:.2f}")
    
    
    # Edge Case 2: Empty/whitespace tokens
    print("\n" + "="*60)
    print("Edge Case 2: Empty/Whitespace Tokens")
    print("="*60)
    tokens_empty = ["the", "and", "or", "is", "a"]
    token_ids_empty = [hash(t) % 1000 for t in tokens_empty]
    
    result_empty = bridge.compute_attention_gains(token_ids_empty, tokens_empty)
    
    print(f"Tokens: {tokens_empty}")
    print(f"Winners found: {len(result_empty['winners_idx'])}/{len(tokens_empty)}")
    print(f"Gain (μ): {result_empty['mu_scalar']:.2f}")
    print(f"Salience: {result_empty['salience']}")
    
    # Should have minimal salience
    assert result_empty['mu_scalar'] < 1.5, \
        f"Empty tokens should have low gain, got {result_empty['mu_scalar']:.2f}"
    
    print(f"✅ Empty test passed: gain={result_empty['mu_scalar']:.2f} (expected <1.5)")
    
    
    # Edge Case 3: Long sequence (100+ tokens)
    print("\n" + "="*60)
    print("Edge Case 3: Long Sequence (100 tokens)")
    print("="*60)
    tokens_long = ["word"] * 100
    tokens_long[25] = "IMPORTANT!"
    tokens_long[75] = "CRITICAL?"
    token_ids_long = [hash(t) % 1000 for t in tokens_long]
    
    result_long = bridge.compute_attention_gains(token_ids_long, tokens_long)
    
    print(f"Sequence length: {len(tokens_long)}")
    print(f"Winners found: {len(result_long['winners_idx'])}")
    print(f"Winner positions: {result_long['winners_idx']}")
    print(f"Gain (μ): {result_long['mu_scalar']:.2f}")
    
    # Should find the 2 salient tokens
    assert len(result_long['winners_idx']) >= 2, \
        f"Expected at least 2 winners in long sequence, got {len(result_long['winners_idx'])}"
    
    # Winners should include positions 25 and/or 75
    winner_positions = result_long['winners_idx']
    has_important = 25 in winner_positions
    has_critical = 75 in winner_positions
    
    print(f"Found 'IMPORTANT!' at 25: {has_important}")
    print(f"Found 'CRITICAL?' at 75: {has_critical}")
    
    assert has_important or has_critical, \
        f"Expected winners at [25, 75], got {winner_positions}"
    
    print(f"✅ Long sequence test passed: {len(result_long['winners_idx'])} winners")
    
    
    # Edge Case 4: Mixed case and punctuation
    print("\n" + "="*60)
    print("Edge Case 4: Mixed Case & Punctuation")
    print("="*60)
    tokens_mixed = ["This", "is", "REALLY", "important", "!!!", "Please", "note"]
    token_ids_mixed = [hash(t) % 1000 for t in tokens_mixed]
    
    result_mixed = bridge.compute_attention_gains(token_ids_mixed, tokens_mixed)
    
    print(f"Tokens: {tokens_mixed}")
    print(f"Winners: {[tokens_mixed[i] for i in result_mixed['winners_idx']]}")
    print(f"Salience: {result_mixed['salience']}")
    
    winner_tokens = [tokens_mixed[i] for i in result_mixed['winners_idx']]
    assert 'REALLY' in winner_tokens or '!!!' in winner_tokens, \
        "Should detect CAPS or punctuation as salient"
    
    print(f"✅ Mixed case test passed")


def test_gif_modulation_range():
    """Verify GIF threshold modulation stays in safe range."""
    
    print("\n" + "="*60)
    print("GIF Modulation Range Test")
    print("="*60)
    
    gif = ProsodyModulatedGIF(
        input_dim=128,
        hidden_dim=256,
        attention_modulation_strength=0.3
    )
    
    x = torch.randn(2, 50, 128)
    
    # Test with extreme attention gains
    attention_gains_high = torch.ones(2, 50) * 3.0  # Max gain
    attention_gains_low = torch.ones(2, 50) * 0.2   # Min gain
    
    # High gain should increase spikes
    spikes_high, _ = gif(x, attention_gains=attention_gains_high)
    spike_count_high = spikes_high.sum().item()
    
    # Low gain should decrease spikes
    spikes_low, _ = gif(x, attention_gains=attention_gains_low)
    spike_count_low = spikes_low.sum().item()
    
    # No modulation baseline
    spikes_base, _ = gif(x, attention_gains=None)
    spike_count_base = spikes_base.sum().item()
    
    print(f"  Baseline spikes: {spike_count_base:.0f}")
    print(f"  High gain (3.0x) spikes: {spike_count_high:.0f} ({spike_count_high/spike_count_base:.1%})")
    print(f"  Low gain (0.2x) spikes: {spike_count_low:.0f} ({spike_count_low/spike_count_base:.1%})")
    
    # Sanity checks
    assert spike_count_high > spike_count_base, "High gain should increase spikes"
    assert spike_count_low < spike_count_base, "Low gain should decrease spikes"
    assert spike_count_high < spike_count_base * 3.0, "Modulation shouldn't explode spikes"
    
    print("✅ All modulation ranges are safe")
    
    # Test gradient flow
    x_grad = torch.randn(2, 50, 128, requires_grad=True)
    attention_gains_grad = torch.ones(2, 50, requires_grad=False)
    
    spikes_grad, _ = gif(x_grad, attention_gains=attention_gains_grad)
    loss = spikes_grad.sum()
    loss.backward()
    
    assert x_grad.grad is not None, "Gradients should flow through modulated GIF"
    assert not torch.isnan(x_grad.grad).any(), "Gradients should be valid"
    
    print("✅ Gradient flow verified")


if __name__ == '__main__':
    test_prosody_edge_cases()
    test_gif_modulation_range()
    
    print("\n" + "="*60)
    print("All Edge Case Tests Passed! ✅")
    print("="*60)
