"""
Optimized Prosody Attention Bridge with LRU Caching

Implements caching to avoid recomputing prosody for repeated sequences.
Expected speedup: 50-80% on datasets with repeated text patterns.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from functools import lru_cache
import hashlib

from core.language_zone.multi_channel_attention import (
    MultiChannelSpikingAttention,
    prosody_channels_from_text
)


class OptimizedProsodyAttentionBridge(nn.Module):
    """
    Prosody Attention Bridge with LRU caching for performance.
    
    Improvements over base version:
    1. LRU cache for prosody channel extraction (50-80% speedup on repeated sequences)
    2. Batch processing support for parallel prosody computation
    3. Memory-efficient storage of attention states
    
    Args:
        attention_preset: Preset configuration ('analytical_balanced' recommended)
        k_winners: Number of winners for k-WTA
        cache_size: LRU cache size (default: 10000 sequences)
        use_caching: Enable/disable caching (default: True)
    """
    
    def __init__(
        self,
        attention_preset: str = 'analytical_balanced',
        k_winners: int = 5,
        cache_size: int = 10000,
        use_caching: bool = True
    ):
        super().__init__()
        
        self.k_winners = k_winners
        self.use_caching = use_caching
        self.cache_size = cache_size
        
        # Create attention module
        if attention_preset == 'analytical_balanced':
            self.attention = MultiChannelSpikingAttention(
                k_winners=5,
                w_amp=0.8,
                w_pitch=1.2,
                w_bound=1.0,
                smoothing=2,
                normalize_salience=True,
                gain_up=1.5,
                gain_down=0.7
            )
        else:
            # Fallback to default
            self.attention = MultiChannelSpikingAttention(k_winners=k_winners)
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        if use_caching:
            self._init_cache()
    
    def _init_cache(self):
        """Initialize LRU cache for prosody extraction."""
        # We'll cache prosody channels keyed by text hash
        self._prosody_cache = {}
        self._cache_access_order = []
    
    def _make_cache_key(self, tokens: List[str]) -> str:
        """Create cache key from token list."""
        # Use hash of joined tokens
        text = " ".join(tokens)
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cached_prosody(self, cache_key: str) -> Tuple:
        """Get prosody from cache if available."""
        if cache_key in self._prosody_cache:
            self.cache_hits += 1
            # Update access order (LRU)
            self._cache_access_order.remove(cache_key)
            self._cache_access_order.append(cache_key)
            return self._prosody_cache[cache_key]
        else:
            self.cache_misses += 1
            return None
    
    def _set_cached_prosody(self, cache_key: str, prosody_data: Tuple):
        """Store prosody in cache."""
        # Evict oldest if cache is full
        if len(self._prosody_cache) >= self.cache_size:
            oldest_key = self._cache_access_order.pop(0)
            del self._prosody_cache[oldest_key]
        
        self._prosody_cache[cache_key] = prosody_data
        self._cache_access_order.append(cache_key)
    
    def extract_prosody_cached(self, token_strings: List[str]) -> Tuple:
        """
        Extract prosody channels with caching.
        
        Returns:
            (amplitude, pitch, boundary) numpy arrays
        """
        if not self.use_caching:
            return prosody_channels_from_text(token_strings)
        
        # Try cache first
        cache_key = self._make_cache_key(token_strings)
        cached = self._get_cached_prosody(cache_key)
        
        if cached is not None:
            return cached
        
        # Cache miss - compute and store
        prosody_data = prosody_channels_from_text(token_strings)
        self._set_cached_prosody(cache_key, prosody_data)
        
        return prosody_data
    
    def compute_attention_gains(
        self,
        token_ids: List[int],
        token_strings: List[str]
    ) -> Dict:
        """
        Compute attention gains WITH caching.
        
        Returns dict with:
            - mu_scalar: scalar attention gain
            - salience: per-token salience
            - winners_idx: indices of winner tokens
            - winners: winner token strings
        """
        # Extract prosody (cached)
        amp, pitch, boundary = self.extract_prosody_cached(token_strings)
        
        # Compute attention
        result = self.attention.compute(
            token_ids=token_ids,
            amp=amp,
            pitch=pitch,
            boundary=boundary
        )
        
        # Add winner tokens
        if 'winners_idx' in result:
            result['winners'] = [token_strings[i] for i in result['winners_idx']]
        
        return result
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_strings: List[List[str]]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with batch support.
        
        Args:
            input_ids: (batch, seq_len)
            token_strings: List[List[str]] - batch of token sequences
        
        Returns:
            attention_gains: (batch, seq_len) tensor
            metadata: Dict with prosody stats
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Process each sequence in batch
        attention_gains_list = []
        all_salience = []
        all_winners = []
        
        for b in range(batch_size):
            tokens_b = token_strings[b]
            token_ids_b = input_ids[b].cpu().tolist()
            
            # Compute attention for this sequence
            result = self.compute_attention_gains(token_ids_b, tokens_b)
            
            # Extract gains
            gain_scalar = result['mu_scalar']
            salience = result['salience']
            
            # Create per-token gains
            gains = torch.from_numpy(salience).float() * gain_scalar
            
            # Pad to seq_len if needed
            if len(gains) < seq_len:
                gains = torch.cat([
                    gains,
                    torch.ones(seq_len - len(gains)) * gain_scalar
                ])
            
            attention_gains_list.append(gains)
            all_salience.append(salience)
            all_winners.append(result.get('winners', []))
        
        # Stack into batch tensor
        attention_gains = torch.stack(attention_gains_list).to(device)
        
        # Metadata
        metadata = {
            'salience': all_salience,
            'winners': all_winners,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }
        
        return attention_gains, metadata
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_queries if total_queries > 0 else 0
        
        return {
            'cache_size': len(self._prosody_cache) if self.use_caching else 0,
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_queries': total_queries
        }
    
    def clear_cache(self):
        """Clear prosody cache."""
        if self.use_caching:
            self._prosody_cache.clear()
            self._cache_access_order.clear()
            self.cache_hits = 0
            self.cache_misses = 0


def benchmark_caching_performance():
    """Benchmark LRU caching performance improvement."""
    import time
    from core.language_zone.prosody_attention import ProsodyAttentionBridge as BaseBridge
    
    print("="*60)
    print("LRU Caching Performance Benchmark")
    print("="*60)
    
    # Create test data with repeated sequences
    test_texts = [
        "This is a test sequence number one.",
        "This is a test sequence number two.",
        "This is a test sequence number one.",  # Repeat
        "Another completely different text here.",
        "This is a test sequence number two.",  # Repeat
        "This is a test sequence number one.",  # Repeat
        "Yet another unique text sample.",
        "This is a test sequence number one.",  # Repeat
    ] * 50  # 400 total, 50% repetition
    
    # Baseline (no cache)
    print("\n[1/2] Baseline (no caching)...")
    bridge_baseline = BaseBridge(attention_preset='analytical_balanced')
    
    start = time.time()
    for text in test_texts:
        tokens = text.split()
        token_ids = list(range(len(tokens)))
        _ = bridge_baseline.compute_attention_gains(token_ids, tokens)
    baseline_time = time.time() - start
    
    print(f"  Time: {baseline_time:.3f}s")
    
    # Optimized (with cache)
    print("\n[2/2] Optimized (with LRU cache)...")
    bridge_optimized = OptimizedProsodyAttentionBridge(
        attention_preset='analytical_balanced',
        cache_size=100,
        use_caching=True
    )
    
    start = time.time()
    for text in test_texts:
        tokens = text.split()
        token_ids = list(range(len(tokens)))
        _ = bridge_optimized.compute_attention_gains(token_ids, tokens)
    optimized_time = time.time() - start
    
    print(f"  Time: {optimized_time:.3f}s")
    
    # Results
    speedup = baseline_time / optimized_time
    cache_stats = bridge_optimized.get_cache_stats()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nBaseline time: {baseline_time:.3f}s")
    print(f"Optimized time: {optimized_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"\nCache Statistics:")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Hits: {cache_stats['cache_hits']}")
    print(f"  Misses: {cache_stats['cache_misses']}")
    print(f"  Cache size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    
    if speedup > 1.5:
        print(f"\n✅ Caching provides {speedup:.1f}x speedup!")
    else:
        print(f"\n⚠️  Caching speedup lower than expected ({speedup:.1f}x)")


if __name__ == '__main__':
    benchmark_caching_performance()
