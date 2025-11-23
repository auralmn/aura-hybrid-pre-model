#!/usr/bin/env python3
"""
FastHashEmbedder: deterministic, training-free text embeddings via n-gram hashing.
"""

from typing import Iterable, List, Tuple
import torch


class FastHashEmbedder:
    def __init__(self, dim: int = 1024, min_n: int = 2, max_n: int = 5):
        self.dim = int(dim)
        self.min_n = int(min_n)
        self.max_n = int(max_n)

    @staticmethod
    def _fnv1a64(s: str) -> int:
        h = 0xcbf29ce484222325
        for ch in s:
            h ^= ord(ch)
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        return h

    def _ngrams(self, text: str) -> Iterable[str]:
        t = text.lower()
        L = len(t)
        for n in range(self.min_n, self.max_n + 1):
            if n > L:
                continue
            for i in range(L - n + 1):
                yield t[i:i+n]

    def encode(self, text: str) -> torch.Tensor:
        """Vectorized n-gram hashing using PyTorch operations."""
        # Fast path for empty text
        if not text:
            return torch.zeros(self.dim, dtype=torch.float32)

        # Convert to tensor (ASCII/UTF-8 bytes)
        b = text.encode('utf-8', errors='ignore')
        if not b:
             return torch.zeros(self.dim, dtype=torch.float32)
        
        t_bytes = torch.tensor(list(b), dtype=torch.long)
        v = torch.zeros(self.dim, dtype=torch.float32)
        
        # Using a prime multiplier per position
        multiplier = 0x100000001b3
        total_grams = 0
        
        for n in range(self.min_n, self.max_n + 1):
            if t_bytes.size(0) < n:
                continue
                
            # Unfold to get sliding windows: [num_grams, n]
            windows = t_bytes.unfold(0, n, 1)
            
            # Precomputed powers for this n
            device = t_bytes.device
            powers = torch.tensor([31**i for i in range(n)], dtype=torch.long, device=device)
            
            # Hash values: [num_grams]
            gram_hashes = (windows * powers).sum(dim=1)
            
            # Map to indices
            indices = gram_hashes.abs() % self.dim
            
            # Using bincount for histogram is faster
            counts = torch.bincount(indices, minlength=self.dim).float()
            v += counts
            total_grams += windows.size(0)

        if total_grams > 0:
            v = v / (v.norm(p=2).clamp(min=1e-6))
            
        return v

    def encode_with_indices(self, text: str) -> Tuple[torch.Tensor, List[int]]:
        """
        Returns both the embedding vector and the sequence of hashed indices (tokens).
        Used for STDP learning which requires temporal token sequence.
        
        Note: This uses the CPU-based iterative implementation for index collection,
        as getting ordered indices from vectorized operations is complex.
        """
        v = torch.zeros(self.dim, dtype=torch.float32)
        indices = []
        count = 0
        for gram in self._ngrams(text):
            idx = self._fnv1a64(gram) % self.dim
            indices.append(idx)
            v[idx] += 1.0
            count += 1
        if count > 0:
            v = v / v.norm(p=2).clamp(min=1e-6)
        return v, indices

    def batch_encode(self, texts: List[str]) -> torch.Tensor:
        return torch.stack([self.encode(t) for t in texts], dim=0)
