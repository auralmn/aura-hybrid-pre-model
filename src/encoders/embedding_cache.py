import os
import json
import torch
import hashlib
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    Caches token indices and embeddings for text files to speed up STDP and Hebbian learning.
    """
    def __init__(self, cache_dir: str = "cache/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, text: str) -> str:
        """Generate a unique cache file path based on text content hash."""
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{h}.pt")

    def save(self, text: str, embedding: torch.Tensor, indices: List[int]):
        """Save embedding and indices to cache."""
        path = self.get_cache_path(text)
        try:
            torch.save({
                'embedding': embedding,
                'indices': torch.tensor(indices, dtype=torch.long)
            }, path)
        except Exception as e:
            logger.warning(f"Failed to save cache to {path}: {e}")

    def load(self, text: str) -> Optional[Tuple[torch.Tensor, List[int]]]:
        """Load from cache if exists."""
        path = self.get_cache_path(text)
        if not os.path.exists(path):
            return None
        
        try:
            data = torch.load(path)
            return data['embedding'], data['indices'].tolist()
        except Exception:
            return None

    def clear(self):
        """Clear all cache files."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)

