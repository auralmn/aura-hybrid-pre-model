import threading
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class PoolStats:
    hits: int = 0
    misses: int = 0
    total_allocations: int = 0
    peak_usage_mb: float = 0.0

class ArrayPool:
    """
    Thread-safe memory pool for NumPy arrays to reduce allocation overhead.
    Critical for SNN/Hebbian learning where many temporary arrays are created.
    """
    def __init__(self, max_pool_size_mb: int = 512):
        self.max_pool_size = max_pool_size_mb * 1024 * 1024
        self.pools: Dict[Tuple[tuple, np.dtype], List[np.ndarray]] = {}
        self.current_usage = 0
        self.stats = PoolStats()
        self._lock = threading.Lock()

    def get_array(self, shape: tuple, dtype: np.dtype = np.float32, zero_fill: bool = True) -> np.ndarray:
        key = (shape, dtype)
        with self._lock:
            if key in self.pools and self.pools[key]:
                arr = self.pools[key].pop()
                self.stats.hits += 1
                if zero_fill:
                    arr.fill(0)
                return arr
            else:
                # Allocate new if pool empty
                arr = np.empty(shape, dtype=dtype)
                if zero_fill:
                    arr.fill(0)
                self.stats.misses += 1
                self.stats.total_allocations += 1
                return arr

    def return_array(self, arr: np.ndarray) -> None:
        if arr is None:
            return
        
        key = (arr.shape, arr.dtype)
        array_size = arr.nbytes
        
        with self._lock:
            # Only pool if we have space
            if self.current_usage + array_size <= self.max_pool_size:
                if key not in self.pools:
                    self.pools[key] = []
                
                # Reset/clear not strictly needed if get_array zero_fills, 
                # but good practice to not hold sensitive data
                # We don't zero-fill here to save time on return, 
                # rely on get_array to zero-fill if requested.
                
                self.pools[key].append(arr)
                self.current_usage += array_size
                
                # Update peak usage stats
                current_mb = self.current_usage / (1024 * 1024)
                if current_mb > self.stats.peak_usage_mb:
                    self.stats.peak_usage_mb = current_mb

    def clear(self):
        with self._lock:
            self.pools.clear()
            self.current_usage = 0

# Global instance for easy access
_global_pool = ArrayPool()

def get_pooled_array(shape: tuple, dtype: np.dtype = np.float32, zero_fill: bool = True) -> np.ndarray:
    return _global_pool.get_array(shape, dtype, zero_fill)

def return_pooled_array(arr: np.ndarray) -> None:
    _global_pool.return_array(arr)

