"""
PyTorch Tensor Shim (Replaces legacy NumPy Memory Pool).

Modern PyTorch (via caching_allocator) manages memory better than manual pooling.
This module maintains the API compatibility but delegates to torch.empty/zeros
on the correct device to ensure full GPU pipeline compatibility.
"""

import torch
import numpy as np
from typing import Tuple, Union

# Global default device
_DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_pooled_array(shape: tuple, 
                     dtype: Union[np.dtype, torch.dtype] = torch.float32, 
                     zero_fill: bool = True,
                     device: torch.device = None) -> torch.Tensor:
    """
    Returns a PyTorch tensor. 
    Maintains API compatibility with legacy code expecting 'pooled arrays'.
    """
    target_device = device or _DEFAULT_DEVICE
    
    # Map numpy types to torch types if necessary
    if isinstance(dtype, type) and dtype.__module__ == 'numpy':
        if dtype == np.float32: dtype = torch.float32
        elif dtype == np.int64: dtype = torch.int64
        # Add others as needed
    
    if zero_fill:
        return torch.zeros(shape, dtype=dtype, device=target_device)
    else:
        return torch.empty(shape, dtype=dtype, device=target_device)

def return_pooled_array(arr: Union[np.ndarray, torch.Tensor]) -> None:
    """
    No-op for PyTorch tensors. 
    Let Python GC and PyTorch Caching Allocator handle it.
    """
    pass

class ArrayPool:
    """Legacy shim."""
    def get_array(self, *args, **kwargs):
        return get_pooled_array(*args, **kwargs)
        
    def return_array(self, *args, **kwargs):
        pass
        
    def clear(self):
        pass

# Singleton compatibility
_global_pool = ArrayPool()