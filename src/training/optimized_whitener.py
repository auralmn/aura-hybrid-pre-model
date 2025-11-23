import numpy as np
from typing import Dict, Any, Optional

class OptimizedWhitener:
    """
    Whitens input data using running mean and variance.
     optimized with in-place operations to reduce memory allocations.
    Essential for Hebbian learning stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6, momentum: float = 0.01):
        self.dim = dim
        self.eps = np.float32(eps)
        self.momentum = np.float32(momentum)
        
        # State parameters
        self.mu = np.zeros(dim, dtype=np.float32)
        self.var = np.ones(dim, dtype=np.float32)
        
        # Pre-allocated buffers for in-place operations
        self._temp_diff = np.zeros(dim, dtype=np.float32)
        self._temp_result = np.zeros(dim, dtype=np.float32)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply whitening transformation: (x - mu) / sqrt(var + eps)
        Updates running stats during transform.
        """
        # Ensure input is float32
        if x.dtype != np.float32:
            x = x.astype(np.float32)
            
        # Update running mean
        # mu = mu * (1 - momentum) + x * momentum
        self.mu *= (1.0 - self.momentum)
        self.mu += self.momentum * x
        
        # Calculate centered data (x - mu)
        # Use in-place subtract for memory efficiency
        np.subtract(x, self.mu, out=self._temp_diff)
        
        # Update running variance
        # var = var * (1 - momentum) + (x - mu)^2 * momentum
        np.multiply(self._temp_diff, self._temp_diff, out=self._temp_result)
        self.var *= (1.0 - self.momentum)
        self.var += self.momentum * self._temp_result
        
        # Apply whitening: diff / sqrt(var + eps)
        np.sqrt(self.var + self.eps, out=self._temp_result)
        np.divide(self._temp_diff, self._temp_result, out=self._temp_result)
        
        # Return copy of result (since _temp_result is reused)
        return self._temp_result.copy()
        
    def state_dict(self) -> Dict[str, Any]:
        return {
            "mu": self.mu.copy(),
            "var": self.var.copy(),
            "dim": self.dim,
            "momentum": self.momentum
        }
        
    def load_state_dict(self, state: Dict[str, Any]):
        if state["dim"] != self.dim:
            raise ValueError(f"Dimension mismatch: state {state['dim']} != model {self.dim}")
        self.mu = state["mu"].astype(np.float32)
        self.var = state["var"].astype(np.float32)
        self.momentum = state.get("momentum", self.momentum)

