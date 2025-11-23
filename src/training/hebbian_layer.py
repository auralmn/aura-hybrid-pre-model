import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Any
import logging

# Use optimized implementations
from training.memory_pool import get_pooled_array, return_pooled_array
from training.optimized_whitener import OptimizedWhitener

logger = logging.getLogger(__name__)

@dataclass
class OjaStepOut:
    y: np.ndarray
    residual_ema: float
    grew: bool

class OjaLayer:
    """
    Unsupervised Hebbian learning layer using Oja's Rule.
    Features:
    - Dynamic component growth (neurogenesis) based on residual error
    - Whitened input processing
    - Optimized for batch processing
    """
    def __init__(self, n_components: int, input_dim: int, eta: float = 0.01, 
                 alpha: float = 0.99, threshold: float = 2.0, max_components: int = 2048):
        self.K = n_components
        self.input_dim = input_dim
        self.eta = eta  # Learning rate
        self.alpha = alpha  # EMA decay for residual tracking
        self.threshold = threshold  # Growth threshold
        self.max_components = max_components
        
        # Initialize weights
        rng = np.random.default_rng(42)
        W_init = rng.standard_normal((input_dim, n_components)).astype(np.float32)
        
        if n_components <= input_dim:
            # Orthonormalize if possible
            q, _ = np.linalg.qr(W_init)
            self.W = q.astype(np.float32)
        else:
            # Overcomplete: just normalize columns
            norms = np.linalg.norm(W_init, axis=0, keepdims=True)
            self.W = (W_init / (norms + 1e-8)).astype(np.float32)
        
        # Tracking metrics
        self.residual_ema = 0.0
        self.age = 0
        self.update_count = 0

    def step(self, xw: np.ndarray) -> OjaStepOut:
        """
        Perform one Hebbian learning step.
        Args:
            xw: Whitened input vector [input_dim]
        """
        # 1. Compute projection: y = W^T * x
        # y shape: [K]
        y = self.W.T @ xw
        
        # 2. Reconstruct input: x_hat = W * y
        # x_hat shape: [input_dim]
        x_hat = self.W @ y
        
        # 3. Compute residual: r = x - x_hat
        # We use pooled array to save memory for the diff
        residual = xw - x_hat
        norm_residual = np.linalg.norm(residual)
        
        # 4. Update EMA of residual
        if self.update_count == 0:
            self.residual_ema = norm_residual
        else:
            self.residual_ema = self.alpha * self.residual_ema + (1 - self.alpha) * norm_residual
            
        # 5. Oja's Rule Update: dW = eta * (x*y^T - W*(y*y^T))
        # Simplified implementation: W += eta * y * (x - W*y)
        # Note: (x - W*y) is exactly the residual we computed!
        # So: dW = eta * outer(residual, y)
        
        dW = self.eta * np.outer(residual, y)
        self.W += dW
        
        # 6. Neurogenesis check
        grew = False
        if self.residual_ema > self.threshold and self.K < self.max_components:
            grew, new_k = self._grow_component(residual)
            if grew:
                # Expand y output for this step with 0 for new component
                y_new = np.zeros(new_k, dtype=np.float32)
                y_new[:len(y)] = y
                y = y_new
        
        self.update_count += 1
        return OjaStepOut(y=y, residual_ema=self.residual_ema, grew=grew)

    def _grow_component(self, residual: np.ndarray) -> Tuple[bool, int]:
        """
        Add a new component initialized to the current residual direction.
        This rapidly captures unexplained variance.
        """
        if self.K >= self.max_components:
            return False, self.K
            
        # Normalize residual for new weight
        norm = np.linalg.norm(residual)
        if norm < 1e-6:
            return False, self.K
            
        new_w = (residual / norm).astype(np.float32)
        
        # Expand weight matrix
        # New shape: [input_dim, K+1]
        self.W = np.column_stack((self.W, new_w))
        self.K += 1
        
        logger.info(f"🧬 OJA NEUROGENESIS: Residual {self.residual_ema:.3f} > {self.threshold}. Grew to K={self.K}")
        
        # Reset EMA to give new component time to settle
        self.residual_ema *= 0.5 
        return True, self.K

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Forward pass without learning"""
        return self.W.T @ x

    def state_dict(self) -> Dict[str, Any]:
        return {
            "W": self.W,
            "K": self.K,
            "residual_ema": self.residual_ema,
            "update_count": self.update_count
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.W = state["W"]
        self.K = state["K"]
        self.residual_ema = state.get("residual_ema", 0.0)
        self.update_count = state.get("update_count", 0)

