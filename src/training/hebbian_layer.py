import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class OjaStepOut:
    y: torch.Tensor
    residual_ema: float
    grew: bool

class OjaLayer(nn.Module):
    """
    GPU-Native Unsupervised Hebbian learning layer using Oja's Rule.
    
    Optimizations:
    - Pure PyTorch implementation (avoids CPU/GPU sync)
    - JIT-friendly operations
    - In-place updates for memory efficiency
    """
    def __init__(self, n_components: int, input_dim: int, eta: float = 0.01, 
                 alpha: float = 0.99, threshold: float = 2.0, max_components: int = 2048):
        super().__init__()
        self.input_dim = input_dim
        self.eta = eta
        self.alpha = alpha
        self.threshold = threshold
        self.max_components = max_components
        
        # We manage K manually since it grows
        self.register_buffer('K', torch.tensor(n_components, dtype=torch.int32))
        
        # Initialize weights [Input, Max_Components]
        # We pre-allocate max_components to avoid reallocation resizing on GPU
        self.register_buffer('W_memory', torch.randn(input_dim, max_components) * 0.02)
        
        # Normalize the active components
        with torch.no_grad():
            self.W_memory[:, :n_components] = torch.nn.functional.normalize(
                self.W_memory[:, :n_components], dim=0
            )
            
        self.register_buffer('residual_ema', torch.tensor(0.0))
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))

    @property
    def W(self):
        """View of currently active weights"""
        return self.W_memory[:, :self.K]

    def forward(self, xw: torch.Tensor) -> torch.Tensor:
        """Forward pass (Projection) without learning"""
        return torch.matmul(xw, self.W)

    def step(self, xw: torch.Tensor) -> OjaStepOut:
        """
        Perform one Hebbian learning step on GPU.
        Args:
            xw: Whitened input vector [Batch, Input_Dim] or [Input_Dim]
        """
        # Ensure input is 2D [Batch, Dim]
        if xw.dim() == 1:
            xw = xw.unsqueeze(0)
            
        batch_size = xw.shape[0]
        
        # 1. Compute projection: y = x @ W -> [Batch, K]
        y = torch.matmul(xw, self.W)
        
        # 2. Reconstruct: x_hat = y @ W.T -> [Batch, Input]
        x_hat = torch.matmul(y, self.W.t())
        
        # 3. Compute residual: r = x - x_hat
        residual = xw - x_hat
        
        # Mean residual norm across batch
        norm_residual = torch.norm(residual, dim=1).mean()
        
        # 4. Update EMA
        if self.update_count == 0:
            self.residual_ema = norm_residual
        else:
            self.residual_ema = self.alpha * self.residual_ema + (1 - self.alpha) * norm_residual
            
        # 5. Oja's Rule Update (Vectorized for Batch)
        # dW = eta * (x^T @ y - W * (y^T @ y))
        # Or simplified: dW = eta * r^T @ y
        # We average updates across the batch
        dW = self.eta * torch.matmul(residual.t(), y) / batch_size
        
        self.W_memory[:, :self.K] += dW
        
        # Re-normalize to prevent drift (Oja's rule approximates this, but explicit is more stable)
        self.W_memory[:, :self.K] = torch.nn.functional.normalize(self.W_memory[:, :self.K], dim=0)
        
        # 6. Neurogenesis check
        grew = False
        current_k = self.K.item()
        if self.residual_ema > self.threshold and current_k < self.max_components:
            grew = self._grow_component(residual.mean(dim=0))
            
        self.update_count += 1
        
        return OjaStepOut(y=y, residual_ema=self.residual_ema.item(), grew=grew)

    def _grow_component(self, avg_residual: torch.Tensor) -> bool:
        current_k = self.K.item()
        if current_k >= self.max_components:
            return False
            
        # Normalize residual for new neuron weights
        new_w = torch.nn.functional.normalize(avg_residual, dim=0)
        
        # Insert into pre-allocated memory
        self.W_memory[:, current_k] = new_w
        self.K += 1
        
        logger.info(f"🧬 GPU NEUROGENESIS: Residual {self.residual_ema:.3f}. Grew to K={self.K}")
        self.residual_ema *= 0.5
        return True