"""
GPU-Native Specialist Registry.

Manages a collection of specialized experts (Specialists) that:
1. Run on GPU (nn.Module).
2. Learn via NLMS-like updates (NLMSExpertAdapter).
3. Maintain biological metadata (maturation, activity).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from src.core.experts import NLMSExpertAdapter
from src.base.neuron import MaturationStage, ActivityState

class Specialist(NLMSExpertAdapter):
    """
    A specialized expert neuron.
    Inherits GPU compute capabilities from NLMSExpertAdapter.
    Adds biological state tracking.
    """
    def __init__(self, neuron_id: str, n_features: int = 384, lr: float = 0.1):
        super().__init__(in_dim=n_features, lr=lr)
        self.id = neuron_id
        
        # Biological Metadata
        self.specialization = 'specialist'
        self.abilities = {'classification': 0.9}
        self.maturation = MaturationStage.PROGENITOR
        self.activity = ActivityState.RESTING
        
        # Parameters for adaptation (clamp used in custom update logic if needed)
        self.clamp_range = (0.0, 1.0)
        self.l2_lambda = 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Wrap parent forward to update activity state
        self.activity = ActivityState.FIRING
        return super().forward(x)

def create_specialist(neuron_id: str, n_features: int = 384) -> Specialist:
    """Factory function for creating a GPU-ready Specialist."""
    return Specialist(neuron_id=neuron_id, n_features=n_features)

class SpecialistRegistry(nn.Module):
    """
    Registry for managing Specialists.
    Inherits from nn.Module so all specialists are properly registered
    as sub-modules (for .to(device), state_dict, etc.).
    """
    def __init__(self, n_features: int = 384):
        super().__init__()
        self.n_features = n_features
        # Use ModuleDict to ensure specialists move to GPU with the registry
        self._specs = nn.ModuleDict()

    def get(self, name: str) -> Optional[Specialist]:
        if name in self._specs:
            return self._specs[name]
        return None

    def ensure(self, name: str) -> Specialist:
        if name not in self._specs:
            # Create and register
            s = create_specialist(name, n_features=self.n_features)
            # If registry is already on GPU, new module needs to move
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
            if device != 'cpu':
                s = s.to(device)
            self._specs[name] = s
        return self._specs[name]

    @property
    def all(self) -> Dict[str, Specialist]:
        return dict(self._specs.items())
    
    def forward(self, x: torch.Tensor):
        """
        Optional: Forward pass through all specialists (e.g. for ensemble).
        """
        return {name: spec(x) for name, spec in self._specs.items()}