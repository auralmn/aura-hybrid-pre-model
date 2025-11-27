"""
The Limbic System: Memory and Emotion.

Integrates:
1. Hippocampus: Episodic memory retrieval.
2. Amygdala: Emotional salience and arousal regulation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from src.core.hippocampal import HippocampalFormation

class Amygdala(nn.Module):
    """
    Computes emotional valance and arousal from input features.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Maps input features to [Arousal, Valence]
        # Arousal: 0 (calm) -> 1 (excited/stressed)
        # Valence: -1 (negative) -> 1 (positive)
        self.sentiment_network = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh() 
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, float]:
        # x: [Batch, Seq, Dim] -> Pool -> [Batch, Dim]
        pooled = x.mean(dim=1)
        
        # [Batch, 2]
        sentiment = self.sentiment_network(pooled)
        
        # Average across batch for global brain state
        avg_sentiment = sentiment.mean(dim=0)
        
        # Map to 0-1 for arousal (using absolute value of first component as proxy magnitude)
        # and raw value for valence
        arousal = (avg_sentiment[0] + 1.0) / 2.0 # Norm to 0-1
        valence = avg_sentiment[1]
        
        return {'arousal': arousal.item(), 'valence': valence.item()}

class LimbicSystem(nn.Module):
    def __init__(self, d_model: int, hippocampus: HippocampalFormation):
        super().__init__()
        self.hippocampus = hippocampus
        self.amygdala = Amygdala(d_model)
        
        # Context projection
        self.memory_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            x: Input Embeddings
        Returns:
            Dict containing 'memory_context' (Tensor) and 'emotional_state' (Dict)
        """
        # 1. Amygdala Assessment
        emotional_state = self.amygdala(x)
        
        # 2. Hippocampal Retrieval
        # We query memory using the input
        # This is simplified; normally we'd use the specialized retrieve methods
        # For the forward pass tensor, we just return the current spatial context
        # In a full loop, we would use x to query retrieve_similar_memories
        
        # Placeholder for memory vector (using place cell activity)
        ctx = self.hippocampus.get_spatial_context()
        place_activity = ctx['place_cells'] # Tensor
        
        # Project place activity to model dimension
        # (Requires a projection matrix if dims mismatch, simplified here)
        # Assuming we want to inject memory bias:
        
        # For this implementation, we'll return the emotional state 
        # and a memory context placeholder
        
        return {
            'emotional_state': emotional_state,
            'memory_context': None # To be implemented with proper projection
        }