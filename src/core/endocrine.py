"""
AURA-MOE Endocrine System.
Homeostatic control for neural routing and plasticity.

Integrates:
- Hypothalamus: Monitoring and control logic.
- Pituitary: Hormone release and decay.
"""

import numpy as np
import time
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

class HormoneType(Enum):
    CORTISOL = "cortisol"           # Stress response (Increases routing temperature)
    GROWTH_HORMONE = "growth_hormone" # Expert capacity (Increases top-k)
    THYROID = "thyroid"             # Metabolic rate (Modulates learning rate)
    INSULIN = "insulin"             # Energy regulation (Not used yet, maybe weight decay?)
    DOPAMINE = "dopamine"           # Reward (Reinforces expert bias)
    NOREPINEPHRINE = "norepinephrine" # Arousal (Gating sensitivity)

@dataclass
class Hormone:
    type: HormoneType
    concentration: float = 0.0
    half_life: float = 3600.0
    max_concentration: float = 10.0
    
    def update(self, dt: float, release: float):
        # Decay
        decay = np.exp(-dt / self.half_life)
        self.concentration *= decay
        # Release
        self.concentration += release
        self.concentration = min(self.concentration, self.max_concentration)
        return self.concentration

@dataclass
class SystemMetrics:
    energy_efficiency: float = 0.0
    expert_utilization: float = 0.0
    prediction_accuracy: float = 0.0
    stress_level: float = 0.0
    
    def update(self, accuracy: float, gate_diversity: float, energy: float):
        alpha = 0.9
        self.prediction_accuracy = alpha * self.prediction_accuracy + (1-alpha) * accuracy
        self.expert_utilization = alpha * self.expert_utilization + (1-alpha) * gate_diversity
        # Simple stress heuristic: High error + High energy = Stress
        current_stress = (1.0 - accuracy) * (1.0 + energy)
        self.stress_level = alpha * self.stress_level + (1-alpha) * current_stress

class EndocrineSystem:
    """
    Master controller for brain homeostasis.
    """
    def __init__(self):
        self.metrics = SystemMetrics()
        self.last_time = time.time()
        
        self.hormones = {
            h: Hormone(h) for h in HormoneType
        }
        
        # Targets
        self.target_accuracy = 0.95
        self.target_utilization = 0.8
        
    def step(self, metrics_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Update system state and release hormones.
        
        Args:
            metrics_dict: {
                'accuracy': 0.0-1.0 (or 1/perplexity),
                'gate_diversity': 0.0-1.0 (std of gate weights),
                'energy': 0.0-1.0 (normalized compute cost)
            }
        Returns:
            hormone_levels: Dict[str, float]
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # 1. Update Internal Metrics
        acc = metrics_dict.get('accuracy', 0.5)
        div = metrics_dict.get('gate_diversity', 0.5)
        eng = metrics_dict.get('energy', 0.1)
        self.metrics.update(acc, div, eng)
        
        # 2. Compute Hypothalamic Signals (Control Logic)
        releases = {h: 0.0 for h in HormoneType}
        
        # Cortisol: Release on high stress (error)
        if self.metrics.stress_level > 0.5:
            releases[HormoneType.CORTISOL] = (self.metrics.stress_level - 0.5) * 2.0
            
        # Dopamine: Release on high accuracy (reward)
        if self.metrics.prediction_accuracy > 0.8:
            releases[HormoneType.DOPAMINE] = (self.metrics.prediction_accuracy - 0.8) * 2.0
            
        # Growth Hormone: Release if utilization is poor (need more capacity/diversity)
        if self.metrics.expert_utilization < 0.4:
            releases[HormoneType.GROWTH_HORMONE] = (0.4 - self.metrics.expert_utilization) * 2.0
            
        # Norepinephrine: Arousal based on immediate change? 
        # For now, link to stress baseline
        releases[HormoneType.NOREPINEPHRINE] = self.metrics.stress_level * 0.5

        # 3. Pituitary Release & Decay
        levels = {}
        for h_type, hormone in self.hormones.items():
            amount = releases.get(h_type, 0.0) * dt # Scale by time step? Or per step?
            # For training loops, dt might be tiny, so we treat 'release' as 'impulse per step'
            # ignoring real-time dt for simulation stability often works better.
            # Let's assume release is per-step magnitude.
            val = hormone.update(dt if dt < 10 else 1.0, amount * 0.1)
            levels[h_type.value] = val
            
        return levels