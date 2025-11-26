# AURA-MOE Hypothalamic-Pituitary System
# Neuroendocrine control for liquid neural routing

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time

# ---------------------- Hormones (Control Signals) ----------------------

class HormoneType(Enum):
    """Different types of neuroendocrine control signals"""
    CORTISOL = "cortisol"           # Stress response, energy mobilization
    GROWTH_HORMONE = "growth_hormone"  # Expert capacity scaling
    THYROID = "thyroid"             # Metabolic rate, learning speed
    INSULIN = "insulin"             # Energy allocation, resource management
    DOPAMINE = "dopamine"           # Reward signaling, expert selection bias
    NOREPINEPHRINE = "norepinephrine"  # Attention, arousal, gain modulation

@dataclass
class Hormone:
    """Individual hormone with concentration and effects"""
    type: HormoneType
    concentration: float = 0.0
    half_life: float = 3600.0      # Seconds for hormone decay
    max_concentration: float = 10.0
    min_concentration: float = 0.0
    last_update: float = field(default_factory=time.time)
    
    def decay(self) -> None:
        """Natural hormone decay over time"""
        current_time = time.time()
        dt = current_time - self.last_update
        decay_factor = np.exp(-dt / self.half_life)
        self.concentration *= decay_factor
        self.concentration = max(self.concentration, self.min_concentration)
        self.last_update = current_time
    
    def release(self, amount: float) -> None:
        """Release hormone into system"""
        self.decay()  # Update first
        self.concentration += amount
        self.concentration = min(self.concentration, self.max_concentration)

# ---------------------- Hypothalamus (Homeostatic Controller) ----------------------

@dataclass
class SystemMetrics:
    """System health and performance metrics monitored by hypothalamus"""
    energy_efficiency: float = 0.0      # Energy per prediction accuracy
    expert_utilization: float = 0.0     # Average expert usage balance
    prediction_accuracy: float = 0.0    # Running average of prediction quality
    learning_rate: float = 0.0          # Rate of system adaptation
    stress_level: float = 0.0           # System load and pressure
    temperature: float = 1.0            # Routing temperature (exploration vs exploitation)
    
    def update_metrics(self, energy_j: float, accuracy: float, 
                      expert_gates: np.ndarray, learning_delta: float) -> None:
        """Update system metrics with latest measurements"""
        alpha = 0.95  # Smoothing factor
        
        # Energy efficiency (lower is better)
        if accuracy > 0:
            efficiency = energy_j / max(accuracy, 0.01)
            self.energy_efficiency = alpha * self.energy_efficiency + (1-alpha) * efficiency
        
        # Expert utilization balance (higher is better balance)
        if len(expert_gates) > 0:
            gate_std = np.std(expert_gates)
            utilization = 1.0 / (1.0 + gate_std)  # Inverse of standard deviation
            self.expert_utilization = alpha * self.expert_utilization + (1-alpha) * utilization
        
        # Prediction accuracy
        self.prediction_accuracy = alpha * self.prediction_accuracy + (1-alpha) * accuracy
        
        # Learning rate
        self.learning_rate = alpha * self.learning_rate + (1-alpha) * abs(learning_delta)
        
        # Stress level (based on multiple factors)
        stress = (1.0 - self.expert_utilization) + (1.0 - self.prediction_accuracy) * 0.5
        self.stress_level = alpha * self.stress_level + (1-alpha) * stress

class Hypothalamus:
    """Hypothalamic control center for AURA-MOE homeostasis"""
    
    def __init__(self, target_metrics: Optional[SystemMetrics] = None):
        # Set points for optimal system operation
        self.target_metrics = target_metrics or SystemMetrics(
            energy_efficiency=1e-10,    # Target: 0.1 nJ per unit accuracy
            expert_utilization=0.8,     # Target: 80% balanced utilization
            prediction_accuracy=0.85,   # Target: 85% accuracy
            learning_rate=0.1,          # Target: moderate adaptation
            stress_level=0.2,           # Target: low stress
            temperature=1.0             # Target: balanced exploration
        )
        
        self.current_metrics = SystemMetrics()
        
        # PID-like control parameters
        self.Kp = 0.5   # Proportional gain
        self.Ki = 0.1   # Integral gain  
        self.Kd = 0.2   # Derivative gain
        
        # Error accumulation for integral term
        self.error_integral: Dict[str, float] = {}
        self.previous_error: Dict[str, float] = {}
        
        # History for trend analysis
        self.metric_history: List[SystemMetrics] = []
        self.max_history = 100
    
    def monitor_system(self, energy_j: float, accuracy: float, 
                      expert_gates: np.ndarray, learning_delta: float) -> None:
        """Monitor system and update current metrics"""
        self.current_metrics.update_metrics(energy_j, accuracy, expert_gates, learning_delta)
        
        # Store history
        self.metric_history.append(SystemMetrics(**self.current_metrics.__dict__))
        if len(self.metric_history) > self.max_history:
            self.metric_history.pop(0)
    
    def compute_control_signals(self) -> Dict[HormoneType, float]:
        """Compute hormone release signals based on system state"""
        signals = {}
        
        # Cortisol: Stress response (high when system is stressed)
        stress_error = self.current_metrics.stress_level - self.target_metrics.stress_level
        cortisol_signal = max(0, stress_error * 2.0)  # Only release when stressed
        signals[HormoneType.CORTISOL] = cortisol_signal
        
        # Growth Hormone: Expert capacity (release when utilization is high)
        utilization_error = self.current_metrics.expert_utilization - self.target_metrics.expert_utilization
        gh_signal = max(0, utilization_error * 1.5)  # Promote growth when highly utilized
        signals[HormoneType.GROWTH_HORMONE] = gh_signal
        
        # Thyroid: Metabolic rate (adjust based on accuracy and learning rate)
        accuracy_error = self.target_metrics.prediction_accuracy - self.current_metrics.prediction_accuracy
        learning_error = self.target_metrics.learning_rate - self.current_metrics.learning_rate
        thyroid_signal = (accuracy_error + learning_error) * 0.8
        signals[HormoneType.THYROID] = max(0, thyroid_signal)
        
        # Insulin: Energy regulation (release when energy efficiency is poor)  
        energy_error = self.current_metrics.energy_efficiency - self.target_metrics.energy_efficiency
        insulin_signal = max(0, energy_error * 1e9)  # Scale for energy units
        signals[HormoneType.INSULIN] = insulin_signal
        
        # Dopamine: Reward signal (release when accuracy is high)
        dopamine_signal = max(0, (self.current_metrics.prediction_accuracy - 0.5) * 2.0)
        signals[HormoneType.DOPAMINE] = dopamine_signal
        
        # Norepinephrine: Arousal (release when stress is moderate)
        if 0.3 < self.current_metrics.stress_level < 0.7:
            norepi_signal = self.current_metrics.stress_level * 1.2
        else:
            norepi_signal = 0.1  # Baseline
        signals[HormoneType.NOREPINEPHRINE] = norepi_signal
        
        return signals
    
    def get_system_health(self) -> Dict[str, float]:
        """Get overall system health assessment"""
        metrics_dict = self.current_metrics.__dict__.copy()
        
        # Compute overall health score (0-1, higher is better)
        health_components = [
            1.0 - min(self.current_metrics.stress_level, 1.0),
            self.current_metrics.prediction_accuracy,
            self.current_metrics.expert_utilization,
            1.0 / (1.0 + self.current_metrics.energy_efficiency * 1e10)  # Energy efficiency
        ]
        
        overall_health = np.mean(health_components)
        metrics_dict['overall_health'] = overall_health
        
        return metrics_dict

# ---------------------- Pituitary (Master Gland) ----------------------

class Pituitary:
    """Pituitary gland - master endocrine controller for AURA-MOE"""
    
    def __init__(self):
        # Hormone storage and release
        self.hormones: Dict[HormoneType, Hormone] = {
            HormoneType.CORTISOL: Hormone(HormoneType.CORTISOL, half_life=7200.0),  # 2 hours
            HormoneType.GROWTH_HORMONE: Hormone(HormoneType.GROWTH_HORMONE, half_life=1800.0),  # 30 min
            HormoneType.THYROID: Hormone(HormoneType.THYROID, half_life=14400.0),  # 4 hours
            HormoneType.INSULIN: Hormone(HormoneType.INSULIN, half_life=300.0),   # 5 min
            HormoneType.DOPAMINE: Hormone(HormoneType.DOPAMINE, half_life=60.0),  # 1 min
            HormoneType.NOREPINEPHRINE: Hormone(HormoneType.NOREPINEPHRINE, half_life=120.0)  # 2 min
        }
        
        # Sensitivity to hypothalamic signals
        self.sensitivity = {
            HormoneType.CORTISOL: 0.8,
            HormoneType.GROWTH_HORMONE: 0.6,
            HormoneType.THYROID: 0.7,
            HormoneType.INSULIN: 0.9,
            HormoneType.DOPAMINE: 1.0,
            HormoneType.NOREPINEPHRINE: 0.8
        }
        
        # Release history for monitoring
        self.release_history: List[Dict[HormoneType, float]] = []
        self.max_history = 1000
    
    def receive_hypothalamic_signals(self, signals: Dict[HormoneType, float]) -> None:
        """Receive control signals from hypothalamus and release hormones"""
        release_amounts = {}
        
        for hormone_type, signal_strength in signals.items():
            if hormone_type in self.hormones:
                # Apply sensitivity and release hormone
                sensitivity = self.sensitivity[hormone_type]
                release_amount = signal_strength * sensitivity
                
                self.hormones[hormone_type].release(release_amount)
                release_amounts[hormone_type] = release_amount
        
        # Store release history
        self.release_history.append(release_amounts)
        if len(self.release_history) > self.max_history:
            self.release_history.pop(0)
    
    def get_hormone_levels(self) -> Dict[HormoneType, float]:
        """Get current circulating hormone levels"""
        levels = {}
        for hormone_type, hormone in self.hormones.items():
            hormone.decay()  # Update concentration
            levels[hormone_type] = hormone.concentration
        return levels
    
    def apply_hormonal_effects(self, aura_system) -> Dict[str, Any]:
        """Apply hormonal modulation to AURA system - delegate to Network adapter"""
        levels = self.get_hormone_levels()
        effects = {}
        if hasattr(aura_system, "apply_endocrine_modulations"):
            effects = aura_system.apply_endocrine_modulations(levels)
        return {"levels": levels, "effects": effects}

print("🧠 AURA-MOE Hypothalamic-Pituitary System Loaded!")
print("=" * 50)
print("Components:")
print("• Hypothalamus: Homeostatic monitoring & control")
print("• Pituitary: Master gland hormone release") 
print("• 6 Hormone types: Cortisol, GH, Thyroid, Insulin, Dopamine, Norepinephrine")
print("• System metrics: Energy, utilization, accuracy, stress")
print("• Biological feedback loops for adaptive control")