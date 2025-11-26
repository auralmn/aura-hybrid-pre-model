# AURA liquid MOE Extended with Hypothalamic-Pituitary Control
# Full integration of neuroendocrine system

from typing import Optional, Any, Dict
import numpy as np
from typing import Optional
from aura.core.hippothalamus import HormoneType, Hypothalamus, Pituitary


class Endocrine:
    """AURA with integrated Hypothalamic-Pituitary Axis"""
    
    def __init__(self, base_aura_moe, enable_neuroendocrine: bool = True):
        # Core AURA-MOE system
        self.aura_moe = base_aura_moe
        
        # Neuroendocrine system
        self.enable_neuroendocrine = enable_neuroendocrine
        if enable_neuroendocrine:
            self.hypothalamus = Hypothalamus()
            self.pituitary = Pituitary()
        
        # Performance tracking
        self.prediction_history = []
        self.energy_history = []
        self.accuracy_history = []
        
        # System state
        self.last_prediction_error = 0.0
        self.system_initialized = False
    
    def forward(self, inputs, text: Optional[str] = None, y_true: Optional[float] = None) -> Dict[str, Any]:
        """Enhanced forward pass with neuroendocrine regulation"""
        
        # Standard AURA-MOE forward pass
        result = self.aura_moe.forward(inputs, text=text)
        
        if self.enable_neuroendocrine and self.system_initialized:
            # Apply current hormonal modulation
            hormonal_effects = self.pituitary.apply_hormonal_effects(self.aura_moe)
            result['hormonal_effects'] = hormonal_effects
            
            # Get current hormone levels
            result['hormone_levels'] = self.pituitary.get_hormone_levels()
            
            # Get system health
            result['system_health'] = self.hypothalamus.get_system_health()
        
        # Track performance if ground truth provided
        if y_true is not None:
            error = abs(result['y'] - y_true)
            accuracy = 1.0 / (1.0 + error)  # Convert error to accuracy
            
            self.prediction_history.append(result['y'])
            self.accuracy_history.append(accuracy)
            self.energy_history.append(result.get('energy_j', 0.0))
            
            self.last_prediction_error = error
            
            # Update neuroendocrine system
            if self.enable_neuroendocrine:
                self._update_neuroendocrine_system(result, accuracy)
        
        return result
    
    def _update_neuroendocrine_system(self, result: Dict[str, Any], accuracy: float) -> None:
        """Update hypothalamic-pituitary system based on performance"""
        
        # Extract metrics for hypothalamic monitoring
        energy_j = result.get('energy_j', 0.0)
        expert_gates = np.array([info['gate'] for info in result['per_expert'].values()])
        learning_delta = abs(self.last_prediction_error - np.mean(self.accuracy_history[-10:]) if len(self.accuracy_history) >= 10 else 0.1)
        
        # Monitor system state
        self.hypothalamus.monitor_system(energy_j, accuracy, expert_gates, learning_delta)
        
        # Generate control signals
        control_signals = self.hypothalamus.compute_control_signals()
        
        # Pituitary releases hormones
        self.pituitary.receive_hypothalamic_signals(control_signals)
        
        # Mark system as initialized
        self.system_initialized = True
    
    async def learn_with_endocrine_regulation(self, x: np.ndarray, y_true: float, 
                                             text: Optional[str] = None) -> Dict[str, Any]:
        """Learning with full neuroendocrine feedback"""
        
        # Perform learning
        result = await self.aura_moe.learn(x, y_true, text=text)
        
        if self.enable_neuroendocrine:
            # Update endocrine system based on learning outcome
            accuracy = 1.0 / (1.0 + abs(result['y'] - y_true))
            self._update_neuroendocrine_system(result, accuracy)
            
            # Add endocrine information to result
            result['hormone_levels'] = self.pituitary.get_hormone_levels()
            result['system_health'] = self.hypothalamus.get_system_health()
        
        return result
    
    def get_endocrine_status(self) -> Dict[str, Any]:
        """Get complete endocrine system status"""
        if not self.enable_neuroendocrine:
            return {"neuroendocrine_enabled": False}
        
        return {
            "neuroendocrine_enabled": True,
            "hormone_levels": self.pituitary.get_hormone_levels(),
            "system_health": self.hypothalamus.get_system_health(),
            "current_metrics": self.hypothalamus.current_metrics.__dict__,
            "target_metrics": self.hypothalamus.target_metrics.__dict__,
            "recent_hormone_releases": self.pituitary.release_history[-5:] if self.pituitary.release_history else []
        }
    
    def reset_endocrine_system(self) -> None:
        """Reset the endocrine system to baseline"""
        if self.enable_neuroendocrine:
            self.hypothalamus = Hypothalamus()
            self.pituitary = Pituitary()
            self.system_initialized = False
            print("🔄 Endocrine system reset to baseline")

# ---------------------- Usage Example ----------------------

# Simulate an AURA-MOE system (mock for demonstration)
class MockAURAMOE:
    """Mock AURA-MOE for demonstration"""
    def __init__(self):
        self.gating = type('obj', (object,), {
            'temperature': 1.0,
            'bias_lr': 0.01
        })()
        self.energy = type('obj', (object,), {
            'e_mac_j': 3e-12
        })()
    
    def forward(self, inputs, text=None):
        # Simulate prediction with some randomness
        prediction = np.random.normal(0.8, 0.2)  # Target around 0.8
        energy = np.random.uniform(1e-11, 5e-11)  # Energy consumption
        
        return {
            'y': prediction,
            'energy_j': energy,
            'per_expert': {
                'expert1': {'gate': 0.6, 'pred': 0.7},
                'expert2': {'gate': 0.4, 'pred': 0.9}
            }
        }
    
    async def learn(self, x, y_true, text=None):
        result = self.forward(x, text)
        # Simulate learning improving accuracy over time
        result['y'] = result['y'] * 0.9 + y_true * 0.1  # Move toward target
        return result

print("\n" + "=" * 60)
print("AURA-MOE + HYPOTHALAMIC-PITUITARY DEMONSTRATION")
print("=" * 60)

# Create enhanced system
base_system = MockAURAMOE()
enhanced_system = Endocrine(base_system, enable_neuroendocrine=True)

print("\n🧪 Running simulation with neuroendocrine feedback...")
print("-" * 50)

# Simulate several predictions with learning
np.random.seed(42)  # For reproducible results
targets = [0.8, 0.75, 0.9, 0.85, 0.7, 0.95, 0.8]

for i, target in enumerate(targets):
    x = np.random.randn(10)  # Mock input
    
    # Forward pass with ground truth
    result = enhanced_system.forward(x, text=f"query_{i}", y_true=target)
    
    print(f"\nStep {i+1}: Target={target:.2f}, Prediction={result['y']:.3f}")
    print(f"  Energy: {result['energy_j']:.2e} J")
    
    if 'hormone_levels' in result:
        hormones = result['hormone_levels']
        print(f"  Hormones: Cortisol={hormones[HormoneType.CORTISOL]:.3f}, "
              f"Dopamine={hormones[HormoneType.DOPAMINE]:.3f}")
        
        health = result['system_health']
        print(f"  Health: Overall={health['overall_health']:.3f}, "
              f"Stress={health['stress_level']:.3f}")

print(f"\n📊 Final System Status:")
final_status = enhanced_system.get_endocrine_status()
print(f"System Health: {final_status['system_health']['overall_health']:.3f}")
print(f"Current Stress: {final_status['system_health']['stress_level']:.3f}")
print(f"Prediction Accuracy: {final_status['system_health']['prediction_accuracy']:.3f}")