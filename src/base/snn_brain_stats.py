#!/usr/bin/env python3
"""
Enhanced brain_stats.py with neuromorphic monitoring capabilities
Extends your existing BrainStats with spiking network metrics
"""

from dataclasses import dataclass
import json
import torch
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np

@dataclass
class BrainStats:
    """Enhanced brain statistics with neuromorphic metrics"""
    num_neurons: int = 0
    num_layers: int = 0
    num_zones: int = 0
    num_experts: int = 0
    num_memories: float = 0
    num_contexts: int = 0
    num_parameters: int = 0
    
    # Enhanced neuromorphic metrics
    total_spikes: int = 0
    avg_firing_rate: float = 0.0
    zone_firing_rates: Dict[str, float] = None
    neuron_type_distribution: Dict[str, int] = None
    surrogate_slope_stats: Dict[str, float] = None
    membrane_potential_stats: Dict[str, float] = None
    zone_health_status: Dict[str, str] = None
    training_stability: str = "unknown"
    gradient_flow_health: Dict[str, float] = None
    
    # Temporal tracking
    firing_rate_history: List[float] = None
    loss_history: List[float] = None
    stability_history: List[str] = None
    
    def __post_init__(self):
        """Initialize complex fields"""
        if self.zone_firing_rates is None:
            self.zone_firing_rates = {}
        if self.neuron_type_distribution is None:
            self.neuron_type_distribution = {}
        if self.surrogate_slope_stats is None:
            self.surrogate_slope_stats = {}
        if self.membrane_potential_stats is None:
            self.membrane_potential_stats = {}
        if self.zone_health_status is None:
            self.zone_health_status = {}
        if self.gradient_flow_health is None:
            self.gradient_flow_health = {}
        if self.firing_rate_history is None:
            self.firing_rate_history = []
        if self.loss_history is None:
            self.loss_history = []
        if self.stability_history is None:
            self.stability_history = []

    def __repr__(self) -> str:
        return self.get_stats_string()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics dictionary"""
        return {
            # Original stats
            'num_neurons': self.num_neurons,
            'num_layers': self.num_layers,
            'num_zones': self.num_zones,
            'num_experts': self.num_experts,
            'num_memories': self.num_memories,
            'num_contexts': self.num_contexts,
            'num_parameters': self.num_parameters,
            
            # Neuromorphic stats
            'total_spikes': self.total_spikes,
            'avg_firing_rate': self.avg_firing_rate,
            'zone_firing_rates': dict(self.zone_firing_rates),
            'neuron_type_distribution': dict(self.neuron_type_distribution),
            'surrogate_slope_stats': dict(self.surrogate_slope_stats),
            'membrane_potential_stats': dict(self.membrane_potential_stats),
            'zone_health_status': dict(self.zone_health_status),
            'training_stability': self.training_stability,
            'gradient_flow_health': dict(self.gradient_flow_health),
            
            # Temporal data (last 10 entries to avoid huge files)
            'recent_firing_rate_history': self.firing_rate_history[-10:] if self.firing_rate_history else [],
            'recent_loss_history': self.loss_history[-10:] if self.loss_history else [],
            'recent_stability_history': self.stability_history[-10:] if self.stability_history else [],
        }

    def get_stats_string(self) -> str:
        """Get formatted statistics string"""
        return json.dumps(self.get_stats(), indent=2)

    def save_stats(self, filename: str):
        """Save statistics to file"""
        with open(filename, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)

    def load_stats(self, filename: str):
        """Load statistics from file"""
        with open(filename, 'r') as f:
            stats = json.load(f)
            
            # Load original stats
            self.num_neurons = stats.get('num_neurons', 0)
            self.num_layers = stats.get('num_layers', 0)
            self.num_zones = stats.get('num_zones', 0)
            self.num_experts = stats.get('num_experts', 0)
            self.num_memories = stats.get('num_memories', 0)
            self.num_contexts = stats.get('num_contexts', 0)
            self.num_parameters = stats.get('num_parameters', 0)
            
            # Load neuromorphic stats
            self.total_spikes = stats.get('total_spikes', 0)
            self.avg_firing_rate = stats.get('avg_firing_rate', 0.0)
            self.zone_firing_rates = stats.get('zone_firing_rates', {})
            self.neuron_type_distribution = stats.get('neuron_type_distribution', {})
            self.surrogate_slope_stats = stats.get('surrogate_slope_stats', {})
            self.membrane_potential_stats = stats.get('membrane_potential_stats', {})
            self.zone_health_status = stats.get('zone_health_status', {})
            self.training_stability = stats.get('training_stability', 'unknown')
            self.gradient_flow_health = stats.get('gradient_flow_health', {})
            
            # Load temporal data
            self.firing_rate_history = stats.get('recent_firing_rate_history', [])
            self.loss_history = stats.get('recent_loss_history', [])
            self.stability_history = stats.get('recent_stability_history', [])

    def update_from_zone_activity(self, zone_activities: Dict[str, Dict[str, Any]]):
        """Update stats from brain zone activity data"""
        
        total_firing_rate = 0.0
        total_neurons_active = 0
        zone_health = {}
        neuron_types = defaultdict(int)
        
        for zone_name, zone_activity in zone_activities.items():
            if 'neuron_metrics' in zone_activity:
                zone_firing_rates = []
                
                for neuron_type, metrics in zone_activity['neuron_metrics'].items():
                    firing_rate = metrics.get('firing_rate', 0.0)
                    zone_firing_rates.append(firing_rate)
                    
                    # Count neuron types
                    neuron_types[neuron_type] += 1
                    
                    # Determine health status
                    if firing_rate < 0.001:
                        zone_health[f"{zone_name}_{neuron_type}"] = "silent"
                    elif firing_rate > 0.8:
                        zone_health[f"{zone_name}_{neuron_type}"] = "hyperactive"
                    else:
                        zone_health[f"{zone_name}_{neuron_type}"] = "healthy"
                
                # Calculate zone average
                if zone_firing_rates:
                    zone_avg = sum(zone_firing_rates) / len(zone_firing_rates)
                    self.zone_firing_rates[zone_name] = zone_avg
                    total_firing_rate += zone_avg
                    total_neurons_active += 1
            elif 'neuron_type_stats' in zone_activity:
                # Fallback to aggregated per-type stats
                type_stats = zone_activity['neuron_type_stats'] or {}
                rates = []
                for neuron_type, stats in type_stats.items():
                    fr = float(stats.get('mean_firing_rate', 0.0))
                    rates.append(fr)
                    neuron_types[neuron_type] += int(stats.get('count', 0) > 0)
                    if fr < 0.001:
                        zone_health[f"{zone_name}_{neuron_type}"] = "silent"
                    elif fr > 0.8:
                        zone_health[f"{zone_name}_{neuron_type}"] = "hyperactive"
                    else:
                        zone_health[f"{zone_name}_{neuron_type}"] = "healthy"
                if rates:
                    zone_avg = sum(rates) / len(rates)
                    self.zone_firing_rates[zone_name] = zone_avg
                    total_firing_rate += zone_avg
                    total_neurons_active += 1
            elif 'avg_firing_rate' in zone_activity:
                # Minimal fallback if only aggregate provided
                zone_avg = float(zone_activity.get('avg_firing_rate', 0.0))
                self.zone_firing_rates[zone_name] = zone_avg
                total_firing_rate += zone_avg
                total_neurons_active += 1
        
        # Update overall statistics
        if total_neurons_active > 0:
            self.avg_firing_rate = total_firing_rate / total_neurons_active
            self.firing_rate_history.append(self.avg_firing_rate)
            
            # Keep only last 1000 entries
            if len(self.firing_rate_history) > 1000:
                self.firing_rate_history = self.firing_rate_history[-1000:]
        
        self.zone_health_status.update(zone_health)
        self.neuron_type_distribution.update(neuron_types)

    def update_surrogate_gradients(self, model):
        """Update surrogate gradient statistics from model"""
        slopes = []
        slope_by_layer = {}
        
        # Some processors are not nn.Module; walk attributes safely
        modules: List[Any] = []
        if hasattr(model, 'named_modules'):
            modules = [m for _, m in model.named_modules()]
        else:
            # Heuristic: collect attributes that look like modules
            for attr in dir(model):
                try:
                    obj = getattr(model, attr)
                except Exception:
                    continue
                if hasattr(obj, 'surrogate_slope') or hasattr(obj, 'named_modules'):
                    modules.append(obj)
        for module in modules:
            if hasattr(module, 'surrogate_slope'):
                try:
                    slope_val = float(module.surrogate_slope.detach().item())
                except Exception:
                    continue
                slopes.append(slope_val)
                name = getattr(module, '__class__', type('X',(object,),{})).__name__
                slope_by_layer[name] = slope_val
        
        if slopes:
            self.surrogate_slope_stats = {
                'mean': float(np.mean(slopes)),
                'std': float(np.std(slopes)),
                'min': float(np.min(slopes)),
                'max': float(np.max(slopes)),
                'by_layer': slope_by_layer
            }

    def update_membrane_potentials(self, model):
        """Update membrane potential statistics from model"""
        membrane_means = []
        membrane_stds = []
        
        modules: List[Any] = []
        if hasattr(model, 'named_modules'):
            modules = [m for _, m in model.named_modules()]
        else:
            for attr in dir(model):
                try:
                    obj = getattr(model, attr)
                except Exception:
                    continue
                if hasattr(obj, 'membrane_potential'):
                    modules.append(obj)
        for module in modules:
            if hasattr(module, 'membrane_potential') and module.membrane_potential is not None:
                try:
                    mem_data = module.membrane_potential.detach()
                    membrane_means.append(float(mem_data.mean().item()))
                    membrane_stds.append(float(mem_data.std().item()))
                except Exception:
                    continue
        
        if membrane_means:
            self.membrane_potential_stats = {
                'mean_of_means': float(np.mean(membrane_means)),
                'mean_of_stds': float(np.mean(membrane_stds)),
                'range': {
                    'min_mean': float(np.min(membrane_means)),
                    'max_mean': float(np.max(membrane_means))
                }
            }

    def update_gradient_health(self, model):
        """Update gradient flow health metrics"""
        total_norm = 0.0
        layer_norms = {}
        param_count = 0
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = float(param.grad.norm().item())
                    layer_norms[name] = grad_norm
                    total_norm += grad_norm ** 2
                    param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** 0.5
            
            self.gradient_flow_health = {
                'total_norm': total_norm,
                'avg_norm': total_norm / param_count,
                'param_count': param_count,
                'status': 'healthy' if 0.001 < total_norm < 10.0 else 'concerning'
            }
            
            # Update training stability
            if total_norm > 100.0:
                self.training_stability = "exploding"
            elif total_norm < 0.001:
                self.training_stability = "vanishing"
            else:
                self.training_stability = "stable"
            
            self.stability_history.append(self.training_stability)
            if len(self.stability_history) > 1000:
                self.stability_history = self.stability_history[-1000:]

    def update_loss_history(self, loss_value: float):
        """Update loss tracking"""
        self.loss_history.append(loss_value)
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a comprehensive health summary"""
        health_summary = {
            'overall_health': 'good',
            'concerns': [],
            'recommendations': []
        }
        
        # Check firing rates
        silent_zones = [zone for zone, status in self.zone_health_status.items() if status == 'silent']
        hyperactive_zones = [zone for zone, status in self.zone_health_status.items() if status == 'hyperactive']
        
        if silent_zones:
            health_summary['concerns'].append(f"Silent zones: {silent_zones}")
            health_summary['recommendations'].append("Increase surrogate gradient slopes for silent zones")
            health_summary['overall_health'] = 'concerning'
        
        if hyperactive_zones:
            health_summary['concerns'].append(f"Hyperactive zones: {hyperactive_zones}")
            health_summary['recommendations'].append("Decrease surrogate gradient slopes for hyperactive zones")
            health_summary['overall_health'] = 'concerning'
        
        # Check gradient flow
        if self.training_stability in ['exploding', 'vanishing']:
            health_summary['concerns'].append(f"Gradient flow: {self.training_stability}")
            if self.training_stability == 'exploding':
                health_summary['recommendations'].append("Reduce learning rate and apply gradient clipping")
            else:
                health_summary['recommendations'].append("Increase learning rate and check network connectivity")
            health_summary['overall_health'] = 'critical' if self.training_stability == 'exploding' else 'concerning'
        
        # Check firing rate trends
        if len(self.firing_rate_history) > 10:
            recent_trend = np.polyfit(range(10), self.firing_rate_history[-10:], 1)[0]
            if recent_trend < -0.01:
                health_summary['concerns'].append("Decreasing firing rate trend")
                health_summary['recommendations'].append("Monitor for potential degradation")
        
        return health_summary

    def get_training_recommendations(self) -> List[str]:
        """Get specific training recommendations based on current stats"""
        recommendations = []
        
        # Firing rate recommendations
        if self.avg_firing_rate < 0.01:
            recommendations.append("Overall firing rate too low - increase surrogate gradient slopes")
        elif self.avg_firing_rate > 0.7:
            recommendations.append("Overall firing rate too high - decrease surrogate gradient slopes")
        
        # Zone-specific recommendations
        for zone_name, firing_rate in self.zone_firing_rates.items():
            if firing_rate < 0.001:
                recommendations.append(f"{zone_name}: Silent zone - increase slope or check connectivity")
            elif firing_rate > 0.8:
                recommendations.append(f"{zone_name}: Hyperactive zone - decrease slope or add inhibition")
        
        # Gradient flow recommendations
        if self.gradient_flow_health.get('status') == 'concerning':
            total_norm = self.gradient_flow_health.get('total_norm', 0)
            if total_norm > 10.0:
                recommendations.append("High gradient norms - apply gradient clipping")
            elif total_norm < 0.001:
                recommendations.append("Low gradient norms - check learning rate and network initialization")
        
        # Stability recommendations
        if len(self.stability_history) > 5:
            recent_stability = self.stability_history[-5:]
            if recent_stability.count('exploding') > 2:
                recommendations.append("Frequent gradient explosion - reduce learning rate significantly")
            elif recent_stability.count('vanishing') > 2:
                recommendations.append("Frequent vanishing gradients - increase learning rate or change architecture")
        
        return recommendations

    def print_summary(self):
        """Print a human-readable summary of brain statistics"""
        print("=" * 60)
        print("NEUROMORPHIC BRAIN STATISTICS SUMMARY")
        print("=" * 60)
        
        print(f"Total Neurons: {self.num_neurons:,}")
        print(f"Total Zones: {self.num_zones}")
        print(f"Total Parameters: {self.num_parameters:,}")
        print(f"Average Firing Rate: {self.avg_firing_rate:.4f}")
        print(f"Training Stability: {self.training_stability}")
        
        if self.zone_firing_rates:
            print("\nZone Firing Rates:")
            for zone, rate in self.zone_firing_rates.items():
                print(f"  {zone}: {rate:.4f}")
        
        if self.neuron_type_distribution:
            print("\nNeuron Type Distribution:")
            for neuron_type, count in self.neuron_type_distribution.items():
                print(f"  {neuron_type}: {count}")
        
        health_summary = self.get_health_summary()
        print(f"\nOverall Health: {health_summary['overall_health'].upper()}")
        
        if health_summary['concerns']:
            print("\nConcerns:")
            for concern in health_summary['concerns']:
                print(f"  ⚠️  {concern}")
        
        if health_summary['recommendations']:
            print("\nRecommendations:")
            for rec in health_summary['recommendations']:
                print(f"  💡 {rec}")
        
        print("=" * 60)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override to avoid automatic saving on every attribute change"""
        super().__setattr__(name, value)
        # Removed automatic saving to prevent frequent file I/O
        # Users can call save_stats() explicitly when needed

# Helper class for collecting statistics during training
class StatsCollector:
    """Helper class to collect statistics from neuromorphic brain during training"""
    
    def __init__(self):
        self.stats = BrainStats()
        self.update_counter = 0
    
    def update_from_brain(self, brain, loss_value: Optional[float] = None):
        """Update statistics from brain state"""
        
        # Basic counts
        if hasattr(brain, 'zones'):
            self.stats.num_zones = len(brain.zones)
        
        # Collect zone activities if available
        zone_activities = {}
        if hasattr(brain, 'enhanced_zones'):
            for zone_name, zone in brain.enhanced_zones.items():
                if hasattr(zone, 'get_activity_stats'):
                    zone_activities[zone_name] = zone.get_activity_stats()
        
        if zone_activities:
            self.stats.update_from_zone_activity(zone_activities)
        
        # Update gradient and model statistics if brain has a model
        if hasattr(brain, 'model') or hasattr(brain, 'neuromorphic_processor'):
            model = getattr(brain, 'model', None)
            if model is None:
                # Try to build a composite object exposing modules/parameters from zones
                class _Composite:
                    pass
                model = _Composite()
                # Aggregate zone modules for inspection
                modules = []
                for _, zone in getattr(brain, 'enhanced_zones', {}).items():
                    modules.append(zone)
                setattr(model, 'modules', modules)
            if model:
                self.stats.update_surrogate_gradients(model)
                self.stats.update_membrane_potentials(model)
                # Skip gradient health if model lacks parameters
                if hasattr(model, 'named_parameters'):
                    self.stats.update_gradient_health(model)
        
        # Update loss if provided
        if loss_value is not None:
            self.stats.update_loss_history(float(loss_value))
        
        self.update_counter += 1
    
    def get_stats(self) -> BrainStats:
        """Get current statistics"""
        return self.stats
    
    def save_checkpoint(self, filename: str):
        """Save current statistics to checkpoint file"""
        self.stats.save_stats(filename)
    
    def should_print_summary(self, every_n_updates: int = 100) -> bool:
        """Check if it's time to print summary"""
        return self.update_counter % every_n_updates == 0