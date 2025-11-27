"""
GPU-Native Brain Statistics Collector.

Collects and aggregates metrics from distributed GPU-native brain zones.
Optimized to minimize CPU-GPU synchronization overhead during training steps.
"""

import torch
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

@dataclass
class BrainStats:
    """
    Data container for brain statistics.
    Holds standard Python types for easy JSON serialization.
    """
    timestamp: float = 0.0
    num_neurons: int = 0
    num_zones: int = 0
    num_layers: int = 0
    
    # Aggregated metrics
    avg_firing_rate: float = 0.0
    max_firing_rate: float = 0.0
    active_zones: int = 0
    
    # Detailed stats per zone (zone_name -> dict)
    zone_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Stability health
    stability_status: str = "unknown"
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)
    
    def print_summary(self):
        print(f"\n--- Brain Stats (t={self.timestamp:.2f}) ---")
        print(f"Neurons: {self.num_neurons:,} | Zones: {self.num_zones}")
        print(f"Global Activity: {self.avg_firing_rate:.4f} Hz (Max: {self.max_firing_rate:.4f})")
        print(f"Status: {self.stability_status}")
        
        if self.zone_stats:
            print("Zone Breakdown:")
            for name, stats in self.zone_stats.items():
                fr = stats.get('avg_firing_rate', 0.0)
                status = stats.get('health', 'unknown')
                print(f"  • {name:<20}: {fr:.4f} Hz [{status}]")

class StatsCollector:
    """
    Low-overhead statistics collector for GPU-native brains.
    """
    def __init__(self):
        self.history: List[BrainStats] = []
        self.start_time = time.time()
        
        # Running buffers for smoothing (on CPU to save VRAM, updated infrequently)
        self._global_fr_ema = 0.0
        self.alpha = 0.1 # Smoothing factor

    def update_from_brain(self, brain) -> None:
        """
        Collect stats from the brain.
        NOTE: This triggers CPU-GPU sync. Call this only when you need to log (e.g., every 10-100 steps),
        not every training step if maximum throughput is required.
        """
        # We assume 'brain' is an instance of EnhancedBrain
        
        current_stats = BrainStats(timestamp=time.time() - self.start_time)
        
        total_neurons = 0
        total_firing_sum = 0.0
        max_firing = 0.0
        active_cnt = 0
        zone_details = {}
        
        # Iterate over enhanced zones (GPU modules)
        if hasattr(brain, 'enhanced_zones'):
            current_stats.num_zones = len(brain.enhanced_zones)
            
            for zone_name, zone in brain.enhanced_zones.items():
                # 1. Get raw metrics from the zone
                # NeuromorphicBrainZone.get_activity_stats() performs the .item() conversion
                # We assume this is acceptable overhead for the logging step
                raw_stats = zone.get_activity_stats()
                
                # 2. Extract key metrics
                n_neurons = raw_stats.get('total_neurons', 0)
                
                # Check for nested neuron stats if available
                neuron_stats = raw_stats.get('neuron_type_stats', {})
                if neuron_stats:
                    # Weighted average across groups
                    avg_fr = 0.0
                    total_count = 0
                    for _, n_s in neuron_stats.items():
                        c = n_s.get('count', 0)
                        r = n_s.get('mean_firing_rate', 0.0)
                        avg_fr += r * c
                        total_count += c
                    avg_fr = avg_fr / max(1, total_count)
                else:
                    # Fallback to simple average
                    avg_fr = raw_stats.get('avg_firing_rate', 0.0)
                
                # 3. Determine Zone Health
                health = "healthy"
                if avg_fr < 0.001: health = "silent"
                elif avg_fr > 0.8: health = "hyperactive"
                
                # 4. Aggregate
                total_neurons += n_neurons
                total_firing_sum += avg_fr * n_neurons
                max_firing = max(max_firing, avg_fr)
                if avg_fr > 0.01:
                    active_cnt += 1
                    
                zone_details[zone_name] = {
                    'avg_firing_rate': avg_fr,
                    'neuron_count': n_neurons,
                    'health': health
                }
                
                # Add layer count if available
                if hasattr(zone, 'config') and hasattr(zone.config, 'num_layers'):
                    current_stats.num_layers += zone.config.num_layers

        # Finalize Global Stats
        current_stats.num_neurons = total_neurons
        current_stats.active_zones = active_cnt
        current_stats.max_firing_rate = max_firing
        
        if total_neurons > 0:
            current_stats.avg_firing_rate = total_firing_sum / total_neurons
        
        current_stats.zone_stats = zone_details
        
        # Stability Heuristic
        silent_ratio = sum(1 for z in zone_details.values() if z['health'] == 'silent') / max(1, len(zone_details))
        hyper_ratio = sum(1 for z in zone_details.values() if z['health'] == 'hyperactive') / max(1, len(zone_details))
        
        if silent_ratio > 0.5:
            current_stats.stability_status = "unstable_silent"
        elif hyper_ratio > 0.3:
            current_stats.stability_status = "unstable_hyperactive"
        else:
            current_stats.stability_status = "stable"
            
        # Update EMA
        self._global_fr_ema = self.alpha * current_stats.avg_firing_rate + (1 - self.alpha) * self._global_fr_ema
        
        # Store
        self.history.append(current_stats)
        
        # Keep history manageable
        if len(self.history) > 1000:
            self.history.pop(0)

    def get_stats(self) -> BrainStats:
        """Get the most recent statistics snapshot."""
        if not self.history:
            return BrainStats()
        return self.history[-1]

    def save_history(self, filepath: str):
        """Save entire stats history to JSON."""
        data = [asdict(s) for s in self.history]
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save brain stats: {e}")

    def reset(self):
        self.history = []
        self.start_time = time.time()