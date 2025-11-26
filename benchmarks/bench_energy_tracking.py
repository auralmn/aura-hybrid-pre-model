"""
Energy Tracking and Validation Framework

Tracks and validates energy consumption in SNN Language Zone components.
Estimates neuromorphic hardware energy based on spike counts.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.language_zone.prosody_attention import ProsodyAttentionBridge
from src.core.language_zone.prosody_gif import ProsodyModulatedGIF
from src.core.language_zone.gif_neuron import GIFNeuron
from src.core.language_zone.synapsis import Synapsis

sns.set_style("whitegrid")


@dataclass
class EnergyMetrics:
    """Energy consumption metrics."""
    component_name: str
    total_spikes: int
    avg_spike_rate: float
    energy_pj: float  # Picojoules
    energy_relative: float  # Relative to baseline
    
    def __str__(self):
        return (
            f"{self.component_name}:\n"
            f"  Spikes: {self.total_spikes}\n"
            f"  Rate: {self.avg_spike_rate:.3f}\n"
            f"  Energy: {self.energy_pj:.1f} pJ ({self.energy_relative:.1%} of baseline)"
        )


class EnergyTracker:
    """Tracks energy consumption in SNN components."""
    
    # Energy costs (based on neuromorphic hardware estimates)
    ENERGY_PER_SPIKE_PJ = 0.3  # Picojoules per spike (TrueNorth-like)
    ENERGY_PER_SYNAPSE_PJ = 0.1  # Picojoules per synaptic operation
    ENERGY_PER_MAC_PJ = 4.6  # Conventional MAC operation (for comparison)
    
    def __init__(self):
        self.metrics = []
        self.baseline_energy = None
    
    def track_component(
        self,
        component_name: str,
        spikes: torch.Tensor,
        synaptic_ops: int = 0
    ) -> EnergyMetrics:
        """
        Track energy for a component.
        
        Args:
            component_name: Name of component
            spikes: Spike tensor (batch, time, neurons)
            synaptic_ops: Number of synaptic operations
        """
        total_spikes = int(spikes.sum().item())
        avg_spike_rate = float(spikes.mean().item())
        
        # Calculate energy
        spike_energy = total_spikes * self.ENERGY_PER_SPIKE_PJ
        synapse_energy = synaptic_ops * self.ENERGY_PER_SYNAPSE_PJ
        total_energy = spike_energy + synapse_energy
        
        # Relative to baseline (if set)
        if self.baseline_energy is None:
            self.baseline_energy = total_energy
            energy_relative = 1.0
        else:
            energy_relative = total_energy / self.baseline_energy if self.baseline_energy > 0 else 1.0
        
        metrics = EnergyMetrics(
            component_name=component_name,
            total_spikes=total_spikes,
            avg_spike_rate=avg_spike_rate,
            energy_pj=total_energy,
            energy_relative=energy_relative
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def compare_with_conventional(self, num_macs: int) -> Dict:
        """Compare SNN energy with conventional computation."""
        snn_energy = sum(m.energy_pj for m in self.metrics)
        conventional_energy = num_macs * self.ENERGY_PER_MAC_PJ
        
        return {
            'snn_energy_pj': snn_energy,
            'conventional_energy_pj': conventional_energy,
            'energy_ratio': snn_energy / conventional_energy if conventional_energy > 0 else 0,
            'energy_savings_pct': (1 - snn_energy / conventional_energy) * 100 if conventional_energy > 0 else 0
        }
    
    def get_summary(self) -> str:
        """Get energy summary."""
        total_energy = sum(m.energy_pj for m in self.metrics)
        total_spikes = sum(m.total_spikes for m in self.metrics)
        
        summary = "Energy Tracking Summary\n"
        summary += "="*50 + "\n"
        summary += f"Total Energy: {total_energy:.1f} pJ\n"
        summary += f"Total Spikes: {total_spikes}\n"
        summary += f"Avg Energy/Spike: {total_energy/total_spikes if total_spikes > 0 else 0:.3f} pJ\n\n"
        
        for metrics in self.metrics:
            summary += str(metrics) + "\n\n"
        
        return summary


def validate_energy_tracking():
    """Validate energy tracking on SNN components."""
    
    print("="*60)
    print("Energy Tracking Validation")
    print("="*60)
    
    # Create components
    # Data flow: x (128) → Synapsis (128→256) → GIF (256→256)
    synapsis = Synapsis(in_features=128, out_features=256)
    gif_standard = GIFNeuron(input_dim=256, hidden_dim=256, L=16)
    gif_prosody = ProsodyModulatedGIF(input_dim=256, hidden_dim=256, L=16)
    
    # Create test input
    batch_size = 4
    seq_len = 50
    x = torch.randn(batch_size, seq_len, 128)
    
    # Energy tracker
    tracker_baseline = EnergyTracker()
    tracker_prosody = EnergyTracker()
    
    print("\n[1/2] Baseline (no prosody)...")
    
    # Forward pass without prosody
    with torch.no_grad():
        # Synapsis
        h_baseline, _ = synapsis(x)
        synaptic_ops_baseline = batch_size * seq_len * 128 * 256
        tracker_baseline.track_component("Synapsis", h_baseline, synaptic_ops_baseline)
        
        # GIF
        spikes_baseline, _ = gif_standard(h_baseline)
        tracker_baseline.track_component("GIF", spikes_baseline)
    
    print("\n[2/2] With prosody modulation...")
    
    # Forward pass with prosody
    with torch.no_grad():
        # Generate attention gains (high for some tokens)
        attention_gains = torch.ones(batch_size, seq_len)
        attention_gains[:, 10:20] = 2.5  # High attention region
        attention_gains[:, 30:35] = 0.5  # Low attention region
        
        # Synapsis
        h_prosody, _ = synapsis(x)
        synaptic_ops_prosody = batch_size * seq_len * 128 * 256
        tracker_prosody.track_component("Synapsis", h_prosody, synaptic_ops_prosody)
        
        # Prosody-modulated GIF
        spikes_prosody, _ = gif_prosody(h_prosody, attention_gains=attention_gains)
        tracker_prosody.track_component("ProsodyGIF", spikes_prosody)
    
    # Compare
    print("\n" + "="*60)
    print("ENERGY COMPARISON")
    print("="*60)
    
    print("\nBaseline (no prosody):")
    print(tracker_baseline.get_summary())
    
    print("\nWith Prosody Modulation:")
    print(tracker_prosody.get_summary())
    
    # Calculate savings
    energy_baseline = sum(m.energy_pj for m in tracker_baseline.metrics)
    energy_prosody = sum(m.energy_pj for m in tracker_prosody.metrics)
    energy_savings = (1 - energy_prosody / energy_baseline) * 100 if energy_baseline > 0 else 0
    
    print(f"\nEnergy Savings with Prosody: {energy_savings:.1f}%")
    
    # Compare with conventional
    num_macs = batch_size * seq_len * 128 * 256  # Equivalent MAC operations
    comparison = tracker_baseline.compare_with_conventional(num_macs)
    
    print(f"\nSNN vs Conventional:")
    print(f"  SNN Energy: {comparison['snn_energy_pj']:.1f} pJ")
    print(f"  Conventional (MACs): {comparison['conventional_energy_pj']:.1f} pJ")
    print(f"  SNN Advantage: {comparison['energy_savings_pct']:.1f}% savings")
    
    # Visualize
    visualize_energy_breakdown(tracker_baseline, tracker_prosody, comparison)
    
    return tracker_baseline, tracker_prosody, comparison


def visualize_energy_breakdown(
    tracker_baseline: EnergyTracker,
    tracker_prosody: EnergyTracker,
    comparison: Dict
):
    """Visualize energy breakdown."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Component Energy Breakdown
    components_baseline = [m.component_name for m in tracker_baseline.metrics]
    energy_baseline = [m.energy_pj for m in tracker_baseline.metrics]
    
    components_prosody = [m.component_name for m in tracker_prosody.metrics]
    energy_prosody = [m.energy_pj for m in tracker_prosody.metrics]
    
    x = np.arange(len(components_baseline))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, energy_baseline, width, label='Baseline', alpha=0.8)
    axes[0, 0].bar(x + width/2, energy_prosody, width, label='Prosody', alpha=0.8)
    axes[0, 0].set_ylabel('Energy (pJ)')
    axes[0, 0].set_title('Component Energy Breakdown', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(components_baseline)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Spike Counts
    spikes_baseline = [m.total_spikes for m in tracker_baseline.metrics]
    spikes_prosody = [m.total_spikes for m in tracker_prosody.metrics]
    
    axes[0, 1].bar(x - width/2, spikes_baseline, width, label='Baseline', alpha=0.8)
    axes[0, 1].bar(x + width/2, spikes_prosody, width, label='Prosody', alpha=0.8)
    axes[0, 1].set_ylabel('Total Spikes')
    axes[0, 1].set_title('Spike Count Comparison', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(components_baseline)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. SNN vs Conventional
    categories = ['SNN\n(Baseline)', 'SNN\n(Prosody)', 'Conventional\n(MACs)']
    energies = [
        sum(energy_baseline),
        sum(energy_prosody),
        comparison['conventional_energy_pj']
    ]
    colors = ['C0', 'C1', 'C3']
    
    bars = axes[1, 0].bar(categories, energies, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Energy (pJ)')
    axes[1, 0].set_title('SNN vs Conventional Energy', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (bar, e) in enumerate(zip(bars, energies)):
        if i < 2:
            pct = (e / energies[2]) * 100
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{e:.0f} pJ\n({pct:.1f}%)',
                           ha='center', va='bottom', fontsize=10)
    
    # 4. Energy Savings Summary
    axes[1, 1].axis('off')
    
    savings_text = "Energy Efficiency Summary\n" + "="*40 + "\n\n"
    
    baseline_total = sum(energy_baseline)
    prosody_total = sum(energy_prosody)
    prosody_savings = (1 - prosody_total / baseline_total) * 100 if baseline_total > 0 else 0
    
    savings_text += f"Prosody Modulation Impact:\n"
    savings_text += f"  Baseline: {baseline_total:.1f} pJ\n"
    savings_text += f"  Prosody: {prosody_total:.1f} pJ\n"
    savings_text += f"  Savings: {prosody_savings:.1f}%\n\n"
    
    savings_text += f"SNN vs Conventional:\n"
    savings_text += f"  SNN: {comparison['snn_energy_pj']:.1f} pJ\n"
    savings_text += f"  Conventional: {comparison['conventional_energy_pj']:.1f} pJ\n"
    savings_text += f"  SNN Advantage: {comparison['energy_savings_pct']:.1f}%\n\n"
    
    savings_text += f"Energy per Spike:\n"
    savings_text += f"  {EnergyTracker.ENERGY_PER_SPIKE_PJ:.1f} pJ\n\n"
    
    savings_text += f"Energy per MAC (conventional):\n"
    savings_text += f"  {EnergyTracker.ENERGY_PER_MAC_PJ:.1f} pJ"
    
    axes[1, 1].text(0.1, 0.9, savings_text, transform=axes[1, 1].transAxes,
                    fontsize=11, family='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('Energy Tracking Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('energy_validation_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved energy validation to energy_validation_results.png")
    plt.show()


if __name__ == '__main__':
    # Run energy validation
    validate_energy_tracking()
    
    print("\n✅ Energy Tracking Validation Complete!")
