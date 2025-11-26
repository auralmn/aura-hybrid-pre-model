"""
Prosody Modulation Benchmark

Compares SNN performance with and without prosody-driven attention on:
1. Spike efficiency (energy)
2. Classification accuracy (if applicable)
3. Attention distribution
4. Threshold adaptation
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.language_zone.prosody_attention import ProsodyAttentionBridge
from src.core.language_zone.prosody_gif import ProsodyModulatedGIF
from src.core.language_zone.gif_neuron import GIFNeuron
from src.core.language_zone.synapsis import Synapsis

sns.set_style("whitegrid")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    total_spikes: float
    avg_spike_rate: float
    winner_spike_ratio: float
    inference_time_ms: float
    memory_mb: float
    attention_entropy: float = 0.0
    accuracy: float = 0.0


class SNN_Baseline(nn.Module):
    """Baseline SNN without prosody modulation."""
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.synapsis = Synapsis(input_dim, hidden_dim)
        self.gif = GIFNeuron(hidden_dim, hidden_dim, L=16)
    
    def forward(self, x):
        h, _ = self.synapsis(x)
        spikes, state = self.gif(h)
        return spikes, state, {}


class SNN_Prosody(nn.Module):
    """SNN with prosody-driven attention."""
    def __init__(self, input_dim=64, hidden_dim=128, attention_preset='emotional'):
        super().__init__()
        self.prosody_bridge = ProsodyAttentionBridge(
            attention_preset=attention_preset,
            k_winners=3
        )
        self.synapsis = Synapsis(input_dim, hidden_dim)
        self.gif = ProsodyModulatedGIF(
            hidden_dim, hidden_dim, L=16,
            attention_modulation_strength=0.5
        )
    
    def forward(self, x, token_strings=None):
        # Compute attention gains
        if token_strings is not None:
            token_ids = torch.arange(x.shape[1])
            attention_gains, metadata = self.prosody_bridge(
                token_ids.unsqueeze(0),
                [token_strings]
            )
        else:
            attention_gains = torch.ones(x.shape[0], x.shape[1], device=x.device)
            metadata = {}
        
        h, _ = self.synapsis(x)
        spikes, state = self.gif(h, attention_gains=attention_gains)
        
        return spikes, state, metadata


def benchmark_model(
    model: nn.Module,
    texts: List[str],
    num_runs: int = 10,
    device: str = 'cpu'
) -> BenchmarkResult:
    """
    Benchmark a single model.
    
    Args:
        model: SNN model to benchmark
        texts: List of text samples
        num_runs: Number of runs for timing
        device: Device to run on
    
    Returns:
        BenchmarkResult with statistics
    """
    model = model.to(device)
    model.eval()
    
    all_spikes = []
    all_times = []
    all_metadata = []
    
    with torch.no_grad():
        for run in range(num_runs):
            for text in texts:
                # Create input
                tokens = text.split()
                seq_len = len(tokens)
                x = torch.randn(1, seq_len, 64).to(device)  # Batch=1
                
                # Measure inference time
                start = time.time()
                
                if isinstance(model, SNN_Prosody):
                    spikes, state, metadata = model(x, token_strings=tokens)
                else:
                    spikes, state, metadata = model(x)
                
                elapsed = (time.time() - start) * 1000  # ms
                
                all_spikes.append(spikes)
                all_times.append(elapsed)
                all_metadata.append(metadata)
    
    # Calculate statistics
    total_spikes = sum(s.sum().item() for s in all_spikes)
    avg_spike_rate = total_spikes / (len(all_spikes) * all_spikes[0].numel())
    avg_time = np.mean(all_times)
    
    # Memory usage (rough estimate)
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    memory_mb = total_params / (1024 ** 2)
    
    # Winner spike ratio (if prosody model)
    winner_spike_ratio = 0.0
    attention_entropy = 0.0
    if isinstance(model, SNN_Prosody) and all_metadata:
        # Calculate from first valid metadata
        for meta in all_metadata:
            if meta and 'winners' in meta:
                winners = meta['winners'][0] if meta['winners'] else []
                if len(winners) > 0:
                    # Estimate winner contribution
                    winner_spike_ratio = len(winners) / len(tokens)
                    break
        
        # Calculate attention entropy
        if all_metadata and all_metadata[0] and 'salience' in all_metadata[0]:
            salience = all_metadata[0]['salience'][0].numpy()
            # Normalize to probability distribution
            p = salience / (salience.sum() + 1e-8)
            attention_entropy = -np.sum(p * np.log(p + 1e-8))
    
    name = "Prosody-Modulated" if isinstance(model, SNN_Prosody) else "Baseline"
    
    return BenchmarkResult(
        name=name,
        total_spikes=total_spikes,
        avg_spike_rate=avg_spike_rate,
        winner_spike_ratio=winner_spike_ratio,
        inference_time_ms=avg_time,
        memory_mb=memory_mb,
        attention_entropy=attention_entropy
    )


def run_comprehensive_benchmark(
    texts: List[str],
    device: str = 'cpu',
    num_runs: int = 10
) -> Dict[str, BenchmarkResult]:
    """
    Run comprehensive benchmark comparing baseline vs prosody-modulated SNNs.
    
    Args:
        texts: List of test texts
        device: Device to run on
        num_runs: Number of runs per model
    
    Returns:
        Dictionary of results
    """
    print("="*60)
    print("SNN Prosody Modulation Benchmark")
    print("="*60)
    print(f"Test samples: {len(texts)}")
    print(f"Runs per model: {num_runs}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Create models
    baseline = SNN_Baseline()
    prosody_emotional = SNN_Prosody(attention_preset='emotional')
    prosody_analytical = SNN_Prosody(attention_preset='analytical')
    
    # Benchmark each
    results = {}
    
    print("Benchmarking Baseline SNN...")
    results['baseline'] = benchmark_model(baseline, texts, num_runs, device)
    
    print("Benchmarking Prosody-Modulated SNN (emotional)...")
    results['prosody_emotional'] = benchmark_model(prosody_emotional, texts, num_runs, device)
    
    print("Benchmarking Prosody-Modulated SNN (analytical)...")
    results['prosody_analytical'] = benchmark_model(prosody_analytical, texts, num_runs, device)
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Total spikes: {result.total_spikes:.0f}")
        print(f"  Avg spike rate: {result.avg_spike_rate:.4f}")
        print(f"  Inference time: {result.inference_time_ms:.2f} ms")
        print(f"  Memory: {result.memory_mb:.2f} MB")
        if result.attention_entropy > 0:
            print(f"  Attention entropy: {result.attention_entropy:.3f}")
            print(f"  Winner ratio: {result.winner_spike_ratio:.2%}")
    
    return results


def visualize_benchmark_results(results: Dict[str, BenchmarkResult], save_path: str = None):
    """Create comprehensive visualization of benchmark results."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    names = list(results.keys())
    display_names = [r.name for r in results.values()]
    
    # 1. Total Spikes (Energy Efficiency)
    ax1 = fig.add_subplot(gs[0, 0])
    spikes = [results[n].total_spikes for n in names]
    colors = ['C0' if 'baseline' in n else 'C1' for n in names]
    bars = ax1.bar(range(len(names)), spikes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add percentage labels
    baseline_spikes = results['baseline'].total_spikes
    for i, (bar, s) in enumerate(zip(bars, spikes)):
        pct = (s / baseline_spikes - 1) * 100
        label = f'{s:.0f}\n({pct:+.1f}%)'
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(display_names, rotation=15, ha='right')
    ax1.set_ylabel('Total Spike Count', fontsize=11)
    ax1.set_title('Energy Efficiency (Total Spikes)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Inference Time
    ax2 = fig.add_subplot(gs[0, 1])
    times = [results[n].inference_time_ms for n in names]
    bars = ax2.bar(range(len(names)), times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{t:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(display_names, rotation=15, ha='right')
    ax2.set_ylabel('Time (ms)', fontsize=11)
    ax2.set_title('Inference Speed', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Spike Rate
    ax3 = fig.add_subplot(gs[0, 2])
    rates = [results[n].avg_spike_rate for n in names]
    bars = ax3.bar(range(len(names)), rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, r in zip(bars, rates):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{r:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(display_names, rotation=15, ha='right')
    ax3.set_ylabel('Spike Rate', fontsize=11)
    ax3.set_title('Average Spike Rate', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Attention Entropy (prosody models only)
    ax4 = fig.add_subplot(gs[1, 0])
    prosody_names = [n for n in names if 'prosody' in n]
    entropies = [results[n].attention_entropy for n in prosody_names]
    prosody_display = [results[n].name for n in prosody_names]
    
    if entropies:
        bars = ax4.bar(range(len(prosody_names)), entropies, color='C2', alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, e in zip(bars, entropies):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{e:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4.set_xticks(range(len(prosody_names)))
        ax4.set_xticklabels(prosody_display, rotation=15, ha='right')
    
    ax4.set_ylabel('Entropy (bits)', fontsize=11)
    ax4.set_title('Attention Entropy', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Efficiency Gains Summary
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')
    
    # Calculate improvements
    improvements = []
    for name in names:
        if 'baseline' not in name:
            spike_reduction = (1 - results[name].total_spikes / baseline_spikes) * 100
            time_overhead = (results[name].inference_time_ms / results['baseline'].inference_time_ms - 1) * 100
            improvements.append((results[name].name, spike_reduction, time_overhead))
    
    summary_text = "Prosody Modulation Impact Summary\n" + "="*50 + "\n\n"
    summary_text += f"Baseline Reference:\n"
    summary_text += f"  Total Spikes: {baseline_spikes:.0f}\n"
    summary_text += f"  Inference Time: {results['baseline'].inference_time_ms:.2f} ms\n\n"
    
    for name, spike_red, time_over in improvements:
        summary_text += f"{name}:\n"
        summary_text += f"  Spike Reduction: {spike_red:.1f}%\n"
        summary_text += f"  Time Overhead: {time_over:+.1f}%\n"
        summary_text += f"  Net Benefit: {'✓' if spike_red > abs(time_over) else '~'}\n\n"
    
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
            verticalalignment='top', fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('SNN Prosody Modulation Benchmark Results', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved benchmark visualization to {save_path}")
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Test texts with varying emotional content
    test_texts = [
        "WOW this is absolutely INCREDIBLE and AMAZING!",
        "I am deeply disappointed and frustrated with this.",
        "The weather today is quite pleasant and mild.",
        "This situation is URGENT and requires immediate attention!",
        "I feel grateful and blessed for this opportunity."
    ]
    
    # Run benchmark
    results = run_comprehensive_benchmark(
        texts=test_texts,
        device='cpu',
        num_runs=10
    )
    
    # Visualize
    visualize_benchmark_results(results, save_path='prosody_benchmark_results.png')
    
    print("\n✅ Benchmark complete!")
