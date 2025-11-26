"""
Prosody Attention Hyperparameter Tuning

Addresses k_winners underutilization issue by testing different configurations
and finding optimal thresholds for salience.
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
from typing import List, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.language_zone.multi_channel_attention import MultiChannelSpikingAttention, prosody_channels_from_text

sns.set_style("whitegrid")


@dataclass
class TuningResult:
    """Results from a single hyperparameter configuration."""
    config_name: str
    k_winners: int
    w_amp: float
    w_pitch: float
    w_bound: float
    smoothing: int
    normalize_salience: bool
    
    avg_winners_found: float
    winner_utilization: float  # % of k_winners actually used
    avg_salience: float
    max_salience: float
    
    def __str__(self):
        return (
            f"{self.config_name}:\n"
            f"  Winners found: {self.avg_winners_found:.1f}/{self.k_winners} "
            f"({self.winner_utilization:.1%} utilization)\n"
            f"  Salience range: {self.avg_salience:.3f} - {self.max_salience:.3f}"
        )


def test_configuration(
    config: Dict,
    test_texts: List[str],
    config_name: str
) -> TuningResult:
    """Test a single attention configuration."""
    
    # Create attention with specific config
    attention = MultiChannelSpikingAttention(
        k_winners=config['k_winners'],
        w_amp=config.get('w_amp', 1.0),
        w_pitch=config.get('w_pitch', 1.0),
        w_bound=config.get('w_bound', 1.0),
        smoothing=config.get('smoothing', 0),
        normalize_salience=config.get('normalize_salience', True),
        gain_up=config.get('gain_up', 1.8),
        gain_down=config.get('gain_down', 0.6)
    )
    
    winners_found_list = []
    salience_list = []
    
    for text in test_texts:
        tokens = text.split()
        token_ids = list(range(len(tokens)))
        
        # Extract prosody
        amp, pitch, boundary = prosody_channels_from_text(tokens)
        
        # Compute attention
        result = attention.compute(
            token_ids=token_ids,
            amp=amp,
            pitch=pitch,
            boundary=boundary
        )
        
        winners_found = len(result['winners_idx'])
        winners_found_list.append(winners_found)
        salience_list.append(result['salience'])
    
    # Calculate statistics
    avg_winners = np.mean(winners_found_list)
    utilization = avg_winners / config['k_winners']
    
    all_salience = np.concatenate(salience_list)
    avg_salience = np.mean(all_salience)
    max_salience = np.max(all_salience)
    
    return TuningResult(
        config_name=config_name,
        k_winners=config['k_winners'],
        w_amp=config.get('w_amp', 1.0),
        w_pitch=config.get('w_pitch', 1.0),
        w_bound=config.get('w_bound', 1.0),
        smoothing=config.get('smoothing', 0),
        normalize_salience=config.get('normalize_salience', True),
        avg_winners_found=avg_winners,
        winner_utilization=utilization,
        avg_salience=avg_salience,
        max_salience=max_salience
    )


def grid_search_hyperparameters(test_texts: List[str]) -> List[TuningResult]:
    """Perform grid search over hyperparameter space."""
    
    configs = {
        # Baseline (current)
        'baseline': {
            'k_winners': 5,
            'w_amp': 1.0,
            'w_pitch': 1.0,
            'w_bound': 1.0,
            'smoothing': 0,
            'normalize_salience': True
        },
        
        # Reduce smoothing
        'less_smoothing': {
            'k_winners': 5,
            'w_amp': 1.0,
            'w_pitch': 1.0,
            'w_bound': 1.0,
            'smoothing': 0,  # No smoothing
            'normalize_salience': False  # No normalization
        },
        
        # Boost weak signals
        'amplified_channels': {
            'k_winners': 5,
            'w_amp': 1.5,
            'w_pitch': 1.5,
            'w_bound': 1.5,
            'smoothing': 0,
            'normalize_salience': True
        },
        
        # Lower k_winners
        'k3_conservative': {
            'k_winners': 3,
            'w_amp': 1.2,
            'w_pitch': 1.2,
            'w_bound': 1.2,
            'smoothing': 1,
            'normalize_salience': True
        },
        
        # Higher k_winners
        'k7_aggressive': {
            'k_winners': 7,
            'w_amp': 0.8,
            'w_pitch': 0.8,
            'w_bound': 0.8,
            'smoothing': 0,
            'normalize_salience': False
        },
        
        # Emotional preset (high sensitivity)
        'emotional_boosted': {
            'k_winners': 5,
            'w_amp': 1.2,
            'w_pitch': 1.5,
            'w_bound': 0.6,
            'smoothing': 0,
            'normalize_salience': True,
            'gain_up': 2.0,
            'gain_down': 0.4
        },
        
        # Analytical preset (balanced)
        'analytical_balanced': {
            'k_winners': 5,
            'w_amp': 0.8,
            'w_pitch': 1.2,
            'w_bound': 1.0,
            'smoothing': 2,
            'normalize_salience': True,
            'gain_up': 1.5,
            'gain_down': 0.7
        }
    }
    
    results = []
    
    print("="*60)
    print("Hyperparameter Grid Search")
    print("="*60)
    
    for config_name, config in configs.items():
        print(f"\nTesting: {config_name}...")
        result = test_configuration(config, test_texts, config_name)
        results.append(result)
        print(result)
    
    return results


def visualize_tuning_results(results: List[TuningResult], save_path: str = None):
    """Visualize tuning results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = [r.config_name for r in results]
    
    # 1. Winner Utilization
    utilizations = [r.winner_utilization * 100 for r in results]
    colors = ['green' if u >= 60 else 'orange' if u >= 40 else 'red' for u in utilizations]
    
    axes[0, 0].barh(names, utilizations, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(60, color='green', linestyle='--', alpha=0.5, label='Target (60%)')
    axes[0, 0].set_xlabel('k_winners Utilization (%)')
    axes[0, 0].set_title('Winner Utilization Rate', fontweight='bold')
    axes[0, 0].legend()
    
    # 2. Average Winners Found
    avg_winners = [r.avg_winners_found for r in results]
    k_winners = [r.k_winners for r in results]
    
    x = np.arange(len(names))
    axes[0, 1].bar(x, avg_winners, alpha=0.7, label='Winners Found', edgecolor='black')
    axes[0, 1].plot(x, k_winners, 'ro-', linewidth=2, markersize=8, label='k_winners (target)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Winner Count')
    axes[0, 1].set_title('Winners Found vs Target k', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Salience Range
    avg_sal = [r.avg_salience for r in results]
    max_sal = [r.max_salience for r in results]
    
    axes[1, 0].scatter(avg_sal, max_sal, s=200, c=utilizations, cmap='RdYlGn', 
                       edgecolor='black', alpha=0.7)
    
    for i, name in enumerate(names):
        axes[1, 0].annotate(name, (avg_sal[i], max_sal[i]), 
                           fontsize=8, ha='center', va='bottom')
    
    axes[1, 0].set_xlabel('Average Salience')
    axes[1, 0].set_ylabel('Max Salience')
    axes[1, 0].set_title('Salience Range (color=utilization)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Configuration Parameters
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary table
    summary_text = "Top 3 Configurations by Utilization:\n" + "="*40 + "\n\n"
    
    sorted_results = sorted(results, key=lambda r: r.winner_utilization, reverse=True)
    
    for i, result in enumerate(sorted_results[:3], 1):
        summary_text += f"{i}. {result.config_name}\n"
        summary_text += f"   Utilization: {result.winner_utilization:.1%}\n"
        summary_text += f"   Winners: {result.avg_winners_found:.1f}/{result.k_winners}\n"
        summary_text += f"   w_amp={result.w_amp}, w_pitch={result.w_pitch}\n"
        summary_text += f"   smoothing={result.smoothing}, norm={result.normalize_salience}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Prosody Attention Hyperparameter Tuning Results', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved tuning visualization to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # Test texts with varying prosody
    test_texts = [
        "WOW this is absolutely INCREDIBLE and AMAZING!",
        "I am so disappointed and frustrated with this situation.",
        "The weather today is quite pleasant and mild.",
        "URGENT! Critical issue requires immediate attention NOW!",
        "Hello world, how are you doing today?",
        "This is terrible! I hate this! Worst experience ever!",
        "The cat sat on the mat and looked around.",
        "Amazing discovery! Scientists found incredible breakthrough!",
        "Please review the document when you get a chance.",
        "BREAKING NEWS: Major announcement coming soon!!!"
    ]
    
    # Run grid search
    results = grid_search_hyperparameters(test_texts)
    
    # Visualize
    visualize_tuning_results(results, save_path='hyperparameter_tuning_results.png')
    
    # Print recommendation
    best_result = max(results, key=lambda r: r.winner_utilization)
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print(f"\nBest configuration: {best_result.config_name}")
    print(f"Achieves {best_result.winner_utilization:.1%} k_winners utilization")
    print(f"\nSuggested parameters:")
    print(f"  k_winners = {best_result.k_winners}")
    print(f"  w_amp = {best_result.w_amp}")
    print(f"  w_pitch = {best_result.w_pitch}")
    print(f"  w_bound = {best_result.w_bound}")
    print(f"  smoothing = {best_result.smoothing}")
    print(f"  normalize_salience = {best_result.normalize_salience}")
    print("="*60)
