"""
Prosody Influence Visualization Script

Visualizes how prosody-driven attention modulates spiking behavior over time.
Shows attention gains, spike patterns, and threshold modulation.
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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict
import seaborn as sns

from src.core.language_zone.prosody_attention import ProsodyAttentionBridge
from src.core.language_zone.prosody_gif import ProsodyModulatedGIF
from src.core.language_zone.gif_neuron import GIFNeuron

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def visualize_prosody_influence(
    text: str,
    attention_preset: str = 'emotional',
    save_path: str = None
):
    """
    Comprehensive visualization of prosody influence on SNN dynamics.
    
    Args:
        text: Input text to analyze
        attention_preset: Attention configuration
        save_path: Optional path to save figure
    """
    # Tokenize (simple word split)
    tokens = text.split()
    token_ids = list(range(len(tokens)))
    
    # Initialize components
    prosody_bridge = ProsodyAttentionBridge(
        attention_preset=attention_preset,
        k_winners=3
    )
    
    gif_standard = GIFNeuron(input_dim=64, hidden_dim=128, L=16)
    gif_prosody = ProsodyModulatedGIF(
        input_dim=64,
        hidden_dim=128,
        L=16,
        attention_modulation_strength=0.5
    )
    
    # Extract prosody channels
    amp, pitch, boundary = prosody_bridge.extract_prosody(tokens)
    
    # Compute attention
    result = prosody_bridge.compute_attention_gains(token_ids, tokens)
    salience = result['salience']
    winners = result['winners_idx']
    spikes_dict = result['spikes']
    
    # Create input and attention gains
    seq_len = len(tokens)
    x = torch.randn(1, seq_len, 64)
    
    attention_gains_torch = torch.from_numpy(salience).float().unsqueeze(0)
    attention_gains_torch = attention_gains_torch * result['mu_scalar']
    
    # Forward passes
    with torch.no_grad():
        spikes_baseline, (v_baseline, theta_baseline) = gif_standard(x)
        spikes_modulated, (v_modulated, theta_modulated) = gif_prosody(
            x, attention_gains=attention_gains_torch
        )
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. Prosody Channels
    ax1 = fig.add_subplot(gs[0, :])
    x_pos = np.arange(len(tokens))
    width = 0.25
    
    ax1.bar(x_pos - width, amp, width, label='Amplitude', alpha=0.8, color='C0')
    ax1.bar(x_pos, pitch, width, label='Pitch', alpha=0.8, color='C1')
    ax1.bar(x_pos + width, boundary, width, label='Boundary', alpha=0.8, color='C2')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(tokens, rotation=45, ha='right')
    ax1.set_ylabel('Channel Activity')
    ax1.set_title(f'Prosody Channels: "{text}"', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Highlight winners
    for w in winners:
        ax1.axvspan(w-0.5, w+0.5, alpha=0.2, color='red')
    
    # 2. Attention Salience
    ax2 = fig.add_subplot(gs[1, :])
    colors = ['red' if i in winners else 'blue' for i in range(len(tokens))]
    bars = ax2.bar(x_pos, salience, color=colors, alpha=0.7)
    
    # Add winner stars
    for w in winners:
        ax2.text(w, salience[w] + 0.05, '★', ha='center', fontsize=20, color='red')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tokens, rotation=45, ha='right')
    ax2.set_ylabel('Salience Score')
    ax2.set_title(f'Attention Salience (μ={result["mu_scalar"]:.2f}, k-winners={len(winners)})',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Spike Rasters - Baseline vs Modulated
    ax3 = fig.add_subplot(gs[2, 0])
    spike_data_baseline = spikes_baseline[0].T.numpy()
    im3 = ax3.imshow(spike_data_baseline[:50], aspect='auto', cmap='hot', interpolation='nearest')
    ax3.set_ylabel('Neuron Index')
    ax3.set_xlabel('Time (tokens)')
    ax3.set_title('Baseline GIF (no prosody)', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    plt.colorbar(im3, ax=ax3, label='Spike count')
    
    ax4 = fig.add_subplot(gs[2, 1])
    spike_data_modulated = spikes_modulated[0].T.numpy()
    im4 = ax4.imshow(spike_data_modulated[:50], aspect='auto', cmap='hot', interpolation='nearest')
    ax4.set_ylabel('Neuron Index')
    ax4.set_xlabel('Time (tokens)')
    ax4.set_title('Prosody-Modulated GIF', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    plt.colorbar(im4, ax=ax4, label='Spike count')
    
    # Highlight winner regions
    for w in winners:
        ax3.axvline(w, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax4.axvline(w, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # 4. Spike Count Comparison
    ax5 = fig.add_subplot(gs[3, 0])
    spike_counts_baseline = spikes_baseline[0].sum(dim=1).numpy()
    spike_counts_modulated = spikes_modulated[0].sum(dim=1).numpy()
    
    ax5.plot(spike_counts_baseline, label='Baseline', linewidth=2, marker='o')
    ax5.plot(spike_counts_modulated, label='Prosody-Modulated', linewidth=2, marker='s')
    ax5.fill_between(x_pos, spike_counts_baseline, spike_counts_modulated, alpha=0.2)
    
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(tokens, rotation=45, ha='right')
    ax5.set_ylabel('Total Spike Count')
    ax5.set_xlabel('Token')
    ax5.set_title('Spike Count per Token', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Highlight winners
    for w in winners:
        ax5.axvspan(w-0.5, w+0.5, alpha=0.2, color='red')
    
    # 5. Threshold Modulation
    ax6 = fig.add_subplot(gs[3, 1])
    
    # Calculate effective thresholds
    baseline_threshold = np.ones(seq_len)
    modulated_threshold = np.zeros(seq_len)
    
    for t in range(seq_len):
        gain = attention_gains_torch[0, t].item()
        threshold_scale = 1.0 - 0.5 * (gain - 1.0)  # modulation_strength=0.5
        threshold_scale = np.clip(threshold_scale, 0.5, 1.5)
        modulated_threshold[t] = threshold_scale
    
    ax6.plot(baseline_threshold, label='Baseline', linewidth=2, linestyle='--')
    ax6.plot(modulated_threshold, label='Prosody-Modulated', linewidth=2, marker='o')
    ax6.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(tokens, rotation=45, ha='right')
    ax6.set_ylabel('Threshold Scale')
    ax6.set_xlabel('Token')
    ax6.set_title('Firing Threshold Modulation', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Highlight winners
    for w in winners:
        ax6.axvspan(w-0.5, w+0.5, alpha=0.2, color='red')
    
    # 6. Energy Efficiency
    ax7 = fig.add_subplot(gs[4, :])
    
    total_baseline = spikes_baseline.sum().item()
    total_modulated = spikes_modulated.sum().item()
    
    categories = ['Baseline\n(no prosody)', 'Prosody-Modulated']
    spike_totals = [total_baseline, total_modulated]
    colors_bar = ['C0', 'C1']
    
    bars = ax7.bar(categories, spike_totals, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, spike_totals)):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f} spikes\n({val/total_baseline*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax7.set_ylabel('Total Spike Count', fontsize=12)
    ax7.set_title('Energy Efficiency: Total Spike Count', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add summary text
    efficiency_gain = (1 - total_modulated / total_baseline) * 100
    summary_text = (
        f"Summary:\n"
        f"  Winners: {len(winners)}/{len(tokens)} tokens ({len(winners)/len(tokens)*100:.1f}%)\n"
        f"  Spike reduction: {efficiency_gain:.1f}%\n"
        f"  Attention gain (μ): {result['mu_scalar']:.2f}"
    )
    
    ax7.text(0.02, 0.98, summary_text, transform=ax7.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, family='monospace')
    
    plt.suptitle(f'Prosody Influence Analysis - {attention_preset.capitalize()} Preset',
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("Prosody Influence Statistics")
    print("="*60)
    print(f"Text: \"{text}\"")
    print(f"Tokens: {len(tokens)}")
    print(f"Winners: {[tokens[i] for i in winners]}")
    print(f"\nSpike Statistics:")
    print(f"  Baseline total: {total_baseline:.0f}")
    print(f"  Modulated total: {total_modulated:.0f}")
    print(f"  Reduction: {efficiency_gain:.1f}%")
    print(f"\nWinner Spikes:")
    winner_baseline = spikes_baseline[0, winners].sum().item()
    winner_modulated = spikes_modulated[0, winners].sum().item()
    print(f"  Baseline: {winner_baseline:.0f} ({winner_baseline/total_baseline*100:.1f}%)")
    print(f"  Modulated: {winner_modulated:.0f} ({winner_modulated/total_modulated*100:.1f}%)")
    print("="*60)


def compare_attention_presets(text: str, save_path: str = None):
    """Compare different attention presets on the same text."""
    presets = ['analytical', 'emotional', 'historical', 'streaming']
    tokens = text.split()
    token_ids = list(range(len(tokens)))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, preset in enumerate(presets):
        bridge = ProsodyAttentionBridge(attention_preset=preset, k_winners=3)
        result = bridge.compute_attention_gains(token_ids, tokens)
        
        sal = result['salience']
        winners = result['winners_idx']
        
        colors = ['red' if j in winners else 'blue' for j in range(len(tokens))]
        axes[i].bar(range(len(tokens)), sal, color=colors, alpha=0.7)
        
        # Add stars
        for w in winners:
            axes[i].text(w, sal[w] + 0.05, '★', ha='center', fontsize=16, color='red')
        
        axes[i].set_xticks(range(len(tokens)))
        axes[i].set_xticklabels(tokens, rotation=45, ha='right')
        axes[i].set_ylabel('Salience')
        axes[i].set_title(f'{preset.capitalize()} (μ={result["mu_scalar"]:.2f})', fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Attention Preset Comparison: "{text}"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    # Example usage
    texts = [
        "WOW this is absolutely INCREDIBLE and AMAZING!",
        "I am so disappointed and frustrated with this situation.",
        "The weather today is quite pleasant and mild."
    ]
    
    for i, text in enumerate(texts):
        print(f"\n{'='*60}")
        print(f"Example {i+1}")
        print(f"{'='*60}")
        visualize_prosody_influence(text, attention_preset='emotional')
        
    # Compare presets
    print(f"\n{'='*60}")
    print("Comparing Attention Presets")
    print(f"{'='*60}")
    compare_attention_presets(texts[0])
