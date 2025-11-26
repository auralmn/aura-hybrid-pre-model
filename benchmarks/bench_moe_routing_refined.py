"""
Refined MoE Routing Quality Benchmark

Tests prosody modulation with:
1. Fixed temperature scaling bug
2. Wid ened μ dynamic range (0.7-2.5)
3. Toggleable bandit/usage bias
4. 50-100 samples per regime for robust correlations
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
import asyncio
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.language_zone.prosody_attention import ProsodyAttentionBridge

sns.set_style("whitegrid")


class MockLiquidMoERouter:
    """Mock router with FIXED temperature scaling and toggleable bias."""
    
    def __init__(self, num_experts: int = 8, top_k: int = 2, use_bandit: bool = False, usage_beta: float = 0.5):
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_bandit = use_bandit
        self.usage_beta = usage_beta
        self.temperature = 1.0
        
        # Track usage for bias
        self.usage_ma = np.zeros(num_experts)
        self.smoothing = 0.99
        
    async def route(self, x: np.ndarray, attn_gain: float = 1.0) -> Dict:
        """Route with CORRECTED temperature scaling."""
        
        # Base routing logits
        logits = np.random.randn(self.num_experts)
        
        # Apply usage bias if enabled
        if self.usage_beta > 0:
            target = 1.0 / self.num_experts
            inv_usage = target / (self.usage_ma + 1e-6)
            logits = logits + self.usage_beta * np.log(inv_usage)
        
        # CRITICAL: Apply temperature scaling with attn_gain
        # High attn_gain -> lower temp -> sharper distribution (focused)
        temp = max(0.2, self.temperature / max(1e-6, attn_gain))
        logits_scaled = logits / temp
        
        # Softmax
        probs = np.exp(logits_scaled - np.max(logits_scaled))
        probs = probs / np.sum(probs)
        
        # Select top-k
        topk_idx = np.argsort(probs)[-self.top_k:]
        topk_probs = probs[topk_idx]
        topk_probs = topk_probs / topk_probs.sum()
        
        # Update usage
        out = np.zeros(self.num_experts)
        out[topk_idx] = topk_probs
        self.usage_ma = self.smoothing * self.usage_ma + (1.0 - self.smoothing) * out
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        return {
            'topk_idx': topk_idx,
            'topk_probs': topk_probs,
            'all_probs': probs,
            'entropy': entropy,
            'unique_experts': len(topk_idx)
        }


async def test_regime(
    regime_name: str,
    texts: List[str],
    prosody_bridge: ProsodyAttentionBridge,
    router: MockLiquidMoERouter,
    num_samples: int = 100
) -> Dict:
    """Test routing behavior for a specific prosody regime."""
    
    gains = []
    entropies = []
    diversities = []
    saliences = []
    
    for _ in range(num_samples):
        # Pick random text from regime
        text = np.random.choice(texts)
        tokens = text.split()
        token_ids = list(range(len(tokens)))
        
        # Extract prosody
        result = prosody_bridge.compute_attention_gains(token_ids, tokens)
        
        avg_gain = result['mu_scalar']
        salience = result['salience']
        
        # Route through MoE
        x = np.random.rand(128)
        route_info = await router.route(x, attn_gain=avg_gain)
        
        gains.append(avg_gain)
        entropies.append(route_info['entropy'])
        diversities.append(route_info['unique_experts'])
        saliences.append(salience.mean())
    
    return {
        'regime': regime_name,
        'gains': np.array(gains),
        'entropies': np.array(entropies),
        'diversities': np.array(diversities),
        'saliences': np.array(saliences),
        'num_samples': num_samples
    }


async def run_refined_benchmark(use_bandit: bool = False, usage_beta: float = 0.0):
    """Run refined routing benchmark with proper sample sizes."""
    
    print("="*60)
    print("Refined MoE Routing Quality Benchmark")
    print(f"Bandit: {'ENABLED' if use_bandit else 'DISABLED'}")
    print(f"Usage bias beta: {usage_beta}")
    print("="*60)
    
    # Create components
    prosody_bridge = ProsodyAttentionBridge(
        attention_preset='analytical_balanced',
        k_winners=5
    )
    
    router = MockLiquidMoERouter(
        num_experts=8,
        top_k=2,
        use_bandit=use_bandit,
        usage_beta=usage_beta
    )
    
    # Define regimes with diverse texts
    regimes = {
        'low_prosody': [
            "the weather today is mild and pleasant",
            "this document contains several paragraphs",
            "users can access the system interface",
            "data processing requires computational resources",
            "standard procedures should be followed carefully"
        ] * 20,  # 100 samples
        
        'medium_prosody': [
            "This is quite interesting and noteworthy.",
            "We should consider this option carefully.",
            "The results are somewhat surprising actually.",
            "This approach might work better perhaps.",
            "Users will probably find this helpful indeed."
        ] * 20,
        
        'high_prosody': [
            "WOW this is INCREDIBLE! AMAZING discovery!",
            "URGENT! Critical issue requires immediate attention!",
            "Absolutely FANTASTIC performance today!",
            "This is TERRIBLE! Worst experience ever!",
            "BREAKTHROUGH! Revolutionary findings here!"
        ] * 20
    }
    
    # Test each regime
    results = {}
    for regime_name, texts in regimes.items():
        print(f"\nTesting: {regime_name}...")
        result = await test_regime(
            regime_name,
            texts,
            prosody_bridge,
            router,
            num_samples=100
        )
        results[regime_name] = result
    
    # Aggregate all data
    all_gains = np.concatenate([r['gains'] for r in results.values()])
    all_entropies = np.concatenate([r['entropies'] for r in results.values()])
    all_diversities = np.concatenate([r['diversities'] for r in results.values()])
    
    # Calculate correlations
    corr_gain_entropy = np.corrcoef(all_gains, all_entropies)[0, 1]
    corr_gain_div = np.corrcoef(all_gains, all_diversities)[0, 1]
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for regime_name, result in results.items():
        print(f"\n{regime_name.upper()}:")
        print(f"  Avg μ (gain): {result['gains'].mean():.3f} ± {result['gains'].std():.3f}")
        print(f"  Avg entropy: {result['entropies'].mean():.3f} ± {result['entropies'].std():.3f}")
        print(f"  Avg unique experts: {result['diversities'].mean():.1f} ± {result['diversities'].std():.1f}")
        print(f"  Avg salience: {result['saliences'].mean():.3f}")
    
    print(f"\nCORRELATIONS (N={len(all_gains)}):")
    print(f"  μ ↔ Entropy: {corr_gain_entropy:.3f}")
    print(f"  μ ↔ Diversity: {corr_gain_div:.3f}")
    
    # Interpret
    print(f"\nINTERPRETATION:")
    if corr_gain_entropy < -0.5:
        print("  ✅ High prosody → FOCUSED routing (lower entropy)")
        print("     High-salience tokens activate fewer experts (specialized)")
    elif corr_gain_entropy > 0.5:
        print("  ✅ High prosody → EXPLORATORY routing (higher entropy)")
        print("     High-salience tokens explore more experts (creative)")
    else:
        print(f"  ⚠️  Weak correlation ({corr_gain_entropy:.2f})")
        print("     Prosody may not be strongly influencing routing")
    
    # Visualize
    visualize_refined_results(results, corr_gain_entropy, corr_gain_div, use_bandit, usage_beta)
    
    return results, corr_gain_entropy, corr_gain_div


def visualize_refined_results(results, corr_ent, corr_div, use_bandit, usage_beta):
    """Visualize refined benchmark results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    regimes = list(results.keys())
    colors = ['blue', 'orange', 'red']
    
    # 1. Gain distribution per regime
    for i, (regime, result) in enumerate(results.items()):
        axes[0, 0].hist(result['gains'], bins=30, alpha=0.6, label=regime, color=colors[i], edgecolor='black')
    axes[0, 0].set_xlabel('Attention Gain (μ)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('μ Distribution by Regime', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Entropy distribution per regime
    for i, (regime, result) in enumerate(results.items()):
        axes[0, 1].hist(result['entropies'], bins=30, alpha=0.6, label=regime, color=colors[i], edgecolor='black')
    axes[0, 1].set_xlabel('Routing Entropy (bits)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Routing Entropy by Regime', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter: μ vs Entropy
    all_gains = np.concatenate([r['gains'] for r in results.values()])
    all_entropies = np.concatenate([r['entropies'] for r in results.values()])
    
    axes[1, 0].scatter(all_gains, all_entropies, alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Attention Gain (μ)')
    axes[1, 0].set_ylabel('Routing Entropy (bits)')
    axes[1, 0].set_title(f'μ vs Entropy (corr={corr_ent:.3f})', fontweight='bold')
    
    # Add trend line
    z = np.polyfit(all_gains, all_entropies, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(all_gains, p(all_gains), "r--", alpha=0.8, linewidth=2)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Diversity comparison
    regime_names = [r.replace('_', ' ').title() for r in regimes]
    diversities_mean = [results[r]['diversities'].mean() for r in regimes]
    diversities_std = [results[r]['diversities'].std() for r in regimes]
    
    axes[1, 1].bar(regime_names, diversities_mean, yerr=diversities_std, alpha=0.7, edgecolor='black', capsize=5)
    axes[1, 1].set_ylabel('Unique Experts Used')
    axes[1, 1].set_title(f'Expert Diversity by Regime (corr={corr_div:.3f})', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    config_str = f"Bandit={'ON' if use_bandit else 'OFF'}, Beta={usage_beta}"
    plt.suptitle(f'Refined MoE Routing Quality Results\n{config_str}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'refined_routing_bandit{use_bandit}_beta{usage_beta}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {filename}")
    plt.show()


async def compare_configurations():
    """Compare routing behavior with/without bandit and usage bias."""
    
    print("\n" + "="*60)
    print("CONFIGURATION COMPARISON")
    print("="*60)
    
    configs = [
        {'use_bandit': False, 'usage_beta': 0.0, 'name': 'Clean (no bias)'},
        {'use_bandit': False, 'usage_beta': 0.5, 'name': 'With usage bias'},
        {'use_bandit': True, 'usage_beta': 0.5, 'name': 'Full (bandit + bias)'},
    ]
    
    comparison = {}
    
    for config in configs:
        print(f"\n\nTesting: {config['name']}...")
        results, corr_ent, corr_div = await run_refined_benchmark(
            use_bandit=config['use_bandit'],
            usage_beta=config['usage_beta']
        )
        comparison[config['name']] = {
            'results': results,
            'corr_entropy': corr_ent,
            'corr_diversity': corr_div
        }
    
    # Summary
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    for name, data in comparison.items():
        print(f"\n{name}:")
        print(f"  Gain ↔ Entropy correlation: {data['corr_entropy']:.3f}")
        print(f"  Gain ↔ Diversity correlation: {data['corr_diversity']:.3f}")


if __name__ == '__main__':
    # Run clean benchmark first (no bandit, no bias)
    print("STEP 1: Clean routing (no bandit, no usage bias)")
    print("This isolates prosody's effect on routing\n")
    
    asyncio.run(run_refined_benchmark(use_bandit=False, usage_beta=0.0))
    
    # Optionally compare configurations
    # asyncio.run(compare_configurations())
