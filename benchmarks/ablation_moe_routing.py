"""
Complete Ablation Study: MoE Routing with Prosody
Tests 4 configurations incrementally to isolate component effects.
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
from dataclasses import dataclass

from src.core.language_zone.prosody_attention import ProsodyAttentionBridge

sns.set_style("whitegrid")


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    name: str
    use_bandit: bool
    usage_beta: float
    description: str


class MockLiquidMoERouter:
    """Mock router with configurable bandit and usage bias."""
    
    def __init__(self, num_experts: int = 8, top_k: int = 2, 
                 use_bandit: bool = False, usage_beta: float = 0.5):
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_bandit = use_bandit
        self.usage_beta = usage_beta
        self.temperature = 1.0
        
        # Usage tracking
        self.usage_ma = np.zeros(num_experts)
        self.smoothing = 0.99
        
        # Bandit (UCB)
        if use_bandit:
            self.total_rewards = np.zeros(num_experts)
            self.selection_counts = np.ones(num_experts)
            self.total_selections = num_experts
        
    async def route(self, x: np.ndarray, attn_gain: float = 1.0) -> Dict:
        """Route with configurable bandit and bias."""
        
        # Base routing logits
        logits = np.random.randn(self.num_experts)
        
        # Apply usage bias if enabled
        if self.usage_beta > 0:
            target = 1.0 / self.num_experts
            inv_usage = target / (self.usage_ma + 1e-6)
            logits = logits + self.usage_beta * np.log(inv_usage)
        
        # Temperature scaling with attn_gain
        temp = max(0.2, self.temperature / max(1e-6, attn_gain))
        logits_scaled = logits / temp
        
        # Softmax
        probs = np.exp(logits_scaled - np.max(logits_scaled))
        probs = probs / np.sum(probs)
        
        # Bandit modulation (if enabled)
        if self.use_bandit:
            ucb_scores = self._get_ucb_scores()
            # Combine with routing probs (70% routing, 30% UCB)
            probs = 0.7 * probs + 0.3 * ucb_scores
            probs = probs / probs.sum()
        
        # Select top-k
        topk_idx = np.argsort(probs)[-self.top_k:]
        topk_probs = probs[topk_idx]
        topk_probs = topk_probs / topk_probs.sum()
        
        # Update usage
        out = np.zeros(self.num_experts)
        out[topk_idx] = topk_probs
        self.usage_ma = self.smoothing * self.usage_ma + (1.0 - self.smoothing) * out
        
        # Update bandit (mock reward)
        if self.use_bandit:
            for idx in topk_idx:
                reward = np.random.rand()  # Mock reward
                self.total_rewards[idx] += reward
                self.selection_counts[idx] += 1
                self.total_selections += 1
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        return {
            'topk_idx': topk_idx,
            'topk_probs': topk_probs,
            'all_probs': probs,
            'entropy': entropy,
            'unique_experts': len(topk_idx)
        }
    
    def _get_ucb_scores(self) -> np.ndarray:
        """Compute UCB scores for bandit."""
        avg_rewards = self.total_rewards / self.selection_counts
        confidence = 2.0 * np.sqrt(
            np.log(self.total_selections + 1) / self.selection_counts
        )
        ucb = avg_rewards + confidence
        # Normalize to probabilities
        ucb = np.exp(ucb - np.max(ucb))
        return ucb / ucb.sum()


async def run_single_config(
    config: AblationConfig,
    prosody_bridge: ProsodyAttentionBridge,
    num_samples: int = 100
) -> Dict:
    """Run benchmark for a single configuration."""
    
    print(f"\n{'='*60}")
    print(f"Config: {config.name}")
    print(f"  Bandit: {config.use_bandit}, Usage Beta: {config.usage_beta}")
    print(f"{'='*60}")
    
    # Create router with this config
    router = MockLiquidMoERouter(
        num_experts=8,
        top_k=2,
        use_bandit=config.use_bandit,
        usage_beta=config.usage_beta
    )
    
    # Test regimes
    regimes = {
        'low_prosody': [
            "the weather today is mild and pleasant",
            "this document contains several paragraphs",
            "users can access the system interface"
        ] * 34,  # ~100 samples
        
        'medium_prosody': [
            "This is quite interesting and noteworthy.",
            "We should consider this option carefully.",
            "The results are somewhat surprising actually."
        ] * 34,
        
        'high_prosody': [
            "WOW this is INCREDIBLE! AMAZING discovery!",
            "URGENT! Critical issue requires immediate attention!",
            "Absolutely FANTASTIC performance today!"
        ] * 34
    }
    
    results = {}
    all_gains = []
    all_entropies = []
    all_diversities = []
    
    for regime_name, texts in regimes.items():
        regime_gains = []
        regime_entropies = []
        regime_diversities = []
        
        for _ in range(num_samples):
            text = np.random.choice(texts)
            tokens = text.split()
            token_ids = list(range(len(tokens)))
            
            # Extract prosody
            result = prosody_bridge.compute_attention_gains(token_ids, tokens)
            avg_gain = result['mu_scalar']
            
            # Route
            x = np.random.rand(128)
            route_info = await router.route(x, attn_gain=avg_gain)
            
            regime_gains.append(avg_gain)
            regime_entropies.append(route_info['entropy'])
            regime_diversities.append(route_info['unique_experts'])
            
            all_gains.append(avg_gain)
            all_entropies.append(route_info['entropy'])
            all_diversities.append(route_info['unique_experts'])
        
        results[regime_name] = {
            'gains': np.array(regime_gains),
            'entropies': np.array(regime_entropies),
            'diversities': np.array(regime_diversities)
        }
        
        print(f"\n{regime_name.upper()}:")
        print(f"  Avg μ: {np.mean(regime_gains):.3f} ± {np.std(regime_gains):.3f}")
        print(f"  Avg entropy: {np.mean(regime_entropies):.3f} ± {np.std(regime_entropies):.3f}")
        print(f"  Avg unique experts: {np.mean(regime_diversities):.1f}")
    
    # Calculate correlations
    all_gains = np.array(all_gains)
    all_entropies = np.array(all_entropies)
    all_diversities = np.array(all_diversities)
    
    corr_ent = np.corrcoef(all_gains, all_entropies)[0, 1]
    corr_div = np.corrcoef(all_gains, all_diversities)[0, 1] if len(np.unique(all_diversities)) > 1 else 0.0
    
    print(f"\nCORRELATIONS:")
    print(f"  μ ↔ Entropy: {corr_ent:.3f}")
    print(f"  μ ↔ Diversity: {corr_div:.3f}")
    
    return {
        'config': config,
        'regime_results': results,
        'correlations': {
            'entropy': corr_ent,
            'diversity': corr_div
        },
        'low_entropy': results['low_prosody']['entropies'].mean(),
        'high_entropy': results['high_prosody']['entropies'].mean(),
    }


async def run_ablation_study():
    """Run complete ablation study across 4 configurations."""
    
    print("="*70)
    print("ABLATION STUDY: MoE Routing with Prosody")
    print("="*70)
    print("\nTesting 4 configurations to isolate component effects:\n")
    
    # Define configurations
    configs = [
        AblationConfig(
            name="Step 1: Clean (baseline)",
            use_bandit=False,
            usage_beta=0.0,
            description="No bandit, no usage bias - pure prosody signal"
        ),
        AblationConfig(
            name="Step 2: Usage bias only",
            use_bandit=False,
            usage_beta=0.5,
            description="Adds load balancing pressure"
        ),
        AblationConfig(
            name="Step 3: Bandit only",
            use_bandit=True,
            usage_beta=0.0,
            description="Adds exploration via UCB"
        ),
        AblationConfig(
            name="Step 4: Full system",
            use_bandit=True,
            usage_beta=0.5,
            description="Both bandit and usage bias enabled"
        ),
    ]
    
    # Create prosody bridge
    prosody_bridge = ProsodyAttentionBridge(
        attention_preset='analytical_balanced',
        k_winners=5
    )
    
    # Run each configuration
    all_results = []
    for config in configs:
        result = await run_single_config(config, prosody_bridge, num_samples=100)
        all_results.append(result)
    
    # Generate summary
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    print(f"\n{'Config':<30} {'Low Ent':>10} {'High Ent':>10} {'μ↔Ent':>10} {'Status':>10}")
    print("-"*70)
    
    for res in all_results:
        config_name = res['config'].name.split(":")[0]  # "Step 1"
        low_ent = res['low_entropy']
        high_ent = res['high_entropy']
        corr = res['correlations']['entropy']
        
        # Status check
        if low_ent > high_ent and corr < -0.3:
            status = "✅ PASS"
        elif low_ent > high_ent:
            status = "⚠️ WEAK"
        else:
            status = "❌ FAIL"
        
        print(f"{config_name:<30} {low_ent:>10.3f} {high_ent:>10.3f} {corr:>10.3f} {status:>10}")
    
    # Visualize
    visualize_ablation(all_results)
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    baseline = all_results[0]
    full = all_results[3]
    
    entropy_degradation = (full['correlations']['entropy'] - baseline['correlations']['entropy'])
    
    print(f"\nBaseline correlation: {baseline['correlations']['entropy']:.3f}")
    print(f"Full system correlation: {full['correlations']['entropy']:.3f}")
    print(f"Correlation degradation: {entropy_degradation:+.3f}")
    
    if abs(full['correlations']['entropy']) > 0.3:
        print("\n✅ CONCLUSION: Prosody signal remains strong even with bandit/bias")
        print("   System is ready for production deployment")
    else:
        print("\n⚠️  WARNING: Prosody signal weakened significantly")
        print("   Consider reducing bandit exploration or usage bias strength")
    
    return all_results


def visualize_ablation(all_results: List[Dict]):
    """Visualize ablation study results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = [r['config'].name.split(":")[0] for r in all_results]
    
    # 1. Entropy comparison (Low vs High prosody)
    low_entropies = [r['low_entropy'] for r in all_results]
    high_entropies = [r['high_entropy'] for r in all_results]
    
    x = np.arange(len(config_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, low_entropies, width, label='Low Prosody', alpha=0.8, edgecolor='black')
    axes[0, 0].bar(x + width/2, high_entropies, width, label='High Prosody', alpha=0.8, edgecolor='black')
    axes[0, 0].set_ylabel('Routing Entropy (bits)')
    axes[0, 0].set_title('Entropy: Low vs High Prosody', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(config_names, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=np.mean(low_entropies), color='blue', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=np.mean(high_entropies), color='orange', linestyle='--', alpha=0.5)
    
    # 2. Correlation degradation
    correlations = [r['correlations']['entropy'] for r in all_results]
    colors = ['green' if c < -0.3 else 'orange' if c < -0.2 else 'red' for c in correlations]
    
    axes[0, 1].bar(config_names, correlations, alpha=0.7, edgecolor='black', color=colors)
    axes[0, 1].axhline(y=-0.3, color='green', linestyle='--', alpha=0.5, label='Target (< -0.3)')
    axes[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 1].set_ylabel('Correlation (μ ↔ Entropy)')
    axes[0, 1].set_title('Prosody Sensitivity Across Configs', fontweight='bold')
    axes[0, 1].set_xticklabels(config_names, rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Entropy distributions for baseline vs full
    baseline = all_results[0]
    full = all_results[3]
    
    for regime in ['low_prosody', 'high_prosody']:
        baseline_ent = baseline['regime_results'][regime]['entropies']
        axes[1, 0].hist(baseline_ent, bins=20, alpha=0.5, label=f'Baseline-{regime}', edgecolor='black')
    
    axes[1, 0].set_xlabel('Entropy (bits)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Baseline (Clean) Entropy Distribution', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for regime in ['low_prosody', 'high_prosody']:
        full_ent = full['regime_results'][regime]['entropies']
        axes[1, 1].hist(full_ent, bins=20, alpha=0.5, label=f'Full-{regime}', edgecolor='black')
    
    axes[1, 1].set_xlabel('Entropy (bits)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Full System Entropy Distribution', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('MoE Routing Ablation Study', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved ablation visualization to ablation_study_results.png")
    plt.show()


if __name__ == '__main__':
    asyncio.run(run_ablation_study())
