"""
End-to-End Benchmark: GoEmotions Emotion Classification
Tests prosody-driven attention on real emotional text data.
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
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.language_zone.prosody_attention import ProsodyAttentionBridge
from src.core.language_zone.prosody_gif import ProsodyModulatedGIF
from src.core.language_zone.gif_neuron import GIFNeuron
from src.core.language_zone.synapsis import Synapsis

sns.set_style("whitegrid")


class EmotionClassifier(nn.Module):
    """Simple emotion classifier using prosody-modulated SNNs."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_emotions: int = 28,
        attention_preset: str = 'analytical_balanced',
        use_prosody: bool = True
    ):
        super().__init__()
        
        self.use_prosody = use_prosody
        
        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Prosody attention bridge (with optimized parameters)
        if use_prosody:
            self.prosody = ProsodyAttentionBridge(
                attention_preset=attention_preset,
                k_winners=5
            )
            self.encoder = ProsodyModulatedGIF(
                embed_dim, hidden_dim, L=16,
                attention_modulation_strength=0.3
            )
        else:
            self.encoder = GIFNeuron(embed_dim, hidden_dim, L=16)
        
        
        # Processing layers
        # Data flow: encoder_spikes (hidden_dim) → Synapsis (hidden→hidden) → decoder (hidden→embed)
        self.synapsis = Synapsis(in_features=hidden_dim, out_features=hidden_dim)
        self.decoder = GIFNeuron(hidden_dim, embed_dim, L=16)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_emotions)
        )
    
    def forward(self, input_ids, token_strings=None):
        # Embed
        embeds = self.embeddings(input_ids)
        
        # Encode with optional prosody modulation
        if self.use_prosody and token_strings is not None:
            attention_gains, attn_meta = self.prosody(input_ids, token_strings)
            spikes_enc, _ = self.encoder(embeds, attention_gains=attention_gains)
        else:
            spikes_enc, _ = self.encoder(embeds)
            attn_meta = {}
        
        # Process
        hidden, _ = self.synapsis(spikes_enc)
        spikes_dec, _ = self.decoder(hidden)
        
        # Pool and classify
        pooled = spikes_dec.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits, {
            'spikes_enc': spikes_enc.sum().item(),
            'spikes_dec': spikes_dec.sum().item(),
            'attention': attn_meta
        }


def benchmark_goemotion(
    num_samples: int = 500,
    use_prosody: bool = True,
    attention_preset: str = 'analytical_balanced',
    device: str = 'cpu'
):
    """Benchmark on GoEmotions dataset."""
    
    print("="*60)
    print(f"GoEmotions E2E Benchmark")
    print(f"Prosody: {'ENABLED' if use_prosody else 'DISABLED'}")
    if use_prosody:
        print(f"Preset: {attention_preset}")
    print("="*60)
    
    # Load GoEmotions
    print("\nLoading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    test_data = dataset['test'].select(range(min(num_samples, len(dataset['test']))))
    
    # Emotion names
    emotion_names = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    # Create model
    model = EmotionClassifier(
        use_prosody=use_prosody,
        attention_preset=attention_preset
    ).to(device)
    model.eval()
    
    # Metrics
    stats = {
        'total_tokens': 0,
        'total_spikes_enc': 0,
        'total_spikes_dec': 0,
        'high_salience_tokens': 0,
        'winner_rate': [],
        'prosody_gains': [],
        'predictions': [],
        'labels': []
    }
    
    print(f"\nProcessing {len(test_data)} samples...")
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_data)):
            # Tokenize (simple whitespace split)
            text = example['text']
            tokens = text.split()[:64]  # Limit length
            
            # Mock token IDs (in production, use real tokenizer)
            token_ids = [hash(t) % 10000 for t in tokens]
            input_ids = torch.tensor([token_ids]).to(device)
            
            # Forward pass
            if use_prosody:
                logits, info = model(input_ids, token_strings=[tokens])
            else:
                logits, info = model(input_ids)
            
            # Get prediction
            pred = torch.softmax(logits, dim=-1)[0]
            top_emotion = pred.argmax().item()
            
            # Update stats
            stats['total_tokens'] += len(tokens)
            stats['total_spikes_enc'] += info['spikes_enc']
            stats['total_spikes_dec'] += info['spikes_dec']
            
            if use_prosody and 'attention' in info and info['attention']:
                attn = info['attention']
                if 'winners' in attn and attn['winners']:
                    winners = attn['winners'][0]
                    stats['high_salience_tokens'] += len(winners)
                    stats['winner_rate'].append(len(winners) / len(tokens))
                
                if 'salience' in attn and attn['salience']:
                    salience = attn['salience'][0].numpy()
                    stats['prosody_gains'].append(salience.mean())
            
            stats['predictions'].append(top_emotion)
            stats['labels'].append(example['labels'])
    
    # Calculate metrics
    avg_spikes_enc = stats['total_spikes_enc'] / len(test_data)
    avg_spikes_dec = stats['total_spikes_dec'] / len(test_data)
    avg_winner_rate = np.mean(stats['winner_rate']) if stats['winner_rate'] else 0
    avg_prosody_gain = np.mean(stats['prosody_gains']) if stats['prosody_gains'] else 0
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total samples: {len(test_data)}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"\nSpike Statistics:")
    print(f"  Avg encoder spikes/sample: {avg_spikes_enc:.1f}")
    print(f"  Avg decoder spikes/sample: {avg_spikes_dec:.1f}")
    print(f"  Total avg spikes/sample: {avg_spikes_enc + avg_spikes_dec:.1f}")
    
    if use_prosody:
        print(f"\nProsody Statistics:")
        print(f"  High-salience tokens: {stats['high_salience_tokens']}/{stats['total_tokens']} "
              f"({stats['high_salience_tokens']/stats['total_tokens']:.1%})")
        print(f"  Avg winner rate: {avg_winner_rate:.1%}")
        print(f"  Avg prosody gain: {avg_prosody_gain:.3f}")
        print(f"  k_winners utilization: {avg_winner_rate * 5:.1f}/5")
    
    return stats


def compare_with_without_prosody(
    num_samples: int = 200,
    device: str = 'cpu'
):
    """Compare performance with and without prosody."""
    
    print("\n" + "="*60)
    print("COMPARISON: With vs Without Prosody")
    print("="*60)
    
    # Run both benchmarks
    print("\n[1/2] Running WITHOUT prosody...")
    stats_no_prosody = benchmark_goemotion(
        num_samples=num_samples,
        use_prosody=False,
        device=device
    )
    
    print("\n[2/2] Running WITH prosody (analytical_balanced)...")
    stats_with_prosody = benchmark_goemotion(
        num_samples=num_samples,
        use_prosody=True,
        attention_preset='analytical_balanced',
        device=device
    )
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    spike_reduction = (1 - (stats_with_prosody['total_spikes_enc'] + stats_with_prosody['total_spikes_dec']) / 
                       (stats_no_prosody['total_spikes_enc'] + stats_no_prosody['total_spikes_dec'])) * 100
    
    print(f"\nEnergy Efficiency:")
    print(f"  Without prosody: {stats_no_prosody['total_spikes_enc'] + stats_no_prosody['total_spikes_dec']:.0f} spikes/sample")
    print(f"  With prosody: {stats_with_prosody['total_spikes_enc'] + stats_with_prosody['total_spikes_dec']:.0f} spikes/sample")
    print(f"  Reduction: {spike_reduction:.1f}%")
    
    print(f"\nWinner Detection (prosody only):")
    print(f"  Winner rate: {np.mean(stats_with_prosody['winner_rate']):.1%}")
    print(f"  k_winners utilization: {np.mean(stats_with_prosody['winner_rate']) * 5:.1f}/5")
    
    # Visualize
    visualize_comparison(stats_no_prosody, stats_with_prosody)
    
    return stats_no_prosody, stats_with_prosody


def visualize_comparison(stats_no_prosody, stats_with_prosody):
    """Visualize comparison results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Spike counts
    categories = ['No Prosody', 'With Prosody\n(analytical_balanced)']
    encoder_spikes = [
        stats_no_prosody['total_spikes_enc'],
        stats_with_prosody['total_spikes_enc']
    ]
    decoder_spikes = [
        stats_no_prosody['total_spikes_dec'],
        stats_with_prosody['total_spikes_dec']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, encoder_spikes, width, label='Encoder', alpha=0.8)
    axes[0].bar(x + width/2, decoder_spikes, width, label='Decoder', alpha=0.8)
    axes[0].set_ylabel('Total Spikes')
    axes[0].set_title('Spike Count Comparison', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Winner rate distribution (prosody only)
    if stats_with_prosody['winner_rate']:
        axes[1].hist(stats_with_prosody['winner_rate'], bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(stats_with_prosody['winner_rate']), 
                       color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(stats_with_prosody["winner_rate"]):.1%}')
        axes[1].set_xlabel('Winner Rate (% of tokens)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Winner Rate Distribution (with prosody)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('goemotion_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison to goemotion_comparison.png")
    plt.show()


if __name__ == '__main__':
    # Run comparison benchmark
    compare_with_without_prosody(num_samples=200, device='cpu')
    
    print("\n✅ E2E Benchmark Complete!")
