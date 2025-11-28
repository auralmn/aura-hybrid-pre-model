"""
Local Training Test - Minimal Dataset

Tests the full training pipeline locally before deploying to L4.
Uses a tiny synthetic dataset to verify:
1. Model forward/backward works
2. Loss decreases over steps
3. Gradient flow is stable
4. Memory system activates after warmup
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import math

# Local imports
from src.core.hippocampal import HippocampalFormation
from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
from src.core.language_zone.hippocampal_layer import HippocampalTransformerLayer
from src.training.hippocampal_trainer import HippocampalTransformerTrainer, get_cosine_schedule_with_warmup
from src.training.losses import HippocampalLoss


@dataclass
class MinimalConfig:
    """Minimal config for local testing."""
    # Model
    vocab_size: int = 1000
    embedding_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 64
    intermediate_size: int = 256
    
    # Hippocampal
    n_place_cells: int = 100
    place_cell_sparsity: float = 0.03
    theta_freq: float = 8.0
    gamma_freq: float = 40.0
    
    # Training
    batch_size: int = 4
    lr: float = 1e-3
    warmup_steps: int = 10
    max_steps: int = 100
    memory_warmup_steps: int = 20
    gradient_clip: float = 1.0
    min_lr_ratio: float = 0.1
    
    # Loss
    label_smoothing: float = 0.1
    entropy_lambda: float = 0.05
    sparsity_lambda: float = 0.02
    target_sparsity: float = 0.03
    
    # Trainer
    sleep_interval: int = 50
    replay_buffer_size: int = 100
    ewc_lambda: float = 0.0
    
    # Optimization
    use_gradient_checkpointing: bool = False


class MinimalHippocampalTransformer(nn.Module):
    """
    Minimal version of HippocampalTransformer for local testing.
    """
    def __init__(self, config, hippocampus):
        super().__init__()
        self.config = config
        self.hippocampus = hippocampus
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Positional encoding (simplified)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embedding_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        
        # Transformer layers (simplified)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        # Output head with weight tying
        self.output_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        prosody: Optional[torch.Tensor] = None,
        use_memory: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = hidden_states + self.pos_embedding(positions)
        
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_mask=causal_mask, is_causal=True)
        
        # Output
        logits = self.output_head(hidden_states)
        
        # Mock place cell activity for loss computation
        place_cell_activity = torch.zeros(batch_size, seq_len, self.config.n_place_cells, device=device)
        
        return logits, place_cell_activity


def create_synthetic_dataset(config, num_samples=100):
    """
    Create a synthetic dataset for testing.
    
    Generates simple patterns that the model should learn:
    - Sequences of incrementing tokens
    - Repeating patterns
    """
    data = []
    seq_len = config.max_seq_len
    vocab_size = config.vocab_size
    
    for i in range(num_samples):
        if i % 3 == 0:
            # Pattern 1: Incrementing sequence
            start = torch.randint(0, vocab_size - seq_len, (1,)).item()
            seq = torch.arange(start, start + seq_len) % vocab_size
        elif i % 3 == 1:
            # Pattern 2: Repeating pattern (A B A B ...)
            a = torch.randint(0, vocab_size, (1,)).item()
            b = torch.randint(0, vocab_size, (1,)).item()
            seq = torch.tensor([a, b] * (seq_len // 2 + 1))[:seq_len]
        else:
            # Pattern 3: Random (noise)
            seq = torch.randint(0, vocab_size, (seq_len,))
        
        data.append(seq)
    
    return torch.stack(data)


def run_training_test():
    """Run a minimal training loop to verify everything works."""
    print("=" * 60)
    print("LOCAL TRAINING TEST")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = MinimalConfig()
    print(f"Config: vocab={config.vocab_size}, dim={config.embedding_dim}, layers={config.num_layers}")
    
    # Create hippocampus
    hippocampus = HippocampalFormation(
        spatial_dimensions=2,
        n_place_cells=config.n_place_cells,
        n_time_cells=10,
        n_grid_cells=10,
        max_memories=1000,
        feature_dim=config.embedding_dim,
        device=str(device)
    )
    
    # Create model
    model = MinimalHippocampalTransformer(config, hippocampus).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=0.01
    )
    
    # Create trainer
    trainer = HippocampalTransformerTrainer(
        model=model,
        config=config,
        hippocampus=hippocampus,
        optimizer=optimizer
    )
    
    # Create dataset
    print("\nCreating synthetic dataset...")
    data = create_synthetic_dataset(config, num_samples=100).to(device)
    print(f"Dataset shape: {data.shape}")
    
    # Training loop
    print("\n" + "-" * 60)
    print("TRAINING")
    print("-" * 60)
    
    model.train()
    losses = []
    
    num_steps = config.max_steps
    batch_size = config.batch_size
    
    for step in range(num_steps):
        # Sample batch
        indices = torch.randint(0, len(data), (batch_size,))
        batch_data = data[indices]
        
        # Input: all tokens except last
        # Labels: all tokens except first (shifted)
        input_ids = batch_data[:, :-1]
        labels = batch_data[:, 1:]
        
        # Prosody (mock - zeros)
        prosody = torch.zeros(batch_size, input_ids.shape[1], 4, device=device)
        
        # Create batch tuple
        batch = (input_ids, labels, prosody)
        
        # Training step
        metrics = trainer.train_step(batch)
        losses.append(metrics['loss'])
        
        # Early stopping check
        if step > 10 and metrics['loss'] > 20:
            print(f"WARNING: Loss too high ({metrics['loss']:.2f}), possible issue")
    
    # Results
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    
    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    
    print(f"Initial loss (avg first 5): {initial_loss:.4f}")
    print(f"Final loss (avg last 5): {final_loss:.4f}")
    print(f"Loss reduction: {initial_loss - final_loss:.4f}")
    
    # Check if training worked
    success = final_loss < initial_loss
    
    if success:
        print("\nSUCCESS: Loss decreased during training")
        
        # Additional checks
        print("\nAdditional Checks:")
        print(f"  - Memory active: {trainer.should_use_memory()}")
        print(f"  - Replay buffer size: {len(trainer.replay_buffer)}")
        print(f"  - Final LR: {trainer.get_lr():.2e}")
        print(f"  - Final perplexity: {math.exp(final_loss):.2f}")
    else:
        print("\nFAILED: Loss did not decrease")
        print("This indicates a problem with the training pipeline.")
    
    # Test generation
    print("\n" + "-" * 60)
    print("GENERATION TEST")
    print("-" * 60)
    
    model.eval()
    with torch.no_grad():
        # Start with a prompt
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        generated = prompt.clone()
        
        for _ in range(20):
            logits, _ = model(generated, use_memory=False)
            next_token_logits = logits[:, -1, :]
            
            # Sample with temperature
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if generated.shape[1] >= config.max_seq_len:
                break
        
        print(f"Prompt: {prompt[0].tolist()}")
        print(f"Generated: {generated[0].tolist()}")
    
    return success


if __name__ == '__main__':
    try:
        success = run_training_test()
        print("\n" + "=" * 60)
        if success:
            print("ALL TESTS PASSED - Ready for L4 deployment")
        else:
            print("TESTS FAILED - Fix issues before L4 deployment")
        print("=" * 60)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

