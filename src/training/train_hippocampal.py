"""
E2E Training Script for Hippocampal Transformer

Runs a complete training loop with:
1. Model Initialization
2. Synthetic Data Generation
3. Wake Phase Training
4. Sleep Phase Consolidation
5. Metrics Logging
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import dataclass
import time

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.core.hippocampal import HippocampalFormation
from src.core.language_zone.hippocampal_transformer import HippocampalTransformer
from src.training.hippocampal_trainer import HippocampalTransformerTrainer

@dataclass
class Config:
    # Model Config
    vocab_size: int = 1000
    embedding_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 128
    intermediate_size: int = 1024
    
    # Hippocampal Config
    theta_frequency: float = 8.0
    gamma_frequency: float = 40.0
    n_place_cells: int = 500
    
    # Training Config
    batch_size: int = 8
    lr: float = 1e-3
    steps: int = 50
    sleep_interval: int = 20
    sleep_steps: int = 5
    replay_buffer_size: int = 1000
    ewc_lambda: float = 0.1

def generate_synthetic_data(config, steps):
    """Generate synthetic sequence data."""
    for _ in range(steps):
        input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len))
        labels = torch.roll(input_ids, -1, dims=1)
        prosody = torch.rand(config.batch_size, config.max_seq_len, 4)
        yield input_ids, labels, prosody

def main():
    print("🧠 Initializing Hippocampal Transformer E2E Training...")
    config = Config()
    
    # 1. Initialize Components
    print("Initializing Hippocampus...")
    hippocampus = HippocampalFormation(
        config.embedding_dim,
        config.n_place_cells,
        50,
        100
    )
    
    print("Initializing Model...")
    model = HippocampalTransformer(config, hippocampus)
    
    print("Initializing Trainer...")
    trainer = HippocampalTransformerTrainer(model, config, hippocampus)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    # 2. Training Loop
    print("\n🚀 Starting Training Loop")
    print(f"Config: Steps={config.steps}, Sleep Interval={config.sleep_interval}")
    
    start_time = time.time()
    losses = []
    
    data_iter = generate_synthetic_data(config, config.steps)
    
    for step, (input_ids, labels, prosody) in enumerate(data_iter):
        trainer.step_counter()
        
        if trainer.phase == "wake":
            optimizer.zero_grad()
            
            logits, place_activity = model(input_ids, prosody=prosody)
            loss = nn.CrossEntropyLoss()(logits.view(-1, config.vocab_size), labels.view(-1))
            
            trainer.replay_buffer.add(input_ids, labels, loss.item())
            
            if trainer.ewc.fisher:
                ewc_loss = trainer.ewc.penalty(model)
                loss += ewc_loss
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"Step {step+1}/{config.steps} [Wake] Loss: {loss.item():.4f}")
            
            if step % 5 == 0:
                hippocampus.create_episodic_memory(
                    memory_id=f"mem_{step}",
                    event_id=f"train_step_{step}",
                    features=place_activity.detach().mean(dim=0).cpu().numpy(),
                    associated_experts=None
                )
                
        elif trainer.phase == "sleep":
            print(f"💤 Sleeping... (Replay & Consolidation)")
            
            if not trainer.ewc.fisher:
                print("  Computing Fisher Information...")
                mock_loader = []
                if len(trainer.replay_buffer) > 0:
                    batch_data = trainer.replay_buffer.sample(10)
                    for item in batch_data:
                        # Add batch dimension [seq_len] -> [1, seq_len]
                        mock_loader.append((item[0].unsqueeze(0), item[1].unsqueeze(0)))
                    trainer.ewc.compute_fisher(mock_loader, device=input_ids.device)
            
            replay_losses = []
            for _ in range(config.sleep_steps):
                optimizer.zero_grad()
                loss = trainer.train_step_sleep()
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    replay_losses.append(loss.item())
            
            avg_replay_loss = sum(replay_losses)/len(replay_losses) if replay_losses else 0
            print(f"  Replay Loss: {avg_replay_loss:.4f}")
            
            trainer.phase = "wake"
            
    total_time = time.time() - start_time
    print(f"\n✅ Training Complete in {total_time:.2f}s")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Memories Stored: {len(hippocampus.episodic_memories)}")
    print(f"Replay Buffer Size: {len(trainer.replay_buffer)}")

if __name__ == "__main__":
    main()
