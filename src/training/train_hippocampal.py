"""
E2E Training Script for Hippocampal Transformer (Optimized)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import dataclass
import time

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from core.hippocampal import HippocampalFormation
from core.language_zone.hippocampal_transformer import HippocampalTransformer
from training.hippocampal_trainer import HippocampalTransformerTrainer

@dataclass
class Config:
    # Model Config
    vocab_size: int = 32000
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 16
    dropout: float = 0.1
    max_seq_len: int = 512
    intermediate_size: int = 4096
    
    # Hippocampal Config
    theta_frequency: float = 8.0
    gamma_frequency: float = 40.0
    n_place_cells: int = 2000
    
    # Training Config
    batch_size: int = 32  # Increased due to Gradient Checkpointing/Flash Attn
    lr: float = 3e-4
    steps: int = 1000
    sleep_interval: int = 200
    sleep_steps: int = 25
    replay_buffer_size: int = 50000
    ewc_lambda: float = 0.4
    
    # Loss Config (New)
    label_smoothing: float = 0.1
    entropy_lambda: float = 0.05
    sparsity_lambda: float = 0.02

def generate_synthetic_data(config, steps, device):
    """Generate synthetic data directly on GPU"""
    for _ in range(steps):
        input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len), device=device)
        labels = torch.roll(input_ids, -1, dims=1)
        # Random prosody features [0, 1]
        prosody = torch.rand(config.batch_size, config.max_seq_len, 4, device=device)
        yield input_ids, labels, prosody

def main():
    print("🧠 Initializing Optimized Hippocampal Training...")
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for Ampere/Hopper
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    config = Config()
    
    # 1. Initialize Components (Move to Device immediately)
    print("Initializing GPU-Native Hippocampus...")
    hippocampus = HippocampalFormation(
        spatial_dimensions=2,
        n_place_cells=config.n_place_cells,
        device=str(device)
    ).to(device)
    
    print("Initializing Model...")
    model = HippocampalTransformer(config, hippocampus).to(device)
    
    # Optional: Compile model (requires PyTorch 2.0+)
    # try:
    #     model = torch.compile(model)
    #     print("✅ Model compiled with torch.compile")
    # except Exception as e:
    #     print(f"⚠️ Compilation skipped: {e}")
    
    print("Initializing Trainer...")
    trainer = HippocampalTransformerTrainer(model, config, hippocampus)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.1)
    
    # 2. Training Loop
    print("\n🚀 Starting Training Loop")
    start_time = time.time()
    losses = []
    
    data_iter = generate_synthetic_data(config, config.steps, device)
    
    model.train()
    
    for step, (input_ids, labels, prosody) in enumerate(data_iter):
        trainer.step_counter()
        
        if trainer.phase == "wake":
            optimizer.zero_grad()
            
            # Forward pass (AutoCast for Mixed Precision)
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                loss = trainer.train_step_wake((input_ids, labels, prosody))
            
            # Scale loss not needed unless using GradScaler (simplified here)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            losses.append(loss.item())
            if step % 10 == 0:
                print(f"Step {step+1}/{config.steps} [Wake] Loss: {loss.item():.4f}")
            
            # Periodic Memory Creation (every 10 steps)
            if step % 10 == 0:
                # Get place cell activity from the last forward pass
                # Note: Real implementation might need to return this from train_step_wake
                # or hook into the model. For simplicity, we assume the hippocampus 
                # state was updated during the forward pass.
                
                # Create memory snapshot
                with torch.no_grad():
                    # Just an example feature vector
                    feats = torch.randn(config.embedding_dim, device=device)
                    hippocampus.create_episodic_memory(
                        memory_id=f"mem_{step}",
                        event_id=f"step_{step}",
                        features=feats
                    )
                
        elif trainer.phase == "sleep":
            print(f"💤 Sleeping... (Replay & Consolidation)")
            
            # EWC Fisher Calc (if empty)
            if not trainer.ewc.fisher and len(trainer.replay_buffer) > 0:
                print("  Computing Fisher Information...")
                # Sample small batch for Fisher
                batch = trainer.replay_buffer.sample(16)
                # Format for EWC: list of (input, label)
                mock_loader = [(b[0].unsqueeze(0).to(device), b[1].unsqueeze(0).to(device)) for b in batch]
                trainer.ewc.compute_fisher(mock_loader, device)
            
            # Replay Training
            replay_losses = []
            for _ in range(config.sleep_steps):
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                    loss = trainer.train_step_sleep()
                
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    replay_losses.append(loss.item())
            
            avg_loss = sum(replay_losses)/len(replay_losses) if replay_losses else 0
            print(f"  Replay Loss: {avg_loss:.4f}")
            
            # Natural Decay
            hippocampus.decay_memories()
            trainer.phase = "wake"
            
    total_time = time.time() - start_time
    print(f"\n✅ Training Complete in {total_time:.2f}s")
    print(f"Speed: {config.steps / total_time:.2f} steps/sec")
    print(f"Final Loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    main()