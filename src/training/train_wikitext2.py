"""
Production Training Script for Hippocampal Transformer on Wikitext-2

Features:
- Real dataset loading (Wikitext-2)
- GPT-2 tokenizer
- Perplexity evaluation
- Model checkpointing
- Wake/Sleep phase training
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import dataclass
import time
from typing import Iterator, Tuple
import math

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
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 256
    intermediate_size: int = 2048
    
    # Hippocampal Config
    theta_frequency: float = 8.0
    gamma_frequency: float = 40.0
    n_place_cells: int = 1000
    
    # Training Config
    batch_size: int = 16
    lr: float = 3e-4
    epochs: int = 3
    sleep_interval: int = 500
    sleep_steps: int = 20
    replay_buffer_size: int = 10000
    ewc_lambda: float = 0.4
    
    # Eval Config
    eval_interval: int = 100
    save_interval: int = 500

def load_wikitext2():
    """
    Load Wikitext-2 dataset using datasets library.
    Falls back to synthetic data if unavailable.
    """
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
        
        print("Loading Wikitext-2 dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        return dataset, tokenizer
    except ImportError:
        print("⚠️  datasets/transformers not installed. Using synthetic data.")
        print("Install with: pip install datasets transformers")
        return None, None

def tokenize_batch(texts, tokenizer, max_length):
    """Tokenize a batch of texts."""
    return tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    
    with torch.no_grad():
        for batch_idx, (input_ids, labels, prosody) in enumerate(dataloader):
            if batch_idx >= 20:  # Limit eval batches
                break
                
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            prosody = prosody.to(device)
            
            logits, _ = model(input_ids, prosody=prosody)
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, config.vocab_size),
                labels.view(-1)
            )
            
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    model.train()
    return avg_loss, perplexity

def main():
    print("🧠 Hippocampal Transformer - Wikitext-2 Training")
    print("=" * 60)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    dataset, tokenizer = load_wikitext2()
    
    if dataset is None:
        print("⚠️  Cannot proceed without dataset. Exiting.")
        return
    
    # Initialize model
    print("\nInitializing model...")
    hippocampus = HippocampalFormation(
        config.embedding_dim,
        config.n_place_cells,
        100,
        200
    )
    
    model = HippocampalTransformer(config, hippocampus).to(device)
    trainer = HippocampalTransformerTrainer(model, config, hippocampus)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\n🚀 Starting Training")
    print(f"Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    print(f"Sleep Interval: {config.sleep_interval} steps")
    
    global_step = 0
    best_perplexity = float('inf')
    
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"{'='*60}")
        
        train_loader = create_dataloader(dataset, tokenizer, config, 'train')
        
        for batch_idx, (input_ids, labels, prosody) in enumerate(train_loader):
            global_step += 1
            trainer.step_counter()
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            prosody = prosody.to(device)
            
            if trainer.phase == "wake":
                optimizer.zero_grad()
                
                logits, place_activity = model(input_ids, prosody=prosody)
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, config.vocab_size),
                    labels.view(-1)
                )
                
                trainer.replay_buffer.add(input_ids, labels, loss.item())
                
                if trainer.ewc.fisher:
                    loss += trainer.ewc.penalty(model)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                if global_step % 50 == 0:
                    print(f"Step {global_step} [Wake] Loss: {loss.item():.4f}")
                
                # Evaluation
                if global_step % config.eval_interval == 0:
                    eval_loader = create_dataloader(dataset, tokenizer, config, 'validation')
                    eval_loss, perplexity = evaluate(model, eval_loader, config, device)
                    print(f"  📊 Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                    
                    if perplexity < best_perplexity:
                        best_perplexity = perplexity
                        torch.save(model.state_dict(), 'hippocampal_best.pt')
                        print(f"  ✅ New best model saved! Perplexity: {perplexity:.2f}")
                
                # Save checkpoint
                if global_step % config.save_interval == 0:
                    torch.save({
                        'step': global_step,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                    }, f'checkpoint_step_{global_step}.pt')
                
            elif trainer.phase == "sleep":
                print(f"💤 Sleep Phase (Step {global_step})")
                
                if not trainer.ewc.fisher:
                    print("  Computing Fisher Information...")
                    mock_loader = []
                    if len(trainer.replay_buffer) > 0:
                        batch_data = trainer.replay_buffer.sample(10)
                        for item in batch_data:
                            mock_loader.append((
                                item[0].unsqueeze(0).to(device),
                                item[1].unsqueeze(0).to(device)
                            ))
                        trainer.ewc.compute_fisher(mock_loader, device=device)
                
                replay_losses = []
                for _ in range(config.sleep_steps):
                    optimizer.zero_grad()
                    loss = trainer.train_step_sleep()
                    if loss is not None:
                        loss.backward()
                        optimizer.step()
                        replay_losses.append(loss.item())
                
                avg_replay = sum(replay_losses)/len(replay_losses) if replay_losses else 0
                print(f"  Replay Loss: {avg_replay:.4f}")
                
                trainer.phase = "wake"
    
    print(f"\n✅ Training Complete!")
    print(f"Best Perplexity: {best_perplexity:.2f}")
    print(f"Replay Buffer Size: {len(trainer.replay_buffer)}")

if __name__ == "__main__":
    main()
