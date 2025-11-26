"""
Hippocampal Transformer Training Script
Follows patterns from training_recipes.md

Copy each section into separate Colab cells for easy execution.
"""

# ============================================================
# CELL 1: Setup & Dependencies
# ============================================================
!git clone https://github.com/auralmn/aura-hybrid-pre-model.git
!git checkout master
!cd aura-hybrid-pre-model
!pip install -e .

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2Tokenizer
from dataclasses import dataclass
import math
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# CELL 2: Configuration (Recipe 1: Basic Training)
# ============================================================
@dataclass
class Config:
    # Model
    vocab_size: int = 50257
    embedding_dim: int = 384
    num_layers: int = 4
    num_heads: int = 6
    dropout: float = 0.1
    max_seq_len: int = 128
    intermediate_size: int = 1536
    
    # Hippocampal
    theta_frequency: float = 8.0
    gamma_frequency: float = 40.0
    n_place_cells: int = 800
    
    # Training
    batch_size: int = 32
    lr: float = 3e-4
    max_steps: int = 2000
    sleep_interval: int = 500  # Sleep every 500 steps
    sleep_steps: int = 10      # 10 replay iterations
    eval_interval: int = 100
    ewc_lambda: float = 0.4

config = Config()
print(f"Config created. Sleep every {config.sleep_interval} steps.")

# ============================================================
# CELL 3: Load Dataset
# ============================================================
print("Loading Wikitext-2...")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print(f"Train: {len(dataset['train'])} samples")
print(f"Validation: {len(dataset['validation'])} samples")

def create_batches(dataset, tokenizer, config, split='train', max_batches=None):
    """Recipe-compliant data loader"""
    texts = [item['text'] for item in dataset[split] if len(item['text'].strip()) > 10]
    
    batch_count = 0
    for i in range(0, len(texts), config.batch_size):
        if max_batches and batch_count >= max_batches:
            break
            
        batch_texts = texts[i:i+config.batch_size]
        if len(batch_texts) < config.batch_size:
            continue
        
        encoded = tokenizer(
            batch_texts,
            max_length=config.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids']
        labels = input_ids.clone()
        prosody = torch.rand(input_ids.size(0), input_ids.size(1), 4)
        
        batch_count += 1
        yield input_ids, labels, prosody

# ============================================================
# CELL 4: Initialize Model Components
# ============================================================
from src.core.hippocampal import HippocampalFormation
from src.core.language_zone.hippocampal_transformer import HippocampalTransformer
from src.training.hippocampal_trainer import HippocampalTransformerTrainer

print("Initializing Hippocampal Formation...")
hippocampus = HippocampalFormation(
    config.embedding_dim,
    config.n_place_cells,
    50,   # time cells
    100   # grid cells
)

print("Initializing HippocampalTransformer...")
model = HippocampalTransformer(config, hippocampus).to(device)

print("Initializing Trainer...")
trainer = HippocampalTransformerTrainer(model, config, hippocampus)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================================================
# CELL 5: Training Loop (Recipe 1 + 3)
# ============================================================
losses = []
eval_losses = []
perplexities = []
steps = []

print("\\nStarting training...")
print(f"Max steps: {config.max_steps}")
print(f"Sleep interval: {config.sleep_interval}")
print("-" * 60)

global_step = 0
train_gen = create_batches(dataset, tokenizer, config, 'train', max_batches=config.max_steps)
pbar = tqdm(total=config.max_steps, desc="Training")

for input_ids, labels, prosody in train_gen:
    global_step += 1
    trainer.step_counter()
    
    # Move to device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    prosody = prosody.to(device)
    
    # ===== WAKE PHASE: Training (Recipe 1) =====
    if trainer.phase == "wake":
        optimizer.zero_grad()
        
        # Forward pass
        logits, place_activity = model(input_ids, prosody=prosody)
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, config.vocab_size),
            labels.view(-1)
        )
        
        # Store in replay buffer (Recipe 1)
        trainer.replay_buffer.add(input_ids, labels, loss.item())
        
        # EWC penalty (Recipe 2: Multi-task learning)
        if trainer.ewc.fisher:
            loss += trainer.ewc.penalty(model)
        
        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track
        losses.append(loss.item())
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'phase': 'Wake',
            'buffer': len(trainer.replay_buffer)
        })
        
        # Create episodic memory (Recipe 3: Long-context)
        if global_step % 10 == 0:
            hippocampus.create_episodic_memory(
                memory_id=f"step_{global_step}",
                event_id=f"train_{global_step}",
                features=place_activity.detach().mean(dim=0).cpu().numpy(),
                associated_experts=None
            )
        
        # Evaluation
        if global_step % config.eval_interval == 0:
            model.eval()
            eval_loss = 0
            eval_batches = 0
            
            with torch.no_grad():
                for eval_input, eval_labels, eval_prosody in create_batches(
                    dataset, tokenizer, config, 'validation', max_batches=20
                ):
                    eval_input = eval_input.to(device)
                    eval_labels = eval_labels.to(device)
                    eval_prosody = eval_prosody.to(device)
                    
                    eval_logits, _ = model(eval_input, prosody=eval_prosody)
                    batch_loss = nn.CrossEntropyLoss()(
                        eval_logits.view(-1, config.vocab_size),
                        eval_labels.view(-1)
                    )
                    eval_loss += batch_loss.item()
                    eval_batches += 1
            
            avg_eval_loss = eval_loss / eval_batches
            perplexity = math.exp(avg_eval_loss)
            
            eval_losses.append(avg_eval_loss)
            perplexities.append(perplexity)
            steps.append(global_step)
            
            print(f"\\n📊 Step {global_step}: Eval Loss={avg_eval_loss:.4f}, Perplexity={perplexity:.2f}")
            model.train()
    
    # ===== SLEEP PHASE: Consolidation (Recipe 1) =====
    elif trainer.phase == "sleep":
        pbar.set_postfix({'phase': '💤 Sleep'})
        print(f"\\n🌙 Sleep Phase at step {global_step}")
        
        # Compute Fisher Information (Recipe 2)
        if not trainer.ewc.fisher and len(trainer.replay_buffer) > 0:
            print("  Computing Fisher Information...")
            mock_loader = []
            samples = trainer.replay_buffer.sample(10)
            for item in samples:
                mock_loader.append((
                    item[0].unsqueeze(0).to(device),
                    item[1].unsqueeze(0).to(device)
                ))
            trainer.ewc.compute_fisher(mock_loader, device=device)
            print("  ✅ Fisher computed!")
        
        # Memory replay (Recipe 1: Forward + Backward replay)
        print(f"  Replaying {config.sleep_steps} batches...")
        replay_losses = []
        for _ in range(config.sleep_steps):
            optimizer.zero_grad()
            loss = trainer.train_step_sleep()
            if loss is not None:
                loss.backward()
                optimizer.step()
                replay_losses.append(loss.item())
        
        avg_replay = sum(replay_losses) / len(replay_losses) if replay_losses else 0
        print(f"  Replay Loss: {avg_replay:.4f}")
        print(f"  Buffer size: {len(trainer.replay_buffer)}")
        
        # Wake up
        trainer.phase = "wake"
    
    pbar.update(1)
    
    if global_step >= config.max_steps:
        break

pbar.close()
print("\\n✅ Training complete!")
print(f"Consolidation cycles: {global_step // config.sleep_interval}")
print(f"Episodic memories: {len(hippocampus.episodic_memories)}")

# ============================================================
# CELL 6: Visualize Results
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Training loss
axes[0].plot(losses, alpha=0.3, label='Raw')
window = 50
if len(losses) > window:
    smoothed = [sum(losses[max(0,i-window):i+1])/min(i+1,window) for i in range(len(losses))]
    axes[0].plot(smoothed, linewidth=2, label='Smoothed')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Perplexity
axes[1].plot(steps, perplexities, marker='o', linewidth=2)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Perplexity')
axes[1].set_title('Validation Perplexity')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\\nFinal Metrics:")
print(f"  Best Perplexity: {min(perplexities):.2f}")
print(f"  Final Perplexity: {perplexities[-1]:.2f}")
print(f"  Replay Buffer: {len(trainer.replay_buffer)} samples")
print(f"  Episodic Memories: {len(hippocampus.episodic_memories)}")

# ============================================================
# CELL 7: Save Model
# ============================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'perplexity': perplexities[-1] if perplexities else None,
    'hippocampus_state': {
        'n_memories': len(hippocampus.episodic_memories),
        'theta_phase': hippocampus.theta_phase
    }
}, 'hippocampal_transformer.pt')

print("Model saved to hippocampal_transformer.pt")
print("Download with: files.download('hippocampal_transformer.pt')")

# ============================================================
# CELL 8: Memory Inspection (Recipe 7)
# ============================================================
print("\\n" + "="*60)
print("HIPPOCAMPAL MEMORY INSPECTION")
print("="*60)

print(f"\\nTotal episodic memories: {len(hippocampus.episodic_memories)}")
print(f"Theta phase: {hippocampus.theta_phase:.2f} rad")
print(f"Place cells: {len(hippocampus.place_cells)}")
print(f"Time cells: {len(hippocampus.time_cells)}")
print(f"Grid cells: {len(hippocampus.grid_cells)}")

# Show recent memories
if hippocampus.episodic_memories:
    print("\\nRecent memories:")
    recent = list(hippocampus.episodic_memories.items())[-5:]
    for mem_id, memory in recent:
        print(f"  {mem_id}: strength={memory.strength:.3f}")
