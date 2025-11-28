# Training Optimization Guide

This document provides optimization recommendations for the Aura Hippocampal Transformer.

## Architecture Optimizations

### 1. Gradient Checkpointing

For memory-constrained environments (e.g., 16GB VRAM), enable gradient checkpointing:

```python
# In HippocampalTransformer.__init__
from torch.utils.checkpoint import checkpoint

# In forward pass
for layer in self.layers:
    if self.training and self.config.use_gradient_checkpointing:
        hidden_states = checkpoint(layer, hidden_states, prosody, use_memory)
    else:
        hidden_states = layer(hidden_states, prosody=prosody, use_memory=use_memory)
```

### 2. Flash Attention

Already implemented via `F.scaled_dot_product_attention`. Ensure PyTorch 2.0+ for optimal performance.

### 3. Mixed Precision Training

Use `bfloat16` for stability on modern GPUs:

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

with autocast('cuda', dtype=torch.bfloat16):
    loss = model(input_ids, prosody=prosody)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Training Hyperparameters

### Recommended Configuration

```python
@dataclass
class OptimizedConfig:
    # Model
    vocab_size: int = 32000
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12  # Reduced from 16 for head_dim=64
    dropout: float = 0.1
    max_seq_len: int = 512
    intermediate_size: int = 3072  # 4x embedding_dim
    
    # Training
    batch_size: int = 32
    gradient_accumulation: int = 4  # Effective batch = 128
    lr: float = 1e-4  # Lower than 3e-4 for stability
    warmup_steps: int = 2000
    max_steps: int = 100000
    weight_decay: float = 0.01  # Lower than 0.1
    
    # Regularization
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
```

### Learning Rate Schedule

Use cosine decay with warmup and minimum LR floor:

```python
def lr_schedule(step, warmup_steps, max_steps, min_lr_ratio=0.1):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine
```

## Memory Optimization

### 1. Reduce Place Cell Count

For initial training, reduce `n_place_cells` from 2000 to 1000:

```python
n_place_cells: int = 1000  # Reduces memory and speeds up sparse ops
```

### 2. Disable Hippocampal Memory During Warmup

The memory system adds overhead. Disable for first 5000 steps:

```python
use_memory = global_step > 5000
logits, _ = model(input_ids, prosody=prosody, use_memory=use_memory)
```

### 3. Reduce Replay Buffer Size

For limited RAM:

```python
replay_buffer_size: int = 100000  # Reduced from 1M
```

## Data Pipeline Optimizations

### 1. Pre-tokenize Dataset

Tokenize once and cache:

```python
def preprocess_and_cache(dataset, tokenizer, cache_path):
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    
    tokenized = []
    for sample in tqdm(dataset):
        ids = tokenizer.encode(sample['text'])
        if len(ids) >= 10:
            tokenized.append(ids[:512])
    
    torch.save(tokenized, cache_path)
    return tokenized
```

### 2. Use DataLoader with Workers

```python
from torch.utils.data import DataLoader, Dataset

class TokenDataset(Dataset):
    def __init__(self, token_ids, max_len):
        self.data = token_ids
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.data[idx]
        # Pad or truncate
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids[:self.max_len])

loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
```

## Monitoring

### Key Metrics to Track

1. **Loss**: Should decrease steadily
2. **Perplexity**: `exp(loss)`, should decrease
3. **Gradient Norm**: Should be stable (0.1-10.0)
4. **Learning Rate**: Verify warmup is working
5. **Memory Usage**: Watch for OOM

### Logging Template

```python
if step % 100 == 0:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"Step {step} | Loss: {loss:.4f} | PPL: {math.exp(loss):.2f} | "
          f"Grad: {grad_norm:.2f} | LR: {scheduler.get_last_lr()[0]:.2e}")
```

## Debugging Checklist

If training is not working:

1. [ ] Verify loss decreases in first 100 steps
2. [ ] Check gradient norms are not NaN or >100
3. [ ] Verify learning rate schedule is correct
4. [ ] Check batch data is not all padding
5. [ ] Verify tokenizer matches training data
6. [ ] Check model is in `.train()` mode
7. [ ] Verify `optimizer.step()` is being called
8. [ ] Check `loss.backward()` is being called

## Quick Start Training Command

```bash
# Using the fixed training script
python src/training/train_hippocampal.py
```

## Expected Training Curve

| Steps | Loss | Perplexity |
|-------|------|------------|
| 0 | ~10.3 | ~30000 |
| 1000 | ~6.0 | ~400 |
| 5000 | ~4.0 | ~55 |
| 10000 | ~3.0 | ~20 |
| 50000 | ~2.5 | ~12 |

If loss does not decrease from ~10.3 within 100 steps, there is a fundamental issue with the training loop.

---

## NaturalBrain Integration

The `NaturalBrain` architecture integrates multiple brain-inspired modules. Here are optimization recommendations for the full system.

### Module Dependencies

```
Input -> Embedding -> LimbicSystem -> Thalamus -> Cortex (FullLanguageZone) -> BasalGanglia -> Output
                          |              |
                     Hippocampus    LiquidMoE Router
```

### Integration Considerations

#### 1. Disable SNN Components Initially

The SNN components (GIFNeuron, Synapsis) add complexity. For initial training:

```python
# In BrainZoneConfig
use_spiking: bool = False  # Start with continuous
```

#### 2. Simplify Routing

The LiquidMoE router adds dynamic routing. For debugging:

```python
# In Thalamus, bypass routing for debugging
if debug_mode:
    return {region: x for region in self.region_names}, None
```

#### 3. Limbic System Warmup

Emotional modulation can destabilize early training:

```python
if global_step < 1000:
    emotional_state = {'arousal': 0.5, 'valence': 0.0}  # Neutral
```

### Memory Considerations

| Module | Memory Usage | Optimization |
|--------|-------------|--------------|
| Hippocampus | ~100MB | Reduce max_memories |
| FullLanguageZone | ~200MB | Reduce num_experts |
| LiquidMoE | ~50MB | Reduce hidden_dim |
| SNN Experts | ~100MB | Reduce num_layers |

### Recommended Training Phases

1. **Phase 1 (Steps 0-5000)**: Train HippocampalTransformer alone
2. **Phase 2 (Steps 5000-20000)**: Add FullLanguageZone
3. **Phase 3 (Steps 20000+)**: Enable full NaturalBrain

