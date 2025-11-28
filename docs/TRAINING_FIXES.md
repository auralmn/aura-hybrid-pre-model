# Training Fixes Document

This document identifies critical issues in the Aura training pipeline and their solutions.

---

## Part 1: Hippocampal Transformer Fixes

### Critical Issues Found

### 1. Duplicate LayerNorm + Dropout in Forward Pass

**File**: `src/core/language_zone/hippocampal_transformer.py`

**Problem**: The forward pass applied `layer_norm` and `dropout` twice consecutively, which:
- Destroys gradient flow by over-normalizing
- Applies dropout twice (effective dropout rate becomes `1 - (1-p)^2`)
- Causes training instability

**Fix Applied**: Removed duplicate `layer_norm` and `dropout` calls.

```python
# BEFORE (broken)
hidden_states = self.layer_norm(hidden_states)
hidden_states = self.dropout(hidden_states)
hidden_states = self.layer_norm(hidden_states)  # DUPLICATE
hidden_states = self.dropout(hidden_states)     # DUPLICATE

# AFTER (fixed)
hidden_states = self.layer_norm(hidden_states)
hidden_states = self.dropout(hidden_states)
```

### 2. Weight Tying Disabled

**File**: `src/core/language_zone/hippocampal_transformer.py`

**Problem**: The output head and embedding weights were not tied. This:
- Doubles the number of parameters for vocabulary projection
- Prevents the model from learning consistent token representations
- Is a known cause of poor generalization in language models

**Fix Applied**: Enabled weight tying.

```python
# BEFORE (broken)
self.output_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
# self.output_head.weight = self.semantic_encoder.token_embedding.weight  # COMMENTED OUT

# AFTER (fixed)
self.output_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
self.output_head.weight = self.semantic_encoder.token_embedding.weight  # ENABLED
```

### 3. Embedding Initialization

**File**: `src/core/language_zone/place_cell_encoder.py`

**Problem**: Default PyTorch embedding initialization uses `N(0, 1)`, which is too large for transformers and causes gradient explosion early in training.

**Fix Applied**: Initialize with `N(0, 0.02)` following GPT-2/BERT conventions.

```python
self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
```

### 4. Place Cell Residual Scaling

**File**: `src/core/language_zone/place_cell_encoder.py`

**Problem**: The place cell reconstruction was added 1:1 to the token embeddings. Since the reconstruction comes from a sparse representation through two linear layers, it can have very different magnitude, causing:
- Embedding signal to be overwhelmed
- Gradient instability
- Loss of original token information

**Fix Applied**: Scale down the place cell contribution.

```python
# BEFORE (broken)
semantic_embedding = reconstructed + token_embeds

# AFTER (fixed)
semantic_embedding = token_embeds + 0.1 * reconstructed
```

### 5. Memory Gate Broadcasting Error

**File**: `src/core/language_zone/hippocampal_attention.py`

**Problem**: The memory gate tensor was incorrectly reshaped, causing dimension mismatch errors during inference with short sequences.

**Fix Applied**: Corrected the broadcasting shape from `[B, 1, 1, L]` to `[B, 1, L, 1]`.

```python
# BEFORE (broken)
mem_scale = 1.0 + (memory_weight.transpose(1, 2).unsqueeze(1) * 0.5)

# AFTER (fixed)
mem_scale = 1.0 + (memory_weight.transpose(1, 2).unsqueeze(-1) * 0.5)
```

### 6. Positional Encoding Instability During Generation

**File**: `src/core/language_zone/theta_gamma_encoding.py`

**Problem**: The theta-gamma positional encoding normalized positions by the current sequence length. During autoregressive generation, this caused positions to "stretch" as new tokens were added, making the model see different positional encodings for the same absolute position.

**Fix Applied**: Use fixed `max_seq_len` for normalization.

```python
# BEFORE (broken)
stable_length = seq_length  # Changes during generation!

# AFTER (fixed)
stable_length = self.max_seq_len  # Fixed coordinate system
```

## Training Loop Issues

### 7. Prosody Not Passed in Training Script

**File**: `src/training/hippocampal_trainer.py`

**Problem**: The `train_step_wake` method calls `self.model(input_ids)` without passing `prosody`, even though the batch contains prosody features.

**Recommendation**: Update the forward call:

```python
# BEFORE
output = self.model(input_ids)

# AFTER
output = self.model(input_ids, prosody=prosody, use_memory=True)
```

### 8. EWC Fisher Computation Without Prosody

**File**: `src/training/hippocampal_trainer.py`

**Problem**: The `EWCConsolidator.compute_fisher` method calls the model without prosody, which may cause shape mismatches or suboptimal Fisher estimation.

## Verification Steps

After applying fixes, verify training is working:

1. **Check Loss Decrease**: Loss should decrease from ~10.3 (random) to <3.0 within 1000 steps
2. **Check Gradient Norms**: Should be stable between 0.1-10.0, not exploding or vanishing
3. **Check Generation**: After 5000+ steps, generation should produce coherent text fragments

## Files Modified (Part 1)

- `src/core/language_zone/hippocampal_transformer.py`
- `src/core/language_zone/place_cell_encoder.py`
- `src/core/language_zone/hippocampal_attention.py`
- `src/core/language_zone/theta_gamma_encoding.py`

---

## Part 2: Core Module Integration Fixes

### 9. LiquidCell dtype Mismatch

**File**: `src/core/liquid_moe.py`

**Problem**: Hidden state tensor was created without matching input dtype, causing errors with mixed precision training.

**Fix Applied**: Propagate input dtype to hidden state.

```python
# BEFORE
h_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device)

# AFTER
h_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
```

### 10. LiquidMoERouter top_k Exceeds num_experts

**File**: `src/core/liquid_moe.py`

**Problem**: If `top_k > num_experts`, `torch.topk` would fail.

**Fix Applied**: Clamp top_k to available experts.

```python
k = min(self.top_k, self.num_experts)
topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
```

### 11. ContinuousToSpikeBridge Reshape Error

**File**: `src/core/language_zone/full_language_zone.py`, `src/core/language_zone/moe_language_zone.py`

**Problem**: The reshape assumed a fixed number of timesteps, but `ContinuousToSpikeBridge` can return variable shapes.

**Fix Applied**: Dynamically handle timestep dimension.

```python
# BEFORE
spikes_moe = spikes_moe.view(batch, seq, -1, self.hidden_dim).mean(dim=2)

# AFTER
num_timesteps = spikes_moe.shape[1] if spikes_moe.dim() == 3 else 1
if spikes_moe.dim() == 2:
    spikes_moe = spikes_moe.unsqueeze(1)
spikes_moe = spikes_moe.view(batch, seq, num_timesteps, self.hidden_dim).mean(dim=2)
```

### 12. GIFNeuron dtype Propagation

**File**: `src/core/language_zone/gif_neuron.py`

**Problem**: State tensors created without matching input dtype.

**Fix Applied**: Use input dtype for state initialization.

```python
v = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
theta = torch.full((batch_size, self.hidden_dim), self.threshold, device=device, dtype=dtype)
```

### 13. SpikeBridge dtype Consistency

**File**: `src/core/language_zone/spike_bridge.py`

**Problem**: Random tensors and outputs created without matching input dtype.

**Fix Applied**: Propagate dtype throughout bridge operations.

### 14. NaturalBrain Residual Scaling

**File**: `src/core/natural_brain.py`

**Problem**: Basal ganglia output added 1:1 to embeddings, potentially overwhelming the signal.

**Fix Applied**: Scale down the residual contribution.

```python
# BEFORE
output = x + final_output if final_output is not None else x

# AFTER
if final_output is not None:
    output = x + 0.1 * final_output
else:
    output = x
```

### 15. Thalamus Attention Gain dtype

**File**: `src/core/thalamus.py`

**Problem**: Attention gain tensor created with default dtype, not matching input.

**Fix Applied**: Create tensor with proper dtype.

```python
attn_gain = torch.full((batch_size, 1), arousal_val, device=x.device, dtype=x.dtype)
```

### 16. BasalGanglia Type Safety

**File**: `src/core/basal_ganglia.py`

**Problem**: Accumulation used `0.0` as initial value, causing type issues.

**Fix Applied**: Proper tensor accumulation with type checking.

### 17. HippocampalFormation Buffer Registration

**File**: `src/core/hippocampal.py`

**Problem**: `k_const` was a plain tensor, not a registered buffer, causing device mismatch.

**Fix Applied**: Register as buffer.

```python
self.register_buffer('k_const', 4 * torch.pi / torch.sqrt(torch.tensor(3.0, device=self.device)))
```

---

## Files Modified (Part 2)

- `src/core/liquid_moe.py`
- `src/core/language_zone/full_language_zone.py`
- `src/core/language_zone/moe_language_zone.py`
- `src/core/language_zone/gif_neuron.py`
- `src/core/language_zone/spike_bridge.py`
- `src/core/natural_brain.py`
- `src/core/thalamus.py`
- `src/core/basal_ganglia.py`
- `src/core/hippocampal.py`

