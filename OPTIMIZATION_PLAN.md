# LLM Training Optimization Plan for Colab (L4/A100)

## Overview
Optimize the AURA codebase (excluding LiquidMoe) to enable efficient LLM training on Colab GPU instances (L4 24GB or A100 40GB/80GB) using vocab_src mixed with HuggingFace datasets and optional OpenRouter embeddings.

**Learning Architecture**:
- **STDP Learning**: For text-based learning from vocab_src (token-level spike-timing dependent plasticity)
- **Hebbian Learning (OjaLayer)**: For emotion/tone/valence extraction (unsupervised feature learning with dynamic component growth)

**Integration Point**: `src/services/continuous_learning.py` processes vocab_src text files and routes to brain zones.

## Working Components from Notebook (INTEGRATE THESE FIRST)

### Already Working Well in Notebook (STDP + Hebbian/SNN Compatible):
1. **Smart CUDA Cache Management** - `maybe_empty_cuda_cache()` only clears when VRAM < 12% (prevents thrashing)
2. **ArrayPool** - Memory pool for array reuse, reduces allocations significantly (critical for SNN)
3. **OptimizedWhitener** - Uses in-place operations (np.subtract, np.multiply with `out=` parameter)
4. **DataLoader** - Already using PyTorch DataLoader with batch_size=256/1024
5. **STDP Learning** - Token-based STDP in attention mechanism (`use_stdp=True`, `stdp_alpha=0.2`) for text learning
6. **Hebbian Learning (OjaLayer)** - Unsupervised feature learning for emotion/tone/valence, grows dynamically
7. **Gradient Clipping** - `torch.nn.utils.clip_grad_norm_(max_norm=1.0)` (for backprop components)
8. **Learning Rate Scheduling** - ReduceLROnPlateau with early stopping (patience-based)
9. **Colab Integration** - Google Drive mounting, HF token from userdata
10. **Sentence Transformers** - For embeddings (batch_size=256, normalize_embeddings=True)
11. **Warmup Training** - Multiple sweeps (3 sweeps, 2000 samples each) with validation
12. **Continuous Learning** - `continuous_learning.py` processes vocab_src text files via NeuromorphicProcessor

## 1. Memory Optimizations

### 1.1 Gradient Checkpointing
- **Target**: `src/core/brain.py`, `src/base/snn_processor.py`, `src/encoders/dual_layer_srffn.py`
- **Action**: Implement gradient checkpointing for large models
- **Impact**: Reduce memory by ~50% at cost of ~20% compute overhead
- **Priority**: HIGH

### 1.2 Mixed Precision Training (STDP-Compatible)
- **Target**: PyTorch-based components (not pure NumPy STDP)
- **Action**: Use `torch.cuda.amp` (Automatic Mixed Precision) for PyTorch ops
- **Impact**: 2x memory reduction, 1.5-2x speedup
- **Priority**: HIGH
- **Note**: STDP/Hebbian learning uses NumPy, AMP applies to PyTorch embedding/transformer layers

### 1.3 Smart CUDA Cache Management (From Notebook)
- **Target**: All training loops
- **Action**: Integrate `maybe_empty_cuda_cache()` - only clears when VRAM < 12%
- **Impact**: Prevents unnecessary cache clears, better performance
- **Priority**: HIGH
- **Code**: Already in notebook, extract to `src/training/memory_manager.py`

### 1.4 Batch Size Management
- **Target**: `src/encoders/pretrain_pipeline.py`
- **Action**: Dynamic batch sizing based on available GPU memory
- **Impact**: Maximize GPU utilization without OOM
- **Priority**: HIGH

### 1.5 Memory Pool Integration (From Notebook)
- **Target**: NumPy array allocations
- **Action**: Integrate `ArrayPool` class for array reuse
- **Impact**: Reduces memory allocations, faster training
- **Priority**: MEDIUM
- **Code**: Already in notebook, extract to `src/training/memory_pool.py`

## 2. Compute Optimizations

### 2.1 Data Loading Pipeline
- **Target**: `src/encoders/pretrain_pipeline.py` (`iter_texts_from_dir`, `build_embedding_dataset`)
- **Action**: Use `torch.utils.data.DataLoader` with `num_workers>0`, implement prefetching
- **Impact**: 2-3x faster data loading
- **Priority**: HIGH
- **Note**: Notebook already uses DataLoader with batch_size=256/1024, extend to vocab_src

### 2.2 Embedding Caching
- **Target**: `src/encoders/fast_hash_embedder.py`, `pretrain_pipeline.py`
- **Action**: Cache FastHash embeddings for `vocab_src` files, use sharded cache files
- **Impact**: Eliminate redundant embedding computation
- **Priority**: HIGH

### 2.3 Compile Models
- **Target**: All `nn.Module` classes
- **Action**: Use `torch.compile()` for PyTorch 2.0+
- **Impact**: 20-30% speedup
- **Priority**: MEDIUM

### 2.4 Optimized Whitener (From Notebook)
- **Target**: Whitening operations (critical for Hebbian learning)
- **Action**: Use in-place operations (np.subtract, np.multiply with `out=` parameter)
- **Impact**: Reduces temporary array allocations (important for SNN training)
- **Priority**: HIGH
- **Code**: Already in notebook as `OptimizedWhitener`, extract to `src/training/optimized_whitener.py`


## 3. Data Pipeline Optimizations

### 3.1 HuggingFace Dataset Integration
- **Target**: New module `src/encoders/hf_dataset_loader.py`
- **Action**: Create dataset loader for HF datasets, stream datasets, support mixing with `vocab_src`
- **Impact**: Access to large-scale training data
- **Priority**: HIGH

### 3.2 OpenRouter Embedding Integration
- **Target**: `src/encoders/pretrain_pipeline.py` (already has Google embedding support)
- **Action**: Add OpenRouter API integration, batch API calls, cache responses
- **Impact**: High-quality embeddings when available
- **Priority**: MEDIUM

### 3.3 Data Sharding
- **Target**: `pretrain_pipeline.py` (already has sharding support)
- **Action**: Ensure sharding works for large datasets, use manifest-based loading
- **Impact**: Handle datasets larger than GPU memory
- **Priority**: HIGH

## 4. Model Architecture Optimizations

### 4.1 Model Size Configuration
- **Target**: `src/core/brain.py` (d_model defaults)
- **Action**: Make model dimensions configurable, provide presets for L4 (smaller) vs A100 (larger)
- **Impact**: Fit models in available GPU memory
- **Priority**: HIGH

### 4.2 Layer Pruning
- **Target**: `src/base/snn_layers.py`, `src/core/brain.py`
- **Action**: Remove unused layers/components, make architecture modular
- **Impact**: Reduce model size and memory
- **Priority**: MEDIUM

## 5. Training Loop Optimizations

### 5.1 Gradient Accumulation + Clipping (From Notebook)
- **Target**: Training loops
- **Action**: Accumulate gradients over multiple micro-batches, use `clip_grad_norm_(max_norm=1.0)`
- **Impact**: Better memory efficiency, training stability
- **Priority**: HIGH
- **Code**: Already in notebook, integrate into training script

### 5.2 Checkpointing Strategy
- **Target**: Training scripts
- **Action**: Save checkpoints periodically, use lightweight checkpoints, resume training
- **Impact**: Handle Colab disconnections
- **Priority**: HIGH

### 5.3 Learning Rate Scheduling (From Notebook)
- **Target**: Training loops
- **Action**: Use ReduceLROnPlateau with early stopping (patience-based)
- **Impact**: Better convergence, fewer epochs
- **Priority**: HIGH
- **Code**: Already in notebook, integrate into training script

### 5.4 Warmup Training (From Notebook)
- **Target**: Training initialization (STDP + Hebbian)
- **Action**: Multiple warmup sweeps (3 sweeps, 2000 samples each) with validation
- **Impact**: Better MoE expert initialization, STDP weight stabilization
- **Priority**: HIGH
- **Code**: Already in notebook, integrate into training script

### 5.5 Hebbian Learning (OjaLayer) Optimization (Emotion/Tone/Valence)
- **Target**: Hebbian cortex training for emotion/tone/valence extraction
- **Action**: 
  - Optimize OjaLayer growth mechanism (dynamic component expansion based on residual error)
  - Batch Hebbian updates (process multiple emotion/tone samples)
  - Cache whitened features for multiple sweeps
  - Vectorize Oja step computation
  - Integrate with emotion/tone/valence prediction pipeline
- **Impact**: Faster emotion/tone/valence feature extraction, scales better
- **Priority**: HIGH
- **Code**: Notebook has `OjaLayer` with neurogenesis, used for emotion prediction in `process_query()`

### 5.6 STDP Integration with Continuous Learning
- **Target**: `src/services/continuous_learning.py` text processing
- **Action**: 
  - Integrate STDP learning into `_process_item()` method
  - Apply STDP updates when processing vocab_src text files
  - Track STDP weight updates per token/word
  - Batch STDP updates across multiple text items
- **Impact**: Text learning from vocab_src becomes spike-timing dependent
- **Priority**: HIGH
- **Code**: Integrate STDP from notebook into `continuous_learning.py` processing pipeline

## 6. Infrastructure Optimizations

### 6.1 Colab-Specific Adaptations (From Notebook)
- **Target**: All modules
- **Action**: Detect Colab environment, auto-configure for available GPU, handle session limits
- **Impact**: Smooth Colab experience
- **Priority**: HIGH
- **Code**: Already in notebook (Google Drive mounting, HF token), extract to utilities

### 6.2 Monitoring and Logging
- **Target**: Training scripts
- **Action**: Add GPU memory monitoring, log training metrics
- **Impact**: Better debugging and optimization
- **Priority**: MEDIUM

### 6.3 Error Handling
- **Target**: All training code
- **Action**: Handle OOM errors gracefully, auto-reduce batch size on OOM
- **Impact**: Robust training
- **Priority**: HIGH

## 7. Implementation Priority (Updated)

### Phase 1: Integrate Working Components (Week 1)
1. ✓ Remove `__pycache__` directories
2. Extract and integrate `maybe_empty_cuda_cache()` from notebook
3. Extract and integrate `ArrayPool` from notebook (critical for SNN)
4. Extract and integrate `OptimizedWhitener` from notebook (for Hebbian learning)
5. Extract and integrate STDP learning from notebook - FOR TEXT LEARNING
6. Extract and integrate OjaLayer (Hebbian learning) from notebook - FOR EMOTION/TONE/VALENCE
7. Integrate STDP into `src/services/continuous_learning.py` text processing pipeline
8. Integrate gradient clipping and LR scheduling from notebook
9. Integrate warmup training pattern from notebook
10. HF dataset integration

### Phase 2: High Impact (Week 2)
1. Mixed precision training (AMP) for PyTorch components
2. STDP learning optimizations (vectorization, batching for text processing)
3. Hebbian learning optimizations (batch processing, vectorization, caching for emotion/tone)
4. Dynamic batch sizing
5. DataLoader optimization with prefetching
6. Embedding caching for `vocab_src`
7. Model size configuration (L4 vs A100 presets)
8. Checkpointing strategy

### Phase 3: Performance (Week 3)
1. Gradient checkpointing
2. `torch.compile()` integration
3. OpenRouter embedding integration
4. Monitoring/logging

### Phase 4: Polish (Week 4)
1. Quantization (inference)
2. Sparse operations optimization
3. Documentation and examples

## 8. Expected Results

### L4 (24GB GPU)
- Model size: ~500M-1B parameters max (SNN + embeddings)
- Batch size: 4-8 (with gradient accumulation)
- Training speed: ~100-200 tokens/sec (STDP for text + Hebbian for emotion/tone)
- Memory usage: ~18-20GB GPU
- STDP updates: Batch process 32-64 text tokens
- Hebbian updates: Batch process 32-64 emotion/tone samples

### A100 (40GB GPU)
- Model size: ~1-3B parameters max (SNN + embeddings)
- Batch size: 8-16 (with gradient accumulation)
- Training speed: ~300-500 tokens/sec (STDP for text + Hebbian for emotion/tone)
- Memory usage: ~35-38GB GPU
- STDP updates: Batch process 64-128 text tokens
- Hebbian updates: Batch process 64-128 emotion/tone samples

### A100 (80GB GPU)
- Model size: ~3-7B parameters max (SNN + embeddings)
- Batch size: 16-32 (with gradient accumulation)
- Training speed: ~500-800 tokens/sec (STDP for text + Hebbian for emotion/tone)
- Memory usage: ~70-75GB GPU
- STDP updates: Batch process 128-256 text tokens
- Hebbian updates: Batch process 128-256 emotion/tone samples

## 9. Files to Modify

### High Priority
- `src/services/continuous_learning.py` - Integrate STDP for text learning from vocab_src
- `src/encoders/pretrain_pipeline.py` - Data loading, embedding caching
- `src/core/brain.py` - Model architecture, memory management
- `src/base/snn_processor.py` - Processing optimizations (emotion/tone/valence routing)
- `src/encoders/fast_hash_embedder.py` - Embedding efficiency

### Medium Priority
- `src/base/snn_layers.py` - Layer optimizations
- `src/base/neuron.py` - Neuron computation efficiency
- `src/encoders/dual_layer_srffn.py` - Feedforward optimizations
- `src/encoders/fast_event_encoder.py` - Event encoding efficiency

### New Files to Create
- `src/encoders/hf_dataset_loader.py` - HuggingFace dataset integration
- `src/encoders/openrouter_embedder.py` - OpenRouter API integration
- `src/training/train_llm.py` - Main training script (integrate notebook patterns)
- `src/training/colab_utils.py` - Colab-specific utilities (extract from notebook)
- `src/training/memory_manager.py` - GPU memory management (maybe_empty_cuda_cache + ArrayPool)
- `src/training/optimized_whitener.py` - In-place operations for whitening (from notebook)
- `src/training/memory_pool.py` - ArrayPool implementation (from notebook)
- `src/training/stdp_learning.py` - STDP learning mechanism for text (extract from notebook)
- `src/training/hebbian_layer.py` - OjaLayer Hebbian learning for emotion/tone/valence (extract from notebook)

## 10. Key Code to Extract from Notebook

### STDP Learning (Text Learning) - from AttentionBundle
```python
# Token-based STDP weight updates for text learning
if self.use_stdp:
    if self.token_weights is None:
        self.token_weights = {}
    spiked = (s_amp + s_pitch + s_bound) > 0
    for i, tok in enumerate(token_ids):
        prev = self.token_weights.get(tok, 0.0)
        delta = self.lr_plus if spiked[i] else -self.lr_minus
        new_w = np.clip(prev + delta, self.w_min, self.w_max)
        self.token_weights[tok] = float(new_w)
    # Apply STDP modulation to salience
    stdp_mod = np.ones_like(sal, dtype=np.float64)
    for i, tok in enumerate(token_ids):
        w_tok = self.token_weights.get(tok, 0.0)
        stdp_mod[i] = max(0.0, 1.0 + self.stdp_alpha * w_tok)
    sal = sal * stdp_mod
```
- **Purpose**: Text-based learning from vocab_src
- **Integration**: Add to `src/services/continuous_learning.py` `_process_item()` method
- **Extract to**: `src/training/stdp_learning.py`

### OjaLayer (Hebbian Learning) - Emotion/Tone/Valence Extraction
```python
class OjaLayer:
    """Unsupervised Hebbian learning with dynamic component growth for emotion/tone/valence"""
    def step(self, xw: np.ndarray) -> OjaStepOut:
        # Oja's rule: W = W + eta * (x * y - y^2 * W)
        # where y = W^T * x
        y = self.W.T @ xw
        # Update weights
        self.W += self.eta * (np.outer(xw, y) - np.outer(self.W, y**2))
        # Dynamic growth: add new component if residual error is high
        if residual_ema > threshold:
            self._grow_component()  # Neurogenesis
        return OjaStepOut(y=y, residual_ema=residual_ema, grew=grew)
```

Key features:
- Unsupervised feature learning with Oja's rule
- Used for emotion/tone/valence extraction (not text learning)
- Dynamic component expansion (neurogenesis) based on residual error
- Whitened input processing (requires OptimizedWhitener)
- Extract to `src/training/hebbian_layer.py`
- **Integration**: Used in notebook's `process_query()` for emotion prediction

### maybe_empty_cuda_cache()
```python
def maybe_empty_cuda_cache(reason: str = "", min_free_ratio: float = 0.12) -> None:
    """Only clear CUDA cache if free VRAM drops below a ratio to avoid thrashing."""
    if not torch.cuda.is_available():
        return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_ratio = free_bytes / float(total_bytes)
        if free_ratio < min_free_ratio:
            tag = f" ({reason})" if reason else ""
            print(f"🔄 Clearing CUDA cache{tag}. Free VRAM ratio: {free_ratio:.3f}")
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"⚠️ Unable to query CUDA memory for cleanup: {exc}")
```

### ArrayPool (simplified)
- Thread-safe memory pool for NumPy arrays
- Reduces allocations by reusing arrays of same shape/dtype
- Max pool size: 512MB (configurable)

### OptimizedWhitener
- Uses in-place operations (np.subtract, np.multiply with `out=` parameter)
- Reduces temporary array allocations
- Momentum-based updates

### Training Pattern
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(max_norm=1.0)`
- LR scheduling: `ReduceLROnPlateau` with patience
- Early stopping: Track `epochs_no_improve`, break when >= patience
- Warmup: 3 sweeps, 2000 samples each, with validation

## 11. Testing Strategy

1. Unit tests for each optimization
2. Memory profiling (torch.profiler)
3. Benchmark on L4 and A100
4. Validate training convergence
5. Test checkpoint/resume functionality
6. Compare with/without Unsloth
7. Compare with/without ArrayPool

