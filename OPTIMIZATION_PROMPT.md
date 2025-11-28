# Optimization Prompt for Google Gemini 3.0

## Progress Update (Latest Session)

### Completed Optimizations

| Category | Item | Status |
|----------|------|--------|
| **Architecture** | Duplicate LayerNorm/Dropout fix | Done |
| **Architecture** | Weight tying enabled | Done |
| **Architecture** | Embedding init (std=0.02) | Done |
| **Architecture** | Place cell residual scaling | Done |
| **Architecture** | Memory gate broadcasting | Done |
| **Architecture** | Positional encoding stability | Done |
| **Architecture** | SNN-based FFN layers | Done |
| **Architecture** | RAG memory retrieval | Done |
| **Training** | Gradient checkpointing | Done |
| **Training** | Cosine LR with warmup | Done |
| **Training** | Memory warmup period | Done |
| **Training** | Gradient clipping/monitoring | Done |
| **Training** | Optimized config presets | Done |
| **Generation** | Top-k/top-p sampling | Done |
| **Generation** | Repetition penalty | Done |
| **Core Modules** | dtype propagation fixes | Done |
| **Core Modules** | LiquidMoE fixes | Done |
| **Core Modules** | GIFNeuron fixes | Done |

### New Files Created

- `src/core/language_zone/snn_ffn.py` - SNN-based FFN with GIF neurons
- `src/core/language_zone/memory_augmented_layer.py` - RAG memory layer
- `src/core/language_zone/snn_rag_transformer.py` - Full SNN+RAG model
- `src/training/config.py` - Optimized training configurations
- `docs/TRAINING_FIXES.md` - Comprehensive bug fix documentation
- `docs/TRAINING_OPTIMIZATION.md` - Optimization guide
- `test_local_training.py` - Local training verification
- `test_snn_rag.py` - SNN+RAG integration tests

---

## Mission

Help us transform the **Aura HippocampalTransformer** from a research prototype into a **world-class neuromorphic language model** that pushes the boundaries of bio-inspired AI.

## 📋 Project Overview

### What We've Built

We've created a novel language model that integrates neuroscience-inspired hippocampal memory systems with transformer architecture:

- **Place Cell Semantic Encoding**: Sparse population coding (3% activation) mimicking hippocampal CA1/CA3 neurons
- **Grid Cell Navigation**: Hexagonal spatial patterns for structural awareness
- **Time Cell Sequences**: Temporal interval coding for event ordering
- **Episodic Memory Formation**: Real-time memory consolidation with cognitive maps
- **Theta-Gamma Coupling**: Neural oscillation-based position encoding
- **Prosody-Modulated Attention**: Emotional and rhythmic feature integration
- **Elastic Weight Consolidation (EWC)**: Catastrophic forgetting prevention

**Architecture**: 112M parameters, 12 layers, 16 attention heads, 768D embeddings, 2000 place cells

**Training Status**: Mid-training checkpoint (11,500 / 50,000 steps) on Nemotron-CC-v2 + WikiText-103

### Current Performance

| Metric | Current | Target |
|--------|---------|--------|
| Perplexity | 8-12 | < 5 |
| Memory Retrieval | 75-85% (top-5) | > 95% (top-5) |
| Inference Speed | 2-5 tok/s (CPU) | 50+ tok/s (GPU) |
| Training Speed | 66 steps/hour | 500+ steps/hour |
| Generation Quality | Moderate (repetitive) | High (diverse, coherent) |

### Key Files to Review

1. **Architecture**:
   - `src/core/language_zone/hippocampal_transformer.py` - Main model (gradient checkpointing added)
   - `src/core/language_zone/hippocampal_attention.py` - Prosody-modulated attention (broadcasting fixed)
   - `src/core/language_zone/place_cell_encoder.py` - Sparse semantic encoding (dtype/init fixed)
   - `src/core/language_zone/theta_gamma_encoding.py` - Neural oscillation encoding (stability fixed)
   - `src/core/hippocampal.py` - Hippocampal formation (memory systems)
   - `src/core/language_zone/snn_rag_transformer.py` - **NEW**: SNN + RAG integrated model
   - `src/core/language_zone/snn_ffn.py` - **NEW**: Spiking neural network FFN
   - `src/core/language_zone/memory_augmented_layer.py` - **NEW**: RAG memory layer

2. **Training**:
   - `src/training/hippocampal_trainer.py` - Training loop, EWC, consolidation (cosine LR, memory warmup added)
   - `src/training/config.py` - **NEW**: Optimized training configurations
   - `src/training/train_hippocampal.py` - Configuration and synthetic training
   - `notebookforaura.py` - Colab training script (converted from notebook)

3. **Documentation**:
   - `MODEL_CARD.md` - Comprehensive model documentation
   - `README.md` - GitHub repository overview
   - `docs/TRAINING_FIXES.md` - **NEW**: All bug fixes documented
   - `docs/TRAINING_OPTIMIZATION.md` - **NEW**: Optimization guide

## 🔬 Critical Areas Requiring Optimization

### 1. Model Architecture

**Current Issues**:
- ~~Repetitive text generation~~ **FIXED**: Multiple fixes (see TRAINING_FIXES.md)
- ~~DirectML scatter compatibility~~ **FIXED**: dtype propagation fixes
- Inference overhead (15-20% from hippocampal operations)
- Place cell sparsity hard-coded at 3% (not learned)

**Implemented Solutions**:
- Fixed duplicate LayerNorm/Dropout in forward pass
- Enabled weight tying (embedding <-> output head)
- Fixed embedding initialization (std=0.02)
- Fixed place cell residual scaling (0.1x)
- Fixed memory gate broadcasting
- Fixed positional encoding stability during generation
- Added SNN-based FFN option (`snn_ffn.py`)
- Added RAG memory retrieval (`memory_augmented_layer.py`)

**Remaining Questions**:
- How can we improve the place cell encoding to be more adaptive and learnable?
- Can we reduce hippocampal memory overhead while maintaining biological plausibility?
- What's the optimal balance between transformer layers and hippocampal components?

### 2. Training Efficiency

**Current Issues**:
- Slow training speed (66 steps/hour on L4 GPU)
- bfloat16 mixed precision not optimally utilized
- Gradient accumulation = 1 (could scale up)
- ~~No gradient checkpointing~~ **FIXED**: Gradient checkpointing implemented

**Implemented Solutions**:
- Gradient checkpointing in HippocampalTransformer (`set_gradient_checkpointing()`)
- Cosine LR schedule with warmup (`get_cosine_schedule_with_warmup()`)
- Memory warmup (disable hippocampal memory during early training)
- Gradient clipping and monitoring in trainer

**Questions**:
- Should we use `torch.compile()` for JIT optimization? (tried but disabled due to issues)
- What distributed training strategies would work best (DDP, FSDP, DeepSpeed)?
- Can we optimize the sleep phase consolidation (currently 25 steps every 2000)?

### 3. Memory Systems

**Current Issues**:
- Replay buffer size (1M samples) may be suboptimal
- Memory decay rate (0.03) is hand-tuned, not learned
- Cognitive map construction is O(n²) in number of memories
- No explicit forgetting mechanism beyond decay

**Questions**:
- Should we implement prioritized experience replay (PER) instead of uniform sampling?
- How can we make memory decay adaptive based on importance/recency?
- Can we use graph neural networks for cognitive map reasoning?
- Should we implement a hippocampal indexing system (like FAISS) for large-scale memory?

### 4. Generation Quality

**Current Issues**:
- ~~Repetition despite numerical stability fixes~~ **FIXED**: Multiple stability fixes applied
- ~~Entropy collapse~~ **MITIGATED**: Entropy regularization in loss
- Temperature scaling helps but is Band-Aid solution
- ~~No beam search or advanced decoding strategies~~ **FIXED**: Top-k, top-p, repetition penalty

**Implemented Solutions**:
- Nucleus (top-p) and top-k sampling in `SNNRAGTransformer.generate()`
- Repetition penalty during generation
- RAG memory augmentation for context-aware generation
- Entropy regularization in `HippocampalLoss`

**Remaining Questions**:
- Should we use contrastive decoding or speculative sampling?
- How can we enforce diversity during training (contrastive loss, unlikelihood training)?
- Can we add a diversity penalty to the attention mechanism itself?

### 5. Biological Plausibility vs Performance

**Current Issues**:
- Trade-off between neuroscience accuracy and computational efficiency
- Some bio-inspired components may not be necessary (e.g., random prosody)
- Theta-gamma encoding adds complexity but unclear benefit

**Questions**:
- Which neuromorphic components provide the most value (ablation study)?
- Should we simplify or double-down on biological mechanisms?
- How can we validate that hippocampal components are actually being used effectively?
- Can we add interpretability tools to visualize what the model learns?

### 6. Continual Learning

**Current Issues**:
- EWC lambda (0.4) is hand-tuned
- Fisher information matrix calculation is expensive
- Sleep consolidation every 2000 steps may be too infrequent
- No benchmarks for catastrophic forgetting prevention

**Questions**:
- Should we implement other continual learning methods (PackNet, Progressive Neural Networks)?
- Can we make consolidation adaptive (trigger when forgetting detected)?
- How do we measure memory retention over long training runs?
- Should we add a separate long-term memory store?

### 7. Multi-Modal Extension

**Current Goals**:
- Extend to vision + language
- Add spatial grounding for place cells (actual 2D/3D coordinates)
- Integrate audio prosody features (real pitch, energy, duration)

**Questions**:
- How should we architect multi-modal fusion with hippocampal components?
- Can place cells encode visual spatial relationships (object positions)?
- Should we train on visually-grounded language datasets (COCO, Visual Genome)?
- How can we leverage pre-trained vision encoders (CLIP, DINOv2)?

## 🚀 World-Class Standards to Target

### Performance Benchmarks

Compare against:
- **GPT-2 Small** (124M params): PPL ~18 on PTB, ~20 on WikiText-103
- **GPT-2 Medium** (355M params): PPL ~15 on PTB
- **LLaMA-3 8B**: PPL ~2-3 on general text
- **Gemma 2B**: State-of-art for small models

**Our Goals**:
- Match or exceed GPT-2 Small performance (we're close!)
- Demonstrate **superior continual learning** (10% catastrophic forgetting vs 40-60% baseline)
- Show **memory-augmented benefits** on long-context tasks
- Prove **biological mechanisms improve generalization**

### Research Impact

Aim for contributions to:
1. **Neuromorphic Computing**: Validate hippocampal-inspired architectures at scale
2. **Continual Learning**: New SOTA on forgetting prevention
3. **Memory-Augmented Networks**: Efficient episodic memory for LLMs
4. **Interpretability**: Neural activations map to cognitive processes

### Code Quality

- Clean, documented, type-annotated codebase
- Comprehensive test suite (unit, integration, benchmarks)
- Reproducible experiments with seed control
- Efficient distributed training scripts
- Easy-to-use inference API

## 🎓 Specific Optimization Requests

### 1. Immediate Fixes (Next 24-48 Hours)

**Priority**: High Impact, Low Effort

- [x] Fix DirectML scatter compatibility in `place_cell_encoder.py` (DONE - dtype fixes)
- [x] Implement Flash Attention for 2-3x speedup (DONE - using F.scaled_dot_product_attention)
- [x] Add nucleus sampling and typical decoding (DONE - in SNNRAGTransformer.generate)
- [ ] Enable `torch.compile()` with proper guards
- [x] Gradient checkpointing for 2x larger batches (DONE - in HippocampalTransformer)

### 2. Short-Term Improvements (Next 1-2 Weeks)

**Priority**: Core Performance

- [ ] Complete training to 50,000 steps with monitoring
- [ ] Implement distributed training (multi-GPU)
- [ ] Add comprehensive evaluation suite (perplexity, few-shot, continual learning)
- [x] Optimize memory systems (RAG retrieval implemented, gate/cross-attention injection)
- [x] Improve generation diversity (repetition penalty, top-k/top-p sampling)

### 3. Medium-Term Enhancements (Next 1-2 Months)

**Priority**: Research Contributions

- [ ] Multi-modal extension (vision + language)
- [ ] Ablation studies on hippocampal components
- [ ] Continual learning benchmarks (split-MNIST, Permuted MNIST, CORe50)
- [ ] Comparison against memory-augmented baselines (Transformer-XL, Compressive Transformer)
- [ ] Publish research paper and release models

### 4. Long-Term Vision (Next 3-6 Months)

**Priority**: World-Class System

- [ ] Scale to 1B+ parameters
- [ ] Real-time online learning system
- [ ] Deploy as API with memory persistence
- [ ] Build ecosystem (fine-tuning tools, applications)
- [ ] Establish benchmark suite for bio-inspired LLMs

## 🧪 Experimental Ideas to Explore

### Cutting-Edge Techniques

1. **Mixture of Depths**: Dynamic layer execution based on token importance
2. **Mamba/State Space Models**: Replace some transformer layers with SSMs
3. **Retrieval-Augmented Generation**: Combine with external knowledge base - **IMPLEMENTED** (`memory_augmented_layer.py`)
4. **Neural Architecture Search**: Optimize hippocampal parameters
5. **Differentiable Neural Computer**: Replace memory with external memory bank
6. **Spiking Neural Networks**: True neuromorphic inference - **IMPLEMENTED** (`snn_ffn.py`, `snn_rag_transformer.py`)

### Novel Research Questions

1. Can place cells learn hierarchical semantic spaces (WordNet structure)?
2. Do grid cells emerge naturally from training on spatial language?
3. Can episodic memory enable true one-shot learning?
4. Does sleep consolidation improve generalization more than standard training?
5. Can we train a "theory of mind" via episodic memory of agent interactions?

## 📊 Success Metrics

### Quantitative

- **Perplexity**: < 5 on WikiText-103 (currently 8-12)
- **Throughput**: > 10,000 tokens/sec (currently ~100-500)
- **Memory Efficiency**: < 20GB VRAM for 512 context (currently ~22GB)
- **Catastrophic Forgetting**: < 10% accuracy drop (need to measure)
- **Few-Shot Learning**: > 70% on SuperGLUE tasks (need to benchmark)

### Qualitative

- Generate coherent, diverse long-form text (> 500 tokens)
- Demonstrate memory retrieval during generation (interpretable)
- Show adaptive learning (improve on repeated prompts)
- Explain predictions via hippocampal activation maps
- Maintain conversation context over multiple turns

## 🤝 How You Can Help

### Analysis & Recommendations

1. **Code Review**: Analyze architecture files and suggest optimizations
2. **Algorithmic Improvements**: Propose better algorithms for memory systems
3. **Training Strategies**: Recommend hyperparameters, curricula, data augmentation
4. **Benchmark Design**: Design rigorous evaluation protocols
5. **Research Roadmap**: Prioritize experiments with highest expected impact

### Specific Outputs Requested

1. **Optimization Report** (markdown):
   - Top 10 highest-impact changes with implementation details
   - Estimated performance gains for each
   - Priority ranking (quick wins vs long-term investments)

2. **Implementation Pseudocode**:
   - Key algorithms needing rewrites
   - Modern best practices (Flash Attention, FSDP, etc.)
   - DirectML/GPU optimization strategies

3. **Research Experiment Design**:
   - Ablation study protocol
   - Continual learning benchmark suite
   - Comparison against baselines

4. **Architectural Proposals**:
   - Alternative hippocampal designs
   - Multi-modal fusion strategies
   - Scalability improvements (1B+ params)

## 🌟 Ultimate Goal

**Make Aura HippocampalTransformer the go-to reference implementation for:**
- Neuromorphic language models
- Hippocampal-inspired memory systems in AI
- Continual learning in large language models
- Biologically-plausible deep learning

**We want to prove that neuroscience-inspired architectures can compete with (and outperform) pure engineering approaches on real-world benchmarks.**

---

## 📎 Additional Context

- **Team**: Small research group, limited compute (L4 GPU, Colab)
- **Timeline**: Aiming for publication in 6 months
- **Constraints**: Must maintain biological plausibility (not pure engineering)
- **Audience**: ML researchers + computational neuroscientists

Thank you for helping us build the future of bio-inspired AI! 🧠✨
