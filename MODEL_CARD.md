---
language:
- en
license: mit
library_name: pytorch
tags:
- neuromorphic
- hippocampal-transformer
- bio-inspired
- episodic-memory
- spiking-neural-networks
- hybrid-learning
- place-cells
- grid-cells
- time-cells
base_model: google/flan-t5-base
pipeline_tag: text-generation
---

# Model Card for Aura HippocampalTransformer

A bio-inspired neuromorphic language model that integrates hippocampal memory systems with transformer architecture, enabling episodic memory formation and spatiotemporal learning.

## Model Details

### Model Description

The HippocampalTransformer is a novel neural architecture that combines traditional transformer-based language modeling with neuroscience-inspired hippocampal formation components. The model integrates place cells, grid cells, and time cells to create episodic memories during language processing, enabling more biologically plausible learning and inference.

Key innovations include:
- **Place Cell Semantic Encoding**: Sparse population coding with ~3% activation mimicking hippocampal place cell activity
- **Theta-Gamma Coupling**: Neural oscillation-based position encoding for spatiotemporal awareness
- **Prosody-Modulated Attention**: Emotional and prosodic features influence attention mechanisms
- **Episodic Memory Formation**: Real-time memory consolidation during wake and sleep phases
- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting in continual learning

- **Developed by:** Aura Team (auralmn)
- **Model type:** Causal Language Model with Hippocampal Memory Systems
- **Language(s) (NLP):** English
- **License:** MIT
- **Finetuned from model:** google/flan-t5-base (tokenizer only)

### Model Sources

- **Repository:** https://github.com/auralmn/aura-hybrid-pre-model
- **Paper:** In preparation
- **Demo:** Interactive hippocampal visualization available in repository

## Uses

### Direct Use

The model is designed for research into bio-inspired language modeling and neuromorphic computing. Direct applications include:

- **Text Generation**: Causal language modeling with episodic memory
- **Memory-Augmented Learning**: Tasks requiring retention of context over extended sequences
- **Continual Learning Research**: Testing catastrophic forgetting mitigation strategies
- **Neuroscience-AI Bridge**: Validating computational models of hippocampal function

### Downstream Use

The architecture can be adapted for:
- Memory-augmented question answering
- Long-form narrative generation
- Personalized language models that retain user-specific episodic memories
- Multi-modal learning with spatial and temporal grounding

### Out-of-Scope Use

This model is **not suitable** for:
- Production deployment without further fine-tuning and safety testing
- Tasks requiring factual accuracy without verification (model may confabulate based on episodic memory)
- Real-time applications requiring low latency (hippocampal memory operations add computational overhead)
- Safety-critical applications without extensive validation

## Bias, Risks, and Limitations

**Technical Limitations:**
- Model trained for only 11,500 steps (early checkpoint)
- May exhibit repetitive text generation due to limited training
- Hippocampal memory system adds ~15-20% inference overhead
- DirectML acceleration encounters scatter operation compatibility issues (CPU recommended)

**Biases:**
- Inherits biases present in training data (Nemotron-CC-v2 / WikiText-103)
- Episodic memory formation may amplify training data biases through selective consolidation
- Place cell representations may encode spatial biases from text structure

**Risks:**
- Experimental architecture may produce unexpected outputs
- Memory decay mechanisms are configurable and may impact output consistency
- Sleep-phase consolidation can alter model behavior over time

### Recommendations

Users should:
- Validate all model outputs for factual accuracy
- Monitor episodic memory formation for unintended bias amplification
- Use appropriate temperature and sampling strategies to mitigate repetition
- Consider the model as a research prototype, not production-ready
- Run on CPU or CUDA devices (DirectML support experimental)

## How to Get Started with the Model

```python
import torch
from src.core.hippocampal import HippocampalFormation
from src.core.language_zone.hippocampal_transformer import HippocampalTransformer
from src.training.train_hippocampal import Config
from transformers import T5Tokenizer

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Initialize configuration
config = Config(
    vocab_size=32000,
    embedding_dim=768,
    num_layers=12,
    num_heads=16,
    max_seq_len=512,
    n_place_cells=2000
)

# Initialize hippocampus
hippocampus = HippocampalFormation(
    embedding_dim=768,
    n_place_cells=2000,
    n_time_cells=100,
    n_grid_cells=200
)

# Initialize model
model = HippocampalTransformer(config, hippocampus)

# Load checkpoint
checkpoint = torch.load("aura-hippocampal-transformer-mid-train.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Generate text
model.eval()
prompt = "The history of artificial intelligence"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
prosody = torch.zeros(1, input_ids.shape[1], 4)  # Dummy prosody features

with torch.no_grad():
    logits, memory_state = model(input_ids, prosody=prosody, use_memory=True)
    # ... continue generation loop ...
```

## Training Details

### Training Data

The model was trained on high-quality web text from **Nemotron-CC-v2** (High-Quality subset) with fallback to **WikiText-103** for text modeling.

**Dataset Characteristics:**
- **Primary:** Nvidia Nemotron-CC-v2 (High-Quality) - curated web text
- **Fallback:** WikiText-103 - linguistic diversity from Wikipedia
- **Preprocessing:** SentencePiece tokenization via T5 tokenizer
- **Context Length:** 512 tokens maximum
- **Streaming:** Enabled for memory-efficient training

### Training Procedure

#### Preprocessing

- **Tokenization:** T5 SentencePiece (google/flan-t5-base, vocab_size=32000)
- **Sequence Packing:** Truncated to 512 tokens, padded with pad_token_id
- **Prosody Features:** Randomly initialized 4D vectors (pitch, energy, duration, voice quality)
- **Attention Masking:** Causal masking for autoregressive generation
- **Label Smoothing:** 0.2 to prevent overconfidence

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision (AMP)
- **Batch Size:** 16 (effective with gradient accumulation: 16)
- **Learning Rate:** 3e-4 with cosine decay
- **Warmup Steps:** 1,500
- **Max Steps:** 50,000 (checkpoint at 11,500)
- **Optimizer:** AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1)
- **Memory Parameters:**
  - Replay buffer: 1M samples
  - Memory creation interval: Every 5 steps
  - Memory decay rate: 0.03
  - EWC lambda: 0.4
- **Consolidation:**
  - Sleep interval: Every 2,000 steps
  - Sleep phase steps: 25
  - Evaluation interval: Every 100 steps

#### Speeds, Sizes, Times

- **Checkpoint Size:** ~500MB (model weights + optimizer state + memory buffers)
- **Training Hardware:** Nvidia L4 GPU (22.5GB VRAM)
- **Training Speed:** ~66 steps/hour
- **Total Training Time:** ~175 hours for 11,500 steps
- **Inference Speed:** CPU ~2-5 tokens/sec, GPU ~15-30 tokens/sec (without DirectML)
- **Parameters:** ~110M (transformer) + 2M (hippocampal formation)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Internal validation performed on held-out sequences from WikiText-103 and Nemotron-CC-v2.

#### Factors

Evaluation disaggregated by:
- Memory load (with/without episodic memory retrieval)
- Sequence length (128, 256, 512 tokens)
- Temperature settings (0.7, 0.8, 1.0)

#### Metrics

- **Perplexity:** Standard language modeling metric
- **Memory Coherence:** Semantic similarity of retrieved episodic memories
- **Repetition Rate:** N-gram repetition analysis
- **Entropy:** Token probability distribution diversity

### Results

**Language Modeling (Step 11,500):**
- Perplexity: ~8-12 (varies by evaluation set)
- Loss: ~2.1-2.3

**Memory Performance:**
- Average episodic memories: ~50-100 per evaluation sequence
- Memory retrieval accuracy: 75-85% (top-5)
- Cognitive map density: 200-300 edges

**Generation Quality:**
- Coherence: Moderate (early checkpoint)
- Repetition: Present without stability fixes (addressed via temperature + blocking)
- Diversity: Entropy 2-4 (below ideal 4-6 range, requires further training)

#### Summary

The model demonstrates functional hippocampal memory formation and retrieval during language processing. Generation quality is limited by early checkpoint status (11,500/50,000 steps). Numerical stability improvements and repetition blocking enable usable text generation. Model shows promise for bio-inspired continual learning research.

## Model Examination

**Interpretability Analysis:**

1. **Place Cell Activation Patterns**: Top-3% sparsity maintained across tokens, consistent with biological hippocampus
2. **Attention Visualization**: Prosody modulation creates emotion-sensitive attention patterns
3. **Memory Trace Analysis**: Episodic memories cluster by semantic similarity in cognitive map
4. **Sleep Phase Consolidation**: EWC successfully preserves performance on earlier training data

**Representational Analysis:**
- Embedding space shows task-relevant clustering
- Hippocampal place cells learn distributed semantic representations
- Theta-gamma encoding captures positional information orthogonal to semantic content

## Environmental Impact

Carbon emissions estimated using training on Nvidia L4 GPU (Google Colab).

- **Hardware Type:** Nvidia L4 GPU (22.5GB VRAM)
- **Hours used:** ~175 hours
- **Cloud Provider:** Google Cloud Platform (Colab)
- **Compute Region:** us-central1 (Iowa)
- **Carbon Emitted:** ~12-15 kg CO2eq (estimated based on L4 TDP ~72W and Iowa grid carbon intensity)

## Technical Specifications

### Model Architecture and Objective

**Architecture:**
```
Input Tokens (vocab_size=32000)
    ↓
PlaceCellSemanticEncoder (sparse 3% activation)
    ↓ [embedding_dim=768, n_place_cells=2000]
Theta-Gamma Position Encoding
    ↓
12× HippocampalTransformerLayer
    ├─ HippocampalProsodyAttention (16 heads × 64 dim)
    │   ├─ Prosody Modulation (4D features)
    │   └─ Memory Gate (hippocampal retrieval)
    └─ Feed-Forward Network (4096 intermediate)
    ↓
LM Head (projection to vocab)
    ↓
Output Logits
```

**Hippocampal Formation:**
- Place Cells: 2,000 units (Gaussian receptive fields)
- Grid Cells: 200 units (hexagonal patterns, multi-scale)
- Time Cells: 100 units (logarithmic temporal intervals)
- Episodic Memory: Spatio-temporal binding with feature vectors
- Cognitive Maps: Graph structure (spatial relationships)

**Training Objective:**
- Causal language modeling with cross-entropy loss
- Label smoothing (0.2) for regularization
- EWC penalty during sleep phases (λ=0.4)

### Compute Infrastructure

#### Hardware

- **Training:** Nvidia L4 GPU (22.5GB VRAM) via Google Colab
- **Inference:** CPU (recommended), CUDA GPUs, DirectML (experimental)

#### Software

- **Framework:** PyTorch 2.6+
- **Precision:** bfloat16 mixed precision (AMP)
- **Tokenizer:** HuggingFace Transformers (T5Tokenizer)
- **Dependencies:** 
  - torch >= 2.0
  - transformers >= 4.30
  - sentencepiece
  - datasets
  - numpy
  - tqdm

## Citation

**BibTeX:**

```bibtex
@software{aura_hippocampal_transformer_2025,
  title={Aura HippocampalTransformer: Bio-Inspired Neuromorphic Language Model},
  author={Aura Team},
  year={2025},
  url={https://github.com/auralmn/aura-hybrid-pre-model},
  note={Checkpoint step 11,500}
}
```

**APA:**

Aura Team. (2025). *Aura HippocampalTransformer: Bio-Inspired Neuromorphic Language Model* (Version 0.1-alpha) [Computer software]. https://github.com/auralmn/aura-hybrid-pre-model

## Glossary

- **Place Cells**: Neurons that fire when an agent is in a specific location (hippocampal CA1/CA3)
- **Grid Cells**: Neurons with hexagonal firing patterns for spatial navigation (entorhinal cortex)
- **Time Cells**: Neurons encoding temporal intervals and event sequences
- **Episodic Memory**: Memory of specific events with spatiotemporal context
- **Theta Oscillation**: 4-12 Hz brain rhythm associated with navigation and memory
- **Gamma Oscillation**: 30-100 Hz rhythm associated with attention and binding
- **EWC**: Elastic Weight Consolidation - prevents catastrophic forgetting by constraining important weights
- **STDP**: Spike-Timing-Dependent Plasticity - Hebbian learning rule based on temporal correlation
- **Prosody**: Rhythm, stress, and intonation patterns in speech (modeled as 4D features)

## More Information

**Research Context:**

This model is part of the Aura Neuromorphic Computing initiative, exploring hybrid ANN-SNN architectures that bridge neuroscience and deep learning. The hippocampal memory system enables:

1. **Continual Learning**: Learning new tasks without forgetting old ones
2. **One-Shot Learning**: Rapid episodic memory formation from single exposures
3. **Compositional Generalization**: Combining memories to handle novel situations
4. **Explainable Retrieval**: Memory traces provide interpretable reasoning

**Related Work:**
- Hippocampal formation neuroscience (O'Keefe & Nadel, 1978)
- Memory-augmented neural networks (Graves et al., 2014)
- Neural Turing Machines (Graves et al., 2014)
- Differentiable Neural Computers (Graves et al., 2016)
- Elastic Weight Consolidation (Kirkpatrick et al., 2017)

**Future Directions:**
- Complete training to 50,000 steps
- Multi-modal integration (vision + language)
- Online continual learning benchmarks
- Biological validation of memory formation dynamics

## Model Card Authors

Aura Team (auralmn organization)

## Model Card Contact

For questions, issues, or collaboration:
- **GitHub Issues:** https://github.com/auralmn/aura-hybrid-pre-model/issues
- **Repository:** https://github.com/auralmn/aura-hybrid-pre-model
