# 🧠 Aura HippocampalTransformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Card-yellow)](MODEL_CARD.md)

A bio-inspired neuromorphic language model that integrates hippocampal memory systems with transformer architecture, enabling episodic memory formation and spatiotemporal learning.

## ✨ Key Features

- 🧬 **Bio-Inspired Architecture**: Integrates place cells, grid cells, and time cells from neuroscience
- 🔄 **Episodic Memory Formation**: Real-time memory consolidation during inference
- 🌊 **Theta-Gamma Coupling**: Neural oscillation-based position encoding
- 🎭 **Prosody-Modulated Attention**: Emotional features influence attention mechanisms  
- 🛡️ **Continual Learning**: EWC prevents catastrophic forgetting
- 🔬 **Neuromorphic Components**: Hybrid ANN-SNN architecture with Hebbian learning

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/auralmn/aura-hybrid-pre-model.git
cd aura-hybrid-pre-model

# Install dependencies
pip install -r requirements.txt

# Or use uv for faster installation
uv sync
```

### Basic Usage

```python
import torch
from src.core.hippocampal import HippocampalFormation
from src.core.language_zone.hippocampal_transformer import HippocampalTransformer
from src.training.train_hippocampal import Config
from transformers import T5Tokenizer

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Initialize model
config = Config(
    vocab_size=32000,
    embedding_dim=768,
    num_layers=12,
    num_heads=16,
    n_place_cells=2000
)

hippocampus = HippocampalFormation(
    embedding_dim=768,
    n_place_cells=2000,
    n_time_cells=100,
    n_grid_cells=200
)

model = HippocampalTransformer(config, hippocampus)

# Load checkpoint
checkpoint = torch.load("models/aura-hippocampal-transformer-mid-train.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Generate text
model.eval()
prompt = "The future of artificial intelligence"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
prosody = torch.zeros(1, input_ids.shape[1], 4)

with torch.no_grad():
    logits, memory_state = model(input_ids, prosody=prosody, use_memory=True)
```

### Inference Script

For stable text generation with repetition blocking:

```bash
python test_inference.py
```

## 🏗️ Architecture

```
Input Tokens (32K vocab)
    ↓
PlaceCellSemanticEncoder
    ├─ Sparse activation (3% sparsity)
    └─ 2000 place cells
    ↓
Theta-Gamma Position Encoding
    ├─ θ rhythm: 8 Hz
    └─ γ rhythm: 40 Hz
    ↓
12× HippocampalTransformerLayer
    ├─ Multi-head Attention (16 heads)
    │   ├─ Prosody modulation
    │   └─ Hippocampal memory gate
    ├─ Feed-forward (4096 dim)
    └─ Layer normalization
    ↓
Language Model Head
    ↓
Output Logits (32K vocab)
```

### Hippocampal Formation

```
Spatial Processing          Temporal Processing
    ↓                            ↓
Place Cells (2000)          Time Cells (100)
Grid Cells (200)                 ↓
    ↓                       Event Sequences
Spatial Maps                     ↓
    ↓                            ↓
    └────────────┬───────────────┘
                 ↓
         Episodic Memory
         (Cognitive Maps)
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Perplexity | 8-12 |
| Training Steps | 11,500 / 50,000 |
| Parameters | ~112M |
| Memory Retrieval Accuracy | 75-85% (top-5) |
| Inference Speed (CPU) | 2-5 tokens/sec |
| Inference Speed (GPU) | 15-30 tokens/sec |

## 🎯 Training Details

### Dataset
- **Primary**: Nvidia Nemotron-CC-v2 (High-Quality subset)
- **Fallback**: WikiText-103
- **Tokenizer**: T5 SentencePiece (google/flan-t5-base)
- **Context Length**: 512 tokens

### Hyperparameters
- **Precision**: bfloat16 mixed precision
- **Batch Size**: 16
- **Learning Rate**: 3e-4 (cosine decay)
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95)
- **Warmup**: 1,500 steps
- **Label Smoothing**: 0.2
- **EWC Lambda**: 0.4

### Consolidation
- **Sleep Interval**: Every 2,000 steps
- **Memory Creation**: Every 5 steps
- **Replay Buffer**: 1M samples
- **Memory Decay**: 0.03 per step

### Hardware
- **Training**: Nvidia L4 GPU (22.5GB VRAM)
- **Time**: ~175 hours for 11,500 steps
- **Inference**: CPU recommended (DirectML experimental)

## 📁 Project Structure

```
aura_clean/
├── src/
│   ├── core/
│   │   ├── hippocampal.py              # Hippocampal formation
│   │   └── language_zone/
│   │       ├── hippocampal_transformer.py
│   │       ├── hippocampal_attention.py
│   │       ├── hippocampal_layer.py
│   │       ├── place_cell_encoder.py
│   │       └── theta_gamma_encoding.py
│   └── training/
│       ├── train_hippocampal.py        # Training script
│       ├── hippocampal_trainer.py      # Trainer class
│       └── train_wikitext2.py          # WikiText-2 training
├── models/
│   └── aura-hippocampal-transformer-mid-train.pt  # Mid-training checkpoint
├── tests/
│   └── test_hippocampal_formation.py   # 17 tests
├── test_inference.py                   # Inference script
├── verify_hippocampal_model.py         # Model verification
├── MODEL_CARD.md                       # HuggingFace model card
└── README.md                           # This file
```

## 🔬 Research Applications

### Neuroscience
- Computational models of hippocampal function
- Episodic memory formation dynamics
- Spatial and temporal coding mechanisms

### Machine Learning
- Continual learning without catastrophic forgetting
- Memory-augmented neural networks
- Bio-inspired attention mechanisms
- One-shot learning from episodic memory

### Applications
- Long-form narrative generation
- Memory-augmented question answering
- Personalized language models
- Multi-modal learning with spatiotemporal grounding

## ⚠️ Limitations

- **Early Checkpoint**: Model trained for 11,500/50,000 steps
- **Repetition**: May generate repetitive text (use temperature + blocking)
- **DirectML Issues**: Scatter operation incompatibility (use CPU/CUDA)
- **Inference Overhead**: Hippocampal operations add 15-20% latency
- **Experimental**: Research prototype, not production-ready

## 🛠️ Development

### Running Tests

```bash
# Run hippocampal formation tests
pytest tests/test_hippocampal_formation.py -v

# Verify model checkpoint
python verify_hippocampal_model.py

# Test inference pipeline
python test_inference.py
```

### Training from Scratch

```bash
# Train on WikiText-2
python src/training/train_wikitext2.py

# Train with custom config
python src/training/train_hippocampal.py
```

### Visualization

```bash
# Generate hippocampal visualizations
python tests/test_hippocampal_visualization.py

# Interactive memory formation demo
python tests/demo_interactive_hippocampus.py
```

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@software{aura_hippocampal_transformer_2025,
  title={Aura HippocampalTransformer: Bio-Inspired Neuromorphic Language Model},
  author={Aura Team},
  year={2025},
  url={https://github.com/auralmn/aura-hybrid-pre-model},
  note={Checkpoint step 11,500}
}
```

## 🤝 Contributing

We welcome contributions! Areas of interest:

- **Model Training**: Continue training to 50,000 steps
- **Multi-modal**: Vision + language integration
- **Benchmarks**: Continual learning evaluations
- **Optimization**: DirectML compatibility, inference speed
- **Documentation**: Tutorials, examples, explanations

Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Model Card**: [MODEL_CARD.md](MODEL_CARD.md)
- **Repository**: [github.com/auralmn/aura-hybrid-pre-model](https://github.com/auralmn/aura-hybrid-pre-model)
- **Issues**: [github.com/auralmn/aura-hybrid-pre-model/issues](https://github.com/auralmn/aura-hybrid-pre-model/issues)
- **HuggingFace**: Coming soon

## 🙏 Acknowledgments

Built on principles from:
- Hippocampal formation neuroscience (O'Keefe & Nadel, 1978)
- Memory-augmented neural networks (Graves et al., 2014)
- Elastic weight consolidation (Kirkpatrick et al., 2017)
- Transformer architectures (Vaswani et al., 2017)
- T5 tokenization (Raffel et al., 2020)

## 🌟 Aura Initiative

**Aura** - A leader in neuromorphic computing and hybrid AI

*Bridging neuroscience and artificial intelligence for the next generation of cognitive systems.*

---

**Status**: 🚧 Research Prototype | **Version**: 0.1-alpha | **Checkpoint**: 11,500/50,000 steps
