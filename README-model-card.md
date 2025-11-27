# Aura Hybrid Pre-Model

A bio-inspired hybrid neural architecture combining traditional ANNs with neuromorphic Spiking Neural Networks (SNNs) and hippocampal-inspired memory systems.

## Overview

Aura is a neuromorphic AI system that integrates:
- **Hybrid Learning**: Combines backpropagation (ANN) with Hebbian learning (SNN)
- **Hippocampal Formation**: Place cells, grid cells, and time cells for episodic memory
- **Liquid Mixture-of-Experts**: Dynamic expert routing with bio-plausible learning
- **Temporal Memory Interpolation**: Four modes (linear, Fourier, Hilbert, Hamiltonian)

## Key Features

### Neuromorphic Components
- **Izhikevich Neurons**: 23 biologically realistic firing patterns
- **STDP Learning**: Spike-timing-dependent plasticity
- **OjaLayer**: Hebbian learning with dynamic neurogenesis
- **Liquid MoE**: Continuous-time expert routing

### Memory Systems
- **Place Cells**: Spatial location encoding with Gaussian receptive fields
- **Grid Cells**: Hexagonal spatial navigation patterns
- **Time Cells**: Temporal interval coding
- **Episodic Memory**: Spatio-temporal event binding
- **Cognitive Maps**: Relationship graphs between memories

### Performance
- **MNIST**: 94.34% accuracy with hybrid Oja + linear readout (CPU only, 5 epochs)
- **Tested**: 17 comprehensive tests for hippocampal formation
- **Visualized**: Real-time memory formation and neural activity

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/auralmn/aura-hybrid-pre-model.git
cd aura-hybrid-pre-model

# Install dependencies (using uv)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Run MNIST Test

```bash
# Train hybrid model on MNIST
python -m pytest tests/test_mnist_performance.py -v

# Expected output: ~94% accuracy after 5 epochs
```

### Interactive Hippocampal Demo

```bash
# Live visualization of memory formation
python tests/demo_interactive_hippocampus.py

# Controls:
#   n = Store next MNIST digit
#   s = Skip 5 digits
#   a = Toggle auto-play
#   q = Quit
```

## Architecture

### Hybrid ANN-SNN Model

```
Input (MNIST 784D)
    ↓
OptimizedWhitener (normalization)
    ↓
OjaLayer (1024 components, Hebbian)
    ↓
Linear Readout (10 classes, backprop)
    ↓
Output
```

### Hippocampal Formation

```
Spatial Input → Place Cells (50) → Spatial Code
                Grid Cells (30)  → Navigation
                
Temporal Input → Time Cells (20) → Temporal Code

Episodic Memory = Spatial + Temporal + Features
```

## Project Structure

```
aura_clean/
├── src/
│   ├── base/              # Brain zones, neurons, processors
│   ├── core/              # Brain, Liquid MoE, experts
│   ├── encoders/          # Event encoders, embeddings
│   ├── maths/             # Mathematical primitives
│   ├── services/          # Brain system services
│   ├── tools/             # Diagnostic tools
│   └── training/          # Hebbian, STDP, memory systems
├── tests/
│   ├── test_hippocampal_formation.py     # 17 comprehensive tests
│   ├── test_hippocampal_visualization.py # 6 static visualizations
│   ├── test_mnist_performance.py         # 94.34% accuracy
│   ├── demo_interactive_hippocampus.py   # Live demo
│   └── artifacts/                        # Generated visualizations
├── old2/                  # Reference implementations
└── README.md
```

## Key Components

### Hippocampal Formation

Complete implementation of hippocampal memory system:

```python
from tests.test_hippocampal_formation import HippocampalFormation

hippo = HippocampalFormation(
    n_place_cells=100,
    n_time_cells=50,
    n_grid_cells=75
)

# Update spatial state
hippo.update_spatial_state(location, dt=0.1)

# Create episodic memory
hippo.create_episodic_memory(
    memory_id="mem_1",
    event_id="event_1", 
    features=feature_vector,
    associated_experts=["expert_a"]
)

# Retrieve similar memories
similar = hippo.retrieve_similar_memories(
    query_features, 
    location=query_location, 
    k=5
)
```

### Temporal Memory Interpolation

Four interpolation modes for smooth memory transitions:

```python
from tests.test_hippocampal_formation import TemporalMemoryInterpolator

interpolator = TemporalMemoryInterpolator()

# Linear interpolation
result = interpolator.interpolate(M0, M1, t=0.5, mode='linear')

# Fourier domain interpolation
result = interpolator.interpolate(M0, M1, t=0.5, mode='fourier')

# Hilbert transform (phase-preserving)
result = interpolator.interpolate(M0, M1, t=0.5, mode='hilbert')

# Hamiltonian (quantum-inspired)
result = interpolator.interpolate(M0, M1, t=0.5, mode='hamiltonian')
```

## Visualizations

Generated visualizations in `tests/artifacts/`:

1. **place_cells_visualization.png** - Receptive fields and trajectory activity
2. **grid_cells_visualization.png** - Hexagonal patterns at multiple scales
3. **time_cells_visualization.png** - Sequential temporal coding
4. **episodic_memories_visualization.png** - Spatial distribution with cognitive map
5. **memory_interpolation_visualization.png** - Four interpolation modes
6. **hippocampal_system_overview.png** - Complete system state
7. **mnist_hippocampal_live.gif** - Real-time memory formation (2.3MB)
8. **mnist_hippocampal_final.png** - Final memory clustering by digit


## Development Philosophy

Aura follows these principles:

1. **Bio-plausibility**: Inspired by real neuroscience
2. **Hybrid Learning**: Best of both ANN and SNN worlds
3. **Test-driven**: All features thoroughly tested
4. **Visualized**: Understanding through visualization
5. **Exploratory**: Implementing cutting-edge concepts


## Performance Benchmarks

| Model | Dataset | Accuracy | Training | Hardware |
|-------|---------|----------|----------|----------|
| Hybrid (Oja + Linear) | MNIST | 94.34% | 5 epochs (~19 min) | CPU |
| Hybrid (Oja + Linear) | MNIST | TBD | TBD | DirectML GPU |

## Technical Details

### Hebbian Learning (OjaLayer)

- Online principal component analysis
- Dynamic neurogenesis (component growth)
- Numerical stability for overcomplete representations
- Learning rate: 0.001

### Memory Formation

- Episodic binding of spatial, temporal, and feature information
- Cognitive map construction (spatial relationships)
- Temporal map construction (event sequences)
- Natural memory decay with configurable rate

### Neural Codes

- **Place cells**: Gaussian receptive fields, theta phase precession
- **Grid cells**: Hexagonal spatial patterns, multi-scale
- **Time cells**: Logarithmically distributed temporal intervals

## License

MIT License - See LICENSE file for details

## Citation

If you use Aura in your research, please cite:

```bibtex
@software{aura_hybrid_2025,
  title={Aura Hybrid Pre-Model: Bio-Inspired Neuromorphic AI},
  author={Aura Team},
  year={2025},
  url={https://github.com/auralmn/aura-hybrid-pre-model}
}
```

## Acknowledgments

Built on principles from:
- Hippocampal formation neuroscience
- Liquid state machines
- Mixture-of-experts architectures
- Hebbian learning theory
- Temporal memory systems

---

**Aura** - A leader in neuromorphic computing and hybrid AI
