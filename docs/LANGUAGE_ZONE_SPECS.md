# Language Zone: Technical Specifications

## Overview
The Language Zone is a production-grade, bio-plausible, efficient language model integrated into the Aura brain. It leverages Generalized Integrate-and-Fire (GIF) neurons and hybrid ANN-SNN training to achieve LLM-scale performance with SNN energy efficiency. Development follows a strict Test-Driven Development (TDD) approach to ensure reliability and scalability.

## Core Technology: Generalized Integrate-and-Fire (GIF) Neurons
To compress spike length and maintain efficiency (aligned with **SpikeLLM 2025**):
-   **Mechanism**: Merge $L$ IF steps into one GIF step.
-   **Accumulation**: Accumulate membrane potential $V$ over $L$ steps.
-   **Spike Generation**: Emits multi-bit spike when $V > V_{th}$.
    -   Spike value $s = \text{clip}(\lfloor V / V_{th} \rfloor, 0, L)$.
    -   **Reset**: $V \leftarrow V - s \cdot V_{th}$ (Soft Reset).
-   **Encoding**: $\log_2 L$ bit encoding per merged step.
-   **Configuration**: $T' = 2$ (merged steps), $L = 16$ (levels).

## Architecture Components

### 1. Binary Embedding Conversion (Input)
Converts continuous embeddings to binary spikes for feed-forward propagation.
-   **Mechanism**: Heaviside function with surrogate gradients.
-   **Reconstruction**:
    $$x^{(\ell)} \approx V_{th} \frac{1}{T'} \sum_{t=1}^{T'} s^{(\ell)}(t) + \min(x^{(\ell-1)})$$

### 2. Spiking Layers (BrainTransformer Architecture)
-   **Neuron Type**: GIF Neurons (Generalized Integrate-and-Fire) with adaptive threshold.
-   **Synaptic Plasticity**: Implement **Synapsis Module** to simulate synaptic plasticity.
-   **SNN-Specific Transformer Components**:
    -   **SNNMatmul**: Convert matrix multiplication to cumulative outer product.
        -   **Stability**: Apply gradient clipping and normalization after accumulation to prevent explosion ($A^T / \sqrt{d_k}$).
    -   **SNNSoftmax**: Spike accumulation and normalization.
    -   **SNNSiLU**: Piecewise approximation of SiLU.
    -   **SNNRMSNorm**: SNN approximation of RMS Normalization.

### 3. Continuous Readout (Output)
Decodes spike activity back to continuous embeddings.

### 4. Attention Mechanism
-   **Spiking Temporal-Sequential Attention (STSA)**:
    -   **Note**: Validate "partial-time dependency" solution with ablation studies.
    -   **Alternative**: Explore spike-driven self-attention (alpha-XNOR) if STSA fails.

### 5. Memory Consolidation Interface
-   **Trigger**: End of sequence generation or specific attention threshold.
-   **Extraction**: Mean spike rate of the final layer or last hidden state.
-   **API**: `consolidate(spikes, metadata) -> TemporalEvent`.

## Training Strategy (Phased Approach)

### Phase 1: Feasibility (Weeks 1-4)
-   **Target**: 100M-500M parameter model.
-   **Hardware**: L4/T4 GPU.
-   **Goal**: Validate architecture and training loop.

### Phase 2: Optimization (Weeks 5-8)
-   **Target**: Scale up if Phase 1 succeeds.
-   **Method**: Hybrid ANN-SNN Training.
    1.  **ANN Training (Quantized)**.
    2.  **ANN-to-SNN Conversion**.
    3.  **STDP Fine-Tuning (Experimental)**: Optional step for bio-plausibility. Skip if performance degrades.

### Optimization for L4/T4 Hardware
-   **Coding**: TTFS (Time-to-First-Spike) or Rate coding (GIF uses a hybrid).
-   **Precision**: Mixed precision (INT8/FP16) where possible.
-   **Target**: Train a small demo model (< 1B params or specialized architecture) in < 4 hours.

## Implementation Plan
1.  **`src/core/language_zone/`**: New directory.
2.  **`gif_neuron.py`**: Implement the GIF neuron class.
3.  **`embeddings.py`**: Binary embedding conversion and readout.
4.  **`attention.py`**: Spiking Temporal-Sequential Attention.
5.  **`model.py`**: Assemble the SNN-LLM.

## Testing Strategy (TDD)
All components must be implemented using Test-Driven Development with specific metrics:
1.  **GIF Neuron**:
    -   Test spike count distribution (histogram).
    -   Test quantization error ($|continuous - decoded| < \epsilon$).
2.  **Attention**:
    -   Test gradient flow magnitude (> 1e-8).
    -   Test numerical stability (no NaN/Inf with long sequences).
3.  **Full Model**:
    -   Perplexity on language benchmarks.
    -   Energy consumption estimation.
    -   Latency vs. timesteps.
