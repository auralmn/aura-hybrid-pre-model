# Aura Hybrid Pre-Model - Development Roadmap

Based on the analysis of the legacy codebase (`old/`), the following components have been identified for integration into the current system to enhance bio-plausibility and functionality.

## 1. Thalamic Routing System
*   [ ] **Port `ThalamicConversationRouter`**: Implement the sophisticated routing logic from `old/core/thalamic_router.py`.
    *   Implement query characteristic analysis (greeting, emotional, historical, etc.).
    *   Implement routing strategies (Direct, Parallel, Staged).
    *   Integrate with `LiquidMoERouter`.
*   [ ] **Attention Mechanism**: Ensure `MultiChannelSpikingAttention` is fully integrated with the router.

## 2. Emotional Processing (Amygdala)
*   [ ] **Implement `Amygdala` Module**: Port `old/core/amygdala.py`.
    *   Create `Amygdala` class with `threat_detectors`, `emotional_processors`, and `social_evaluators`.
    *   Implement `process_threat`, `process_emotional_salience`, and `assess_social_threat`.
    *   Integrate fear conditioning and emotional memory.

## 3. Homeostatic Control (Hypothalamus)
*   [ ] **Implement Endocrine System**: Port `old/core/hippothalamus.py`.
    *   Create `Hypothalamus` for monitoring system metrics (energy, accuracy, stress).
    *   Create `Pituitary` for hormone release (Cortisol, Dopamine, etc.).
    *   Implement hormone effects on network hyperparameters (learning rate, expert capacity).

## 4. Personality System
*   [ ] **Integrate Personality Profiles**: Port `old/core/personality.py`.
    *   Define `PersonalityProfile` dataclass.
    *   Integrate personality embeddings into the routing and response generation.

## 5. System Integration
*   [ ] **Connect Components**:
    *   Wire Amygdala outputs to Thalamic Router (emotional routing).
    *   Apply Hypothalamic hormones to global system parameters.
    *   Ensure Hippocampal formation interacts with the new components.

## 6. Language Zone (LLM Integration)
*   [ ] **Implement Language Zone (Production)**:
    *   **Methodology**: **Test-Driven Development (TDD)**. Write tests first.
    *   **Design Spec**: See `docs/LANGUAGE_ZONE_SPECS.md`.
    *   **Core Components**:
        *   [ ] **GIF Neuron**: Implement Generalized Integrate-and-Fire neuron with compression ($T' = 2, L = 16$).
        *   [ ] **Synapsis Module**: Implement synaptic plasticity and efficient linear transformations.
        *   [ ] **SNN Transformer Blocks**: Implement `SNNMatmul`, `SNNSoftmax`, `SNNSiLU`.
        *   [ ] **Binary Embeddings**: Implement conversion with surrogate gradients.
        *   [ ] **Spiking Attention**: Implement Spiking Temporal-Sequential Attention (STSA).
    *   **Training**:
        *   [ ] **3-Stage Pipeline**: Implement Quantized ANN -> Conversion -> STDP Fine-tuning.
        *   [ ] **FPT**: Implement Fixed-point Parallel Training for speed.
    *   **Integration**:
        *   [ ] **Thalamic Connection**: Connect to Thalamic Router.
        *   [ ] **Memory Consolidation**: Implement write-back to Hippocampal Formation.
