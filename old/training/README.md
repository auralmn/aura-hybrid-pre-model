# AURA Training Pipeline

This directory contains the clean, efficient training pipeline for AURA's Amygdala system with Liquid-MoE integration.

## Overview

The training pipeline implements surgical fixes for AURA-ready training:

- **Efficient SBERT encoding**: Batch-encode once, cache results
- **Stratified splitting**: Proper train/test splits by emotion labels
- **Linear model export**: Compatible with AmygdalaRelay
- **Multi-task training**: Single model for emotion, intent, and tone
- **Liquid-MoE integration**: Ready for streaming inference

## Key Features

### 1. Efficient Feature Building
- **Precomputed SBERT**: Batch-encode all texts once, cache embeddings
- **Fused features**: Sine waves + extras + SBERT (419 dimensions)
- **No duplicate processing**: Single feature building pipeline

### 2. Proper Data Splitting
- **Stratified splits**: Maintain label distribution across train/test
- **Consistent indexing**: Same indices for all feature types
- **Reproducible**: Fixed random seeds for consistent results

### 3. Model Architecture
- **Linear classifiers**: For AmygdalaRelay compatibility
- **Multi-task model**: Shared representation for all tasks
- **Export format**: W.npy, b.npy, labels.json for easy loading

### 4. Liquid-MoE Integration
- **Expert routing**: Trained models as MoE experts
- **Attention modulation**: Spiking attention affects routing
- **Streaming learning**: Real-time adaptation without backprop

## File Structure

```
aura/training/
├── clean_amygdala_trainer.py    # Main training pipeline
├── README.md                    # This file
└── examples/
    └── liquid_moe_amygdala_integration.py  # Integration example
```

## Usage

### 1. Training

```bash
cd aura/training
python clean_amygdala_trainer.py
```

This will:
- Load and preprocess the dataset
- Precompute SBERT embeddings
- Create stratified train/test splits
- Train linear classifiers for AmygdalaRelay
- Train multi-task model for Liquid-MoE
- Export all models and weights

### 2. Integration

```python
from examples.liquid_moe_amygdala_integration import LiquidMoEAmygdalaSystem

# Initialize system
system = LiquidMoEAmygdalaSystem(model_path="models")

# Analyze text
result = system.analyze_emotion("I'm so excited about this! 😊")

# Get predictions
print(result['expert_predictions'])
print(result['moe_routing'])
print(result['attention_analysis'])
```

## Model Outputs

### Linear Models (AmygdalaRelay Compatible)
- `emotion_classifier_W.npy`: Weight matrix (419, num_emotions)
- `emotion_classifier_b.npy`: Bias vector (num_emotions,)
- `emotion_classifier_labels.json`: Label mappings
- Similar files for intent and tone classifiers

### Multi-Task Model
- `multitask_model.pt`: PyTorch state dict
- `all_labels.json`: All label mappings

## Feature Engineering

### Sine Wave Embeddings
- **Length**: 32 dimensions
- **Parameters**: Frequency, amplitude, phase per emotion
- **Intensity scaling**: Multiplied by emotion intensity
- **Secondary emotions**: 50% weight addition

### Extra Features
- **Text length**: Normalized by 100
- **Exclamation marks**: Binary indicator
- **Tone keywords**: Binary indicator for specific tones

### SBERT Embeddings
- **Model**: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Normalization**: L2 normalized
- **Batch processing**: Efficient encoding

## Training Configuration

```python
# Feature dimensions
SINE_LENGTH = 32
SBERT_DIM = 384
EXTRA_FEATURES = 3
TOTAL_FEATURES = 419

# Training parameters
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 5e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

## Performance Optimizations

### 1. SBERT Batching
- Process texts in batches of 128
- Use GPU acceleration when available
- Cache embeddings to avoid recomputation

### 2. Feature Building
- Single pass through data
- Vectorized operations where possible
- Consistent feature dimensions

### 3. Model Training
- AdamW optimizer with weight decay
- Cross-entropy loss for classification
- Multi-task loss with task weighting

## Integration with AURA

### AmygdalaRelay
The linear models are directly compatible with the existing AmygdalaRelay:

```python
# Load weights
W = np.load("emotion_classifier_W.npy")
b = np.load("emotion_classifier_b.npy")

# Use in relay
relay = AmygdalaRelay(W=W, b=b, labels=label_maps)
```

### Liquid-MoE Router
The multi-task model integrates with the Liquid-MoE system:

```python
# Create expert from trained model
expert = AmygdalaMoEExpert("models", "emotion_classifier")

# Use in MoE router
moe_router = LiquidMoERouter(experts={"emotion": expert})
```

### Attention System
The system works with the spiking attention mechanism:

```python
# Analyze with attention
intent = router.analyze_conversation_intent(text, features)
attention_gain = intent['attention_gain']

# Route with attention modulation
moe_result = moe_router.route(features, attn_gain=attention_gain)
```

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1-Macro**: Macro-averaged F1 score
- **F1-Weighted**: Weighted F1 score by class frequency
- **Per-class metrics**: Detailed classification report

## Troubleshooting

### Common Issues

1. **Dimension mismatch**: Ensure feature dimensions match (419)
2. **Label mapping**: Check that label files are consistent
3. **Memory issues**: Reduce batch size for large datasets
4. **GPU memory**: Use CPU if GPU memory is insufficient

### Debugging

```python
# Check feature dimensions
assert X_train.size(1) == 419, f"Expected 419, got {X_train.size(1)}"

# Verify label mappings
print("Emotion labels:", len(label_maps['emotion']))
print("Intent labels:", len(label_maps['intent']))
print("Tone labels:", len(label_maps['tone']))
```

## Future Enhancements

1. **Fine-tuning**: SBERT fine-tuning for domain-specific tasks
2. **Data augmentation**: Synthetic data generation for rare classes
3. **Model compression**: Quantization and pruning for efficiency
4. **Multi-modal**: Integration with audio and visual features
5. **Real-time training**: Online learning with streaming data

## Contributing

When adding new features or models:

1. Maintain backward compatibility with AmygdalaRelay
2. Ensure feature dimensions are consistent
3. Add comprehensive tests
4. Update documentation
5. Follow the existing code style

## License

This training pipeline is part of the AURA project and follows the same license terms.
