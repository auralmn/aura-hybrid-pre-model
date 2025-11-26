# HippocampalTransformer: Training Implementation Recipes

## Recipe 1: Basic Training with Consolidation

```python
import torch
from torch.utils.data import DataLoader
from your_modules import HippocampalTransformer, HippocampalFormation
from hippocampal_training import HippocampalTransformerTrainer, TrainingConfig

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize hippocampus
hippocampus = HippocampalFormation(
    spatial_dimensions=768,
    n_place_cells=2000,
    n_time_cells=100,
    n_grid_cells=500
)

# Initialize model
model = HippocampalTransformer(config, hippocampus).to(device)

# Setup trainer with consolidation
config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    consolidation_frequency=1000,  # Sleep every 1000 steps
    sleep_duration=100,            # 100 replay iterations
    ewc_lambda=0.4
)

trainer = HippocampalTransformerTrainer(model, config)

# Training loop
for epoch in range(3):
    for batch_idx, batch in enumerate(train_loader):
        # Everything automated:
        # 1. Forward pass + memory creation
        # 2. Backprop
        # 3. Store in replay buffer
        # 4. Every 1000 steps: automatic sleep consolidation
        results = trainer.train_step(
            batch,
            metadata={'epoch': epoch}
        )
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Step {trainer.global_step}: Loss = {results['loss']:.4f}")

print("Training complete!")
print(f"Consolidation cycles: {trainer.consolidation_count}")
```

## Recipe 2: Multi-Task Learning with Continual Learning

```python
# Initialize once
model = HippocampalTransformer(config, hippocampus).to(device)
trainer = HippocampalTransformerTrainer(model, config)

# Task sequence: Task A → Task B → Task C
tasks = ['translation', 'summarization', 'qa']

for task_id, task_name in enumerate(tasks):
    print(f"\n{'='*60}")
    print(f"Training Task {task_id + 1}: {task_name}")
    print(f"{'='*60}")
    
    task_loader = get_task_dataloader(task_name)
    
    for epoch in range(num_epochs):
        for batch in task_loader:
            results = trainer.train_step(
                batch,
                metadata={
                    'task': task_name,
                    'task_id': task_id,
                    'importance': 0.8  # All task data important
                }
            )
    
    # === Key: Compute Fisher Information for task ===
    # This protects previous task's weights during next task training
    print(f"Computing task consolidation for {task_name}...")
    val_loader = get_task_dataloader(task_name, split='validation')
    trainer.consolidator.compute_fisher_information(val_loader)
    print("Fisher Information updated!")

# Test: Model remembers all three tasks
print("\nEvaluating all tasks:")
for task_name in tasks:
    accuracy = evaluate(model, get_task_dataloader(task_name, split='test'))
    print(f"{task_name}: {accuracy:.2%}")
    
# Without EWC: accuracy would degrade significantly on old tasks
# With EWC + consolidation: minimal forgetting
```

## Recipe 3: Long-Context Learning via Episodic Memory

```python
# Process very long documents
long_document = """
[100K tokens of text...]
"""

# Split into chunks
chunks = split_into_sequences(long_document, max_len=2048, stride=1024)
print(f"Processing {len(chunks)} chunks...")

# Each chunk creates episodic memory
for chunk_idx, chunk_embedding in enumerate(chunks):
    batch = create_batch([chunk_embedding])
    
    results = trainer.train_step(
        batch,
        metadata={
            'document': 'long_doc_1',
            'chunk_idx': chunk_idx,
            'total_chunks': len(chunks),
            'importance': 0.5
        }
    )

# === Later: Query across entire document ===
query = "What was mentioned in chunk 50?"
query_embedding = encode(query)

# Retrieve relevant chunks from episodic memory
memories = hippocampus.retrieve_similar_memories(
    query_features=query_embedding.cpu().numpy(),
    k=10
)

print("Retrieved memories:")
for mem_id, similarity in memories:
    memory = hippocampus.episodic_memories[mem_id]
    chunk_idx = memory.temporal_event.context.get('chunk_idx', 0)
    print(f"  Chunk {chunk_idx}: similarity = {similarity:.3f}")

# Answer query using retrieved chunks
context = aggregate_memory_features(memories)
answer = model.generate_from_context(query, context)
```

## Recipe 4: Importance-Based Training

```python
# Different importance levels for different data types
data_distribution = [
    (0.7, important_dataset),      # 70% important (difficulty=0.8)
    (0.2, medium_dataset),         # 20% medium (difficulty=0.5)
    (0.1, easy_dataset)            # 10% easy (difficulty=0.1)
]

for epoch in range(num_epochs):
    for prob, dataset in data_distribution:
        # Sample with probability
        if random.random() < prob:
            batch = next(dataset.iterator())
            
            # Compute importance from batch difficulty
            batch_difficulty = compute_difficulty(batch)
            importance = batch_difficulty  # Hard examples = high importance
            
            results = trainer.train_step(
                batch,
                metadata={'importance': importance}
            )

# During sleep consolidation:
# - Important examples (difficulty=0.8) replayed 8x more than easy (0.1)
# - Model focuses consolidation effort on hard examples
# - Better generalization and faster learning
```

## Recipe 5: Custom Consolidation Schedule

```python
class AdaptiveConsolidationTrainer(HippocampalTransformerTrainer):
    """Adapts consolidation frequency based on training dynamics"""
    
    def train_step(self, batch, metadata=None):
        results = super().train_step(batch, metadata)
        
        # Monitor recent loss trend
        recent_losses = list(self.train_losses)[-10:]
        loss_trend = recent_losses[-1] - recent_losses[0]
        
        # Adaptive consolidation:
        # - If loss increasing: consolidate more frequently (prevent drift)
        # - If loss decreasing: consolidate less frequently (speed up training)
        
        if loss_trend > 0.05:  # Loss increasing - need consolidation
            self.config.consolidation_frequency = min(
                self.config.consolidation_frequency // 2,  # 2x more frequent
                100  # But not more than every 100 steps
            )
            print("Loss increasing - consolidation frequency increased")
        
        elif loss_trend < -0.05:  # Loss stable/decreasing
            self.config.consolidation_frequency = min(
                self.config.consolidation_frequency * 1.5,  # Consolidate less
                2000  # But not less than every 2000 steps
            )
        
        return results

# Usage
trainer = AdaptiveConsolidationTrainer(model, config)

# Training automatically adjusts consolidation!
for batch in train_loader:
    trainer.train_step(batch)
```

## Recipe 6: Reward-Based Memory Prioritization

```python
# Custom metadata with reward signals
def compute_reward(batch_outputs, labels, ground_truth=None):
    """Compute reward for batch (what makes memory important to keep?)"""
    
    # Option 1: Inverse loss
    loss = F.cross_entropy(batch_outputs, labels)
    reward_loss = -loss.item()
    
    # Option 2: Surprise / KL divergence
    if hasattr(model, 'get_prediction_uncertainty'):
        uncertainty = model.get_prediction_uncertainty(batch_outputs)
        reward_surprise = uncertainty.mean().item()
    else:
        reward_surprise = 0
    
    # Option 3: Task relevance
    if ground_truth and 'task_id' in ground_truth:
        task_reward = 1.0 if ground_truth['task_id'] in active_tasks else 0.1
    else:
        task_reward = 0.5
    
    # Combine rewards
    total_reward = (
        0.5 * reward_loss +
        0.3 * reward_surprise +
        0.2 * task_reward
    )
    
    # Normalize to [0, 1]
    importance = sigmoid(total_reward)
    
    return importance

# Training with reward-based importance
for batch in train_loader:
    output = model(batch['input_ids'])
    reward = compute_reward(output, batch['labels'])
    
    trainer.train_step(
        batch,
        metadata={'importance': reward}
    )

# Result: Memorable/surprising/task-relevant data consolidated more
```

## Recipe 7: Memory Inspection and Analysis

```python
# View hippocampal state at any time
def inspect_memory_state(hippocampus, top_k=10):
    print(f"Total episodic memories: {len(hippocampus.episodic_memories)}")
    print(f"Theta phase: {hippocampus.theta_phase:.2f} rad")
    
    # Get strongest memories
    memories_by_strength = sorted(
        hippocampus.episodic_memories.items(),
        key=lambda x: x[1].strength,
        reverse=True
    )
    
    print("\nTop {} memories by strength:".format(top_k))
    for i, (mem_id, memory) in enumerate(memories_by_strength[:top_k]):
        age_hours = (time.time() - memory.temporal_event.timestamp) / 3600
        print(f"  {i+1}. {mem_id}")
        print(f"     Strength: {memory.strength:.3f}")
        print(f"     Age: {age_hours:.1f}h")
        print(f"     Retrievals: {memory.retrieval_count}")
        print()

# Analyze cognitive map (semantic relationships)
def analyze_cognitive_map(hippocampus):
    print("Cognitive map statistics:")
    distances = list(hippocampus.cognitive_map.values())
    print(f"  Mean distance: {np.mean(distances):.3f}")
    print(f"  Max distance: {np.max(distances):.3f}")
    print(f"  Min distance: {np.min(distances):.3f}")
    
    # Find closest memory pairs
    sorted_pairs = sorted(hippocampus.cognitive_map.items(), key=lambda x: x[1])
    print("\nClosest memory pairs (most similar):")
    for (mem_id1, mem_id2), distance in sorted_pairs[:5]:
        print(f"  {mem_id1} ↔ {mem_id2}: distance={distance:.3f}")

# Usage
inspect_memory_state(hippocampus)
analyze_cognitive_map(hippocampus)
```

## Recipe 8: Generating Synthetic Data via Memory Replay

```python
# Use replayed memories as synthetic training data for other tasks
class SyntheticDataGenerator:
    def __init__(self, hippocampus, model):
        self.hippocampus = hippocampus
        self.model = model
        
    def generate_samples(self, num_samples=1000):
        """Generate synthetic training data by replaying memories"""
        synthetic_data = []
        
        for _ in range(num_samples):
            # Sample memory
            memory_ids = list(self.hippocampus.episodic_memories.keys())
            if not memory_ids:
                break
            
            mem_id = np.random.choice(memory_ids)
            memory = self.hippocampus.episodic_memories[mem_id]
            
            # Get features from memory
            features = memory.temporal_event.features
            
            # Use model to generate completion/continuation
            # (This is a generative use of episodic memory)
            generated = self.model.generate_from_memory(
                features,
                max_length=100
            )
            
            synthetic_data.append({
                'original_memory': mem_id,
                'features': features,
                'generated_text': generated,
                'confidence': memory.strength
            })
        
        return synthetic_data

# Use for data augmentation or pre-training new tasks
synthetic_generator = SyntheticDataGenerator(hippocampus, model)
synthetic_data = synthetic_generator.generate_samples(5000)

# Pre-train on synthetic data for new task
print(f"Generated {len(synthetic_data)} synthetic samples")
print(f"Average memory strength: {np.mean([d['confidence'] for d in synthetic_data]):.3f}")
```

## Recipe 9: Debugging & Profiling Consolidation

```python
class ProfilingTrainer(HippocampalTransformerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consolidation_times = []
        self.spike_rates = []
        
    def sleep_phase(self):
        import time
        start_time = time.time()
        
        result = super().sleep_phase()
        
        elapsed = time.time() - start_time
        self.consolidation_times.append(elapsed)
        
        print(f"Consolidation time: {elapsed:.2f}s")
        print(f"Avg consolidation time: {np.mean(self.consolidation_times):.2f}s")
        
        return result
    
    def print_spike_statistics(self):
        """Print spike-related statistics from last consolidation"""
        for i, aux in enumerate(self.latest_aux_outputs):
            spike_rate = aux.get('spike_rate')
            if spike_rate:
                print(f"Layer {i}: spike rate = {spike_rate:.3f}")

# Usage with profiling
trainer = ProfilingTrainer(model, config)

# Training runs with automatic profiling
for batch in train_loader:
    trainer.train_step(batch)

# Analyze consolidation efficiency
print("\nConsolidation Analysis:")
print(f"Total consolidations: {len(trainer.consolidation_times)}")
print(f"Total time: {sum(trainer.consolidation_times):.2f}s")
print(f"Average time per consolidation: {np.mean(trainer.consolidation_times):.2f}s")
print(f"Max time: {max(trainer.consolidation_times):.2f}s")
```

## Recipe 10: Integration with Your Custom Hippocampus

```python
# Your existing hippocampus module
from your_hippocampal_module import (
    HippocampalFormation,
    LanguageHippocampalFormation,
    TemporalMemoryInterpolator
)

# Extend trainer to use your hippocampus features
class AdvancedHippocampalTrainer(HippocampalTransformerTrainer):
    def train_step(self, batch, metadata=None):
        results = super().train_step(batch, metadata)
        
        # Access your custom hippocampus methods
        
        # 1. Use temporal interpolation for smooth transitions
        if len(self.model.hippocampus.episodic_memories) > 1:
            memory_ids = list(self.model.hippocampus.episodic_memories.keys())
            if len(memory_ids) >= 2:
                mem1, mem2 = memory_ids[-2:]
                interpolator = TemporalMemoryInterpolator()
                
                mem1_features = self.model.hippocampus.episodic_memories[mem1].temporal_event.features
                mem2_features = self.model.hippocampus.episodic_memories[mem2].temporal_event.features
                
                # Smooth context transition using your Hilbert interpolation
                smooth_context = interpolator.interpolate(
                    mem1_features, mem2_features, t=0.5, mode='hilbert'
                )
        
        # 2. Use place cells for semantic distance
        place_cell_activity = results.get('place_cell_activity')
        # Analyze place cell firing patterns...
        
        # 3. Decay memories with your custom mechanism
        if results['global_step'] % 5000 == 0:
            self.model.hippocampus.decay_memories(decay_rate=0.02)
        
        return results

trainer = AdvancedHippocampalTrainer(model, config)
```

---

These recipes provide concrete templates for:
- Basic to advanced training scenarios
- Multi-task learning with forgetting prevention
- Long-context processing via episodic memory
- Adaptive and reward-based consolidation
- Debugging and analysis tools
- Integration with your custom hippocampus implementation

Choose the recipe matching your use case and customize parameters as needed!
