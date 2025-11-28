"""
Training Configuration for Hippocampal Transformer.

Provides optimized default values based on empirical testing.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HippocampalTransformerConfig:
    """
    Model configuration for HippocampalTransformer.
    """
    # Model Architecture
    vocab_size: int = 32000
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    max_seq_len: int = 512
    intermediate_size: int = 3072
    
    # Hippocampal Components
    n_place_cells: int = 2000
    place_cell_sparsity: float = 0.03
    theta_freq: float = 8.0
    gamma_freq: float = 40.0
    
    # Optimization Flags
    use_gradient_checkpointing: bool = False
    use_weight_tying: bool = True


@dataclass
class TrainingConfig:
    """
    Training hyperparameters with optimized defaults.
    """
    # Basic Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_steps: int = 100000
    
    # Learning Rate
    lr: float = 1e-4
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Regularization
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    dropout: float = 0.1
    
    # Loss Components
    entropy_lambda: float = 0.05
    sparsity_lambda: float = 0.02
    target_sparsity: float = 0.03
    
    # Memory System
    memory_warmup_steps: int = 5000
    replay_buffer_size: int = 50000
    ewc_lambda: float = 0.4
    
    # Sleep-Wake Cycle
    sleep_interval: int = 1000
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Mixed Precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    
    # Gradient Checkpointing
    use_gradient_checkpointing: bool = False


@dataclass 
class OptimizedConfig:
    """
    Combined model and training config with optimized defaults.
    
    Usage:
        config = OptimizedConfig()
        model = HippocampalTransformer(config.model, hippocampus)
        trainer = HippocampalTransformerTrainer(model, config.training, hippocampus)
    """
    model: HippocampalTransformerConfig = field(default_factory=HippocampalTransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        # Sync gradient checkpointing flag
        if self.training.use_gradient_checkpointing:
            self.model.use_gradient_checkpointing = True


# Preset configurations for different scenarios

def get_debug_config() -> OptimizedConfig:
    """Small config for debugging."""
    config = OptimizedConfig()
    config.model.num_layers = 2
    config.model.embedding_dim = 256
    config.model.num_heads = 4
    config.model.n_place_cells = 500
    config.training.batch_size = 4
    config.training.max_steps = 1000
    config.training.warmup_steps = 100
    config.training.memory_warmup_steps = 500
    return config


def get_small_config() -> OptimizedConfig:
    """Small model for limited VRAM (8GB)."""
    config = OptimizedConfig()
    config.model.num_layers = 6
    config.model.embedding_dim = 512
    config.model.num_heads = 8
    config.model.n_place_cells = 1000
    config.training.batch_size = 16
    config.training.use_gradient_checkpointing = True
    return config


def get_medium_config() -> OptimizedConfig:
    """Medium model for 16GB VRAM."""
    config = OptimizedConfig()
    config.model.num_layers = 12
    config.model.embedding_dim = 768
    config.model.num_heads = 12
    config.model.n_place_cells = 2000
    config.training.batch_size = 32
    config.training.use_gradient_checkpointing = True
    return config


def get_large_config() -> OptimizedConfig:
    """Large model for 24GB+ VRAM."""
    config = OptimizedConfig()
    config.model.num_layers = 24
    config.model.embedding_dim = 1024
    config.model.num_heads = 16
    config.model.intermediate_size = 4096
    config.model.n_place_cells = 4000
    config.training.batch_size = 64
    config.training.use_gradient_checkpointing = True
    return config

