import torch
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
project_root = r"c:\Users\nickn\OneDrive\Desktop\aura_clean"
sys.path.insert(0, project_root)

from src.core.hippocampal import HippocampalFormation
from src.core.language_zone.hippocampal_transformer import HippocampalTransformer
from src.training.train_hippocampal import Config

def infer_config_from_state_dict(state_dict):
    """Infer model configuration from state dict shapes."""
    config = Config()
    
    # Infer vocab_size and embedding_dim from token_embedding
    if 'semantic_encoder.token_embedding.weight' in state_dict:
        vocab_size, embedding_dim = state_dict['semantic_encoder.token_embedding.weight'].shape
        config.vocab_size = vocab_size
        config.embedding_dim = embedding_dim
        print(f"Inferred vocab_size: {vocab_size}, embedding_dim: {embedding_dim}")
    
    # Infer n_place_cells from semantic_projection
    if 'semantic_encoder.semantic_projection.weight' in state_dict:
        n_place_cells, embedding_dim = state_dict['semantic_encoder.semantic_projection.weight'].shape
        config.n_place_cells = n_place_cells
        # embedding_dim is already inferred from token_embedding, but good to double check or set if missing
        if config.embedding_dim == 256: # Default
             config.embedding_dim = embedding_dim
        print(f"Inferred n_place_cells: {n_place_cells}")

    # Infer num_heads from prosody_gate
    # Look for any layer's prosody_gate
    for key in state_dict.keys():
        if 'attention.prosody_gate.weight' in key:
            num_heads, _ = state_dict[key].shape
            config.num_heads = num_heads
            print(f"Inferred num_heads: {num_heads}")
            break
            
    # Infer num_layers
    max_layer = -1
    for key in state_dict.keys():
        if key.startswith('layers.'):
            try:
                layer_idx = int(key.split('.')[1])
                max_layer = max(max_layer, layer_idx)
            except ValueError:
                pass
    if max_layer >= 0:
        config.num_layers = max_layer + 1
        print(f"Inferred num_layers: {config.num_layers}")

    # Infer intermediate_size from FFN
    # Look for layers.0.ffn.0.weight (Linear(embedding_dim, intermediate_size))
    # Shape is [out_features, in_features] -> [intermediate_size, embedding_dim]
    for key in state_dict.keys():
        if 'ffn.0.weight' in key:
            intermediate_size, _ = state_dict[key].shape
            config.intermediate_size = intermediate_size
            print(f"Inferred intermediate_size: {intermediate_size}")
            break
            
    return config

def verify_model():
    checkpoint_path = os.path.join(project_root, "models", "aura-hippocampal-transformer-mid-train.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    print("Inferring config...")
    config = infer_config_from_state_dict(state_dict)
    
    print("Initializing components...")
    # Initialize Hippocampus (needed for model init)
    hippocampus = HippocampalFormation(
        config.embedding_dim,
        config.n_place_cells,
        50, # Default
        100 # Default
    )
    
    # Initialize Model
    model = HippocampalTransformer(config, hippocampus)
    
    # Load State Dict
    print("Loading state dict into model...")
    try:
        model.load_state_dict(state_dict, strict=False) # strict=False to handle potential missing keys like 'hippocampus'
        print("State dict loaded successfully.")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return
        
    model.eval()
    
    # Run Dummy Inference
    print("Running dummy inference...")
    dummy_input = torch.randint(0, config.vocab_size, (1, 10)) # Batch size 1, seq len 10
    
    try:
        with torch.no_grad():
            logits, place_activity = model(dummy_input)
        print("Inference successful!")
        print(f"Logits shape: {logits.shape}")
        print(f"Place activity shape: {place_activity.shape}")
        
        # Check if output makes sense (not all NaNs)
        if torch.isnan(logits).any():
            print("WARNING: Logits contain NaNs!")
        else:
            print("Logits check passed (no NaNs).")
            
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    verify_model()
