import torch
import os
import sys
import numpy as np

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.core.hippocampal import HippocampalFormation
from src.core.language_zone.hippocampal_transformer import HippocampalTransformer
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 32000
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 16
    head_dim: int = 64
    dropout: float = 0.15
    max_seq_len: int = 512
    intermediate_size: int = 4096
    
    # Hippocampal
    theta_frequency: float = 8.0
    gamma_frequency: float = 40.0
    n_place_cells: int = 2000
    n_time_cells: int = 100
    n_grid_cells: int = 200

def check_keys():
    checkpoint_path = 'checkpoint_latest.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        ckpt_keys = set(checkpoint['model_state_dict'].keys())
        print(f"Checkpoint has {len(ckpt_keys)} keys.")
        
        # Initialize model
        config = Config()
        hippocampus = HippocampalFormation(
            config.embedding_dim,
            config.n_place_cells,
            config.n_time_cells,
            config.n_grid_cells,
            device='cpu'
        )
        model = HippocampalTransformer(config, hippocampus)
        model_keys = set(model.state_dict().keys())
        print(f"Model has {len(model_keys)} keys.")
        
        # Compare
        missing_in_ckpt = model_keys - ckpt_keys
        missing_in_model = ckpt_keys - model_keys
        
        print("\n=== MISSING IN CHECKPOINT (Model expects these, but not found) ===")
        # Filter out likely buffers
        critical_missing = []
        for k in sorted(list(missing_in_ckpt)):
            if 'hippocampus' in k: # We know these might be missing buffers
                continue
            if 'num_batches_tracked' in k: # BatchNorm buffer
                continue
            critical_missing.append(k)
            
        if not critical_missing:
            print("No critical keys missing (ignoring hippocampal buffers).")
        else:
            for k in critical_missing:
                print(f"  {k}")
                
        print("\n=== MISSING IN MODEL (Checkpoint has these, but model doesn't) ===")
        for k in sorted(list(missing_in_model)):
            print(f"  {k}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_keys()

