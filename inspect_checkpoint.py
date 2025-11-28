import torch
import os

#checkpoint_path = r"c:\Users\nickn\OneDrive\Desktop\aura_clean\models\aura-hippocampal-transformer-mid-train.pt"
checkpoint_path = r"checkpoint_latest.pt"

if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found at: {checkpoint_path}")
    exit(1)

try:
    # Try loading as a standard torch checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint loaded successfully.")
    print(f"Type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print("Keys:", checkpoint.keys())
        if 'state_dict' in checkpoint:
            print("\nState Dict Keys (first 10):")
            print(list(checkpoint['state_dict'].keys())[:100])
        elif 'model_state_dict' in checkpoint:
            print("\nModel State Dict Keys (first 10):")
            print(list(checkpoint['model_state_dict'].keys())[:100])
        else:
            print("\nFirst 10 Keys:")
            print(list(checkpoint.keys())[:100])
            
        # Check for specific architecture hints
        if any('gating' in k for k in checkpoint.keys()):
            print("\nFound 'gating' in keys -> Likely LiquidMoE")
        if any('hippocampus' in k for k in checkpoint.keys()):
            print("\nFound 'hippocampus' in keys -> Likely Aura IBNN")
        print(f"losses: {checkpoint['losses']}")
        print(f"perplexities: {checkpoint['perplexities']}")
        print(f"steps: {checkpoint['steps']}")
        print(f"global_step: {checkpoint['global_step']}")
        print(f"hippocampus_memories: {checkpoint['hippocampus_memories']}")
        print(f"replay_buffer_size: {checkpoint['replay_buffer_size']}")
        print(f"config: {checkpoint['config']}")
            
    else:
        print("Checkpoint is not a dict.")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
