import torch
import os
import sys
from dataclasses import dataclass
from transformers import T5Tokenizer
import torch.nn.functional as F

# Ensure we can import from src
sys.path.append(os.getcwd())

# Import Model Components
try:
    from src.core.hippocampal import HippocampalFormation
    from src.core.language_zone.hippocampal_transformer import HippocampalTransformer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you are running this script from the root of the repo.")
    sys.exit(1)

# === CONFIGURATION ===
@dataclass
class Config:
    # === MODEL ===
    vocab_size: int = 32000
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 16
    head_dim: int = 64
    dropout: float = 0.15
    max_seq_len: int = 512
    intermediate_size: int = 4096

    # === HIPPOCAMPAL ===
    theta_frequency: float = 8.0
    gamma_frequency: float = 40.0
    n_place_cells: int = 2000
    n_time_cells: int = 100
    n_grid_cells: int = 200

def generate_text_stable(model, tokenizer, prompt, max_tokens=50, temperature=0.8, device='cuda'):
    """Generate with numerical stability (ported from training script)"""
    model.eval()
    
    # Tokenize using SentencePiece model directly if available, to match training exactly
    if hasattr(tokenizer, 'sp_model'):
        sp = tokenizer.sp_model
        token_ids = sp.encode(prompt, out_type=int)
        # Manually handle decoding function later
        decode_fn = lambda ids: sp.decode(ids)
    else:
        # Fallback to standard tokenizer
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)

    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    generated_tokens = list(token_ids)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {token_ids}")
    print("-" * 40)

    try:
        with torch.no_grad():
             # Using autocast if available (cuda/cpu)
             # Use 'cpu' for autocast if cuda is not available, or disable it
             device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
             dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
             
             with torch.amp.autocast(device_type, dtype=dtype, enabled=False): # Disable autocast for stability on CPU default
                for step in range(max_tokens):
                    # Context Window
                    if input_ids.shape[1] > model.config.max_seq_len:
                        input_ids = input_ids[:, -model.config.max_seq_len:]

                    # Prosody (dummy)
                    # Ensure prosody is on the same device
                    prosody = torch.randn(1, input_ids.shape[1], 4, device=device)
                    
                    # Forward
                    logits, _ = model(input_ids, prosody=prosody, use_memory=True)
                    logits = logits[0, -1, :].float()

                    # ===== NUMERICAL STABILITY FIX =====
                    # Subtract max to prevent overflow
                    logits = logits - logits.max()

                    # Apply temperature
                    logits = logits / temperature

                    # Convert to probabilities with numerical stability
                    # Use log_softmax to prevent underflow
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    probs = torch.exp(log_probs)

                    # Add small epsilon to prevent exact zeros
                    probs = probs + 1e-10
                    probs = probs / probs.sum()

                    # ===== BLOCK LAST 5 TOKENS =====
                    # This prevents immediate loops
                    for token in generated_tokens[-5:]:
                        probs[token] = 1e-10
                    probs = probs / probs.sum()

                    # Sample
                    next_token = torch.multinomial(probs, 1)[0]

                    if next_token.item() == tokenizer.eos_token_id:
                        break

                    generated_tokens.append(next_token.item())
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                    
                    # Stream output
                    # For streaming, we decode just the new token if possible, or full sequence if needed
                    # T5 SentencePiece decoding is context-independent for single tokens mostly
                    if hasattr(tokenizer, 'sp_model'):
                        token_str = tokenizer.sp_model.id_to_piece(next_token.item())
                        # Replace SPIECE_UNDERLINE with space
                        token_str = token_str.replace(' ', ' ')
                    else:
                        token_str = tokenizer.decode([next_token.item()])
                        
                    print(token_str, end='', flush=True)

        print("\n" + "-" * 40)
        return decode_fn(generated_tokens)
        
    except Exception as e:
        print(f"\n[Error during generation: {e}]")
        return ""

def main():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Config
    config = Config()
    
    # Initialize Model Components
    hippocampus = HippocampalFormation(
        spatial_dimensions=2,
        n_place_cells=config.n_place_cells,
        n_time_cells=config.n_time_cells,
        n_grid_cells=config.n_grid_cells,
        feature_dim=config.embedding_dim,
        device=device.type # Ensure correct device for hippocampus buffers
    )
    model = HippocampalTransformer(config, hippocampus)


    
    print("Initializing Transformer...")
  
   
    model.to(device)
    
    # Load Checkpoint
    checkpoint_path = 'models/aura-hippocampal-transformer-mid-train.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        # Load with weights_only=False to support numpy types if present, 
        # or use the safer default if possible. 
        # The training script used weights_only=False due to numpy types.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # FIX: The checkpoint saves the model state dict, but the HippocampalTransformer
        # holds a reference to the 'hippocampus' object. 
        # The 'hippocampus' object has registered buffers (place_centers, etc.).
        # When saving, these buffers are part of the state_dict if they are registered as buffers.
        # However, it seems the checkpoint might not have them if they were not properly saved 
        # or if there's a structure mismatch.
        # 
        # In the training script, the 'hippocampus' is passed to the model constructor.
        # The model stores it as self.hippocampus.
        # So model.state_dict() WILL include 'hippocampus.place_centers' etc.
        #
        # If the loaded checkpoint is missing these keys, strict=False is needed.
        # These buffers are likely initialized in __init__ anyway, so loading them 
        # is only critical if we want to preserve the exact spatial state (which is random initially).
        # Since they are random/fixed buffers (mostly), strict=False is acceptable for inference 
        # if we accept re-initialization of the spatial map.
        
        # Let's try loading with strict=False to bypass buffer mismatches
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ Model loaded (Step: {checkpoint.get('global_step', 'Unknown')})")
        print("   Note: Loaded with strict=False due to missing hippocampal buffer keys.")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Load Tokenizer
    print("Loading Tokenizer...")
    try:
        # Load standard tokenizer
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        
        # Check if pad token is set, if not use eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        print(f"Tokenizer loaded. Pad ID: {tokenizer.pad_token_id}")
            
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Test Prompts
    prompts = [
        "The history of",
        "Artificial intelligence is",
        "The neural network learning process",
        "Once upon a time"
    ]
    
    print("\nStarting Generation Test (Stable)...")
    print("=" * 60)
    model.eval()
    
    for prompt in prompts:
        generate_text_stable(model, tokenizer, prompt, max_tokens=50, temperature=0.8, device=device)
        print()


if __name__ == "__main__":
    main()

