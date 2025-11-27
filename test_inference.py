import torch
import sys
import os
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

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
        if config.embedding_dim == 256: 
             config.embedding_dim = embedding_dim
        print(f"Inferred n_place_cells: {n_place_cells}")

    # Infer num_heads from prosody_gate
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
    for key in state_dict.keys():
        if 'ffn.0.weight' in key:
            intermediate_size, _ = state_dict[key].shape
            config.intermediate_size = intermediate_size
            print(f"Inferred intermediate_size: {intermediate_size}")
            break
            
    return config

def generate_text_stable(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, device='cpu'):
    """Generate text with numerical stability and repetition blocking."""
    model.eval()
    
    # Use sp_model if available (as in notebook), otherwise standard tokenizer
    if hasattr(tokenizer, 'sp_model'):
        sp = tokenizer.sp_model
        token_ids = sp.encode(prompt, out_type=int)
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
        generated_tokens = list(token_ids)
        decode_fn = sp.decode
        eos_id = sp.eos_id()
    else:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated_tokens = input_ids[0].tolist()
        decode_fn = lambda t: tokenizer.decode(t, skip_special_tokens=True)
        eos_id = tokenizer.eos_token_id
        
    print(f"Prompt: '{prompt}'")
    print("Generating...", end=' ', flush=True)
    
    # Check for out-of-vocab tokens
    if (input_ids >= model.config.vocab_size).any():
        print(f"WARNING: Input contains tokens >= vocab_size ({model.config.vocab_size}). Clamping.")
        input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Create dummy prosody
            curr_seq_len = input_ids.shape[1]
            prosody = torch.zeros(1, curr_seq_len, 4).to(device)
            
            logits, _ = model(input_ids, prosody=prosody)
            next_token_logits = logits[0, -1, :].float()
            
            # ===== NUMERICAL STABILITY FIX (from notebook) =====
            # Subtract max to prevent overflow
            next_token_logits = next_token_logits - next_token_logits.max()
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Convert to probabilities with numerical stability
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            probs = torch.exp(log_probs)
            
            # Add small epsilon
            probs = probs + 1e-10
            probs = probs / probs.sum()
            
            # ===== BLOCK LAST 5 TOKENS (Repetition Penalty) =====
            for token in generated_tokens[-5:]:
                if token < len(probs):
                    probs[token] = 1e-10
            probs = probs / probs.sum()
            
            # Sample
            next_token = torch.multinomial(probs, 1)[0]
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            print(".", end='', flush=True)
            
            if next_token.item() == eos_id:
                break
                
    print(" Done.")
    decoded = decode_fn(generated_tokens)
    return decoded

def diagnose_repetition(model, tokenizer, device):
    """Diagnose repetition issues by checking entropy and top-k probabilities."""
    print("="*70)
    print("DIAGNOSING REPETITION ISSUE")
    print("="*70)

    model.eval()
    test_prompt = "The history of"
    
    if hasattr(tokenizer, 'sp_model'):
        sp = tokenizer.sp_model
        test_ids = sp.encode(test_prompt, out_type=int)
        id_to_piece = sp.id_to_piece
    else:
        test_ids = tokenizer.encode(test_prompt, add_special_tokens=False)
        id_to_piece = tokenizer.convert_ids_to_tokens

    input_ids = torch.tensor([test_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        # Create dummy prosody
        prosody = torch.randn(1, len(test_ids), 4, dtype=torch.float32, device=device) # Use float32 for compatibility
        logits, _ = model(input_ids, prosody=prosody, use_memory=True)

        # Check the probability distribution
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, 10)

        print("\nTop 10 predictions for next token:")
        for prob, idx in zip(top_k_probs, top_k_indices):
            try:
                token_str = id_to_piece(idx.item())
            except:
                token_str = str(idx.item())
            print(f"  {token_str:20} : {prob.item():.4f}")

        # Check entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        print(f"\nEntropy: {entropy.item():.2f} (should be 2-5)")
        print(f"Max prob: {probs.max().item():.4f} (should be < 0.5)")

        if probs.max().item() > 0.8:
            print("\n⚠️ WARNING: Model is outputting one token with >80% probability!")
            print("   This causes repetition. The model may be underfitting or")
            print("   the learning rate might be too high causing instability.")
    
    model.train()
    print("="*70)

def main():
    checkpoint_path = os.path.join(project_root, "models", "aura-hippocampal-transformer-mid-train.pt")
    
    # Force CPU as requested
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 1. Load Checkpoint & Config
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        config = infer_config_from_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # 2. Initialize Model
    print("Initializing model components...")
    hippocampus = HippocampalFormation(
        config.embedding_dim,
        config.n_place_cells,
        50, 100
    )
    model = HippocampalTransformer(config, hippocampus).to(device)
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 3. Initialize Tokenizer
    print("Initializing T5 Tokenizer (google/flan-t5-base)...")
    try:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)
        # Verify sp_model access as requested
        sp = tokenizer.sp_model
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        print(f"SentencePiece model loaded: {type(sp)}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        # Fallback to AutoTokenizer if T5Tokenizer fails (e.g. if sentencepiece install failed)
        try:
            print("Falling back to AutoTokenizer (Fast)...")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", use_fast=True)
            print(f"Fallback Tokenizer vocab size: {tokenizer.vocab_size}")
        except Exception as e2:
            print(f"Fallback failed: {e2}")
            return

    # 4. Diagnose Repetition
    try:
        diagnose_repetition(model, tokenizer, device)
    except Exception as e:
        print(f"Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Run Inference on Multiple Prompts
    prompts = [
        "The history of",
        "In the future",
        "Neural networks",
        "Machine learning",
        "Deep learning"
    ]
    
    print("\nRunning Inference Tests...")
    for p in prompts:
        try:
            output = generate_text_stable(model, tokenizer, p, max_new_tokens=25, temperature=0.8, device=device)
            print(f"  '{p}' → '{output}'")
        except Exception as e:
            print(f"\nInference failed for prompt '{p}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
