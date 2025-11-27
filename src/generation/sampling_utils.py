import torch
import torch.nn.functional as F

def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """
    Nucleus sampling: restricts candidate pool to the smallest set of tokens 
    whose cumulative probability exceeds top_p.
    """
    # Apply temperature
    logits = logits / max(temperature, 1e-5)
    
    # Sort logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    return logits

def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: torch.Tensor, penalty: float = 1.2):
    """
    Penalize tokens that have already been generated.
    """
    if generated_tokens.numel() == 0:
        return logits
        
    score = torch.gather(logits, 1, generated_tokens)
    
    # If score < 0, then penalty makes it smaller (more negative) -> multiply
    # If score > 0, then penalty makes it smaller -> divide
    score = torch.where(score < 0, score * penalty, score / penalty)
    
    logits.scatter_(1, generated_tokens, score)
    return logits