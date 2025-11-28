import torch
import torch.nn as nn
import torch.nn.functional as F

class HippocampalLoss(nn.Module):
    """
    Composite loss function for Hippocampal Transformer.
    
    Components:
    1. CrossEntropy: Main task loss (next token prediction)
    2. Entropy Regularization: Penalizes low-entropy distributions (repetition loops)
    3. Sparsity Penalty: Enforces ~3% activity in place cells (biological constraint)
    """
    def __init__(self, 
                 label_smoothing: float = 0.1, 
                 entropy_lambda: float = 0.05,   # Increased strength for anti-repetition
                 sparsity_lambda: float = 0.02, 
                 target_sparsity: float = 0.03):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.entropy_lambda = entropy_lambda
        self.sparsity_lambda = sparsity_lambda
        self.target_sparsity = target_sparsity
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, place_cell_activity: torch.Tensor = None) -> torch.Tensor:
        # 1. Task Loss
        # Flatten: [batch*seq_len, vocab]
        loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # 2. Entropy Regularization (Maximize Entropy)
        # H(p) = -sum(p * log(p))
        # We subtract entropy from loss (min loss -> max entropy)
        if self.entropy_lambda > 0:
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            loss = loss - (self.entropy_lambda * entropy)

        # 3. Place Cell Sparsity
        # Activity: [Batch, Seq, N_cells]
        if place_cell_activity is not None and self.sparsity_lambda > 0:
            # Calculate mean activity per token
            current_sparsity = place_cell_activity.mean()
            # L2 penalty towards target (0.03)
            sparsity_loss = (current_sparsity - self.target_sparsity) ** 2
            loss = loss + (self.sparsity_lambda * sparsity_loss)
            
        return loss