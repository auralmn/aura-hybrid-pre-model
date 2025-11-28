"""
Place Cell Semantic Encoder

Encodes tokens using sparse place cell population coding.
Mimics hippocampal place cell representations for semantic embeddings.
"""

import torch
import torch.nn as nn


class PlaceCellSemanticEncoder(nn.Module):
    """
    Encode tokens using place cell population coding.
    
    Maps discrete tokens to continuous semantic space with sparse activation,
    mimicking how hippocampal place cells encode spatial positions.
    
    Key Properties:
    - Sparse activation (~3% of place cells active per token)
    - Population coding (distributed representation)
    - Reconstruction via linear readout
    - Residual connection preserves original embedding
    
    Args:
        config: Configuration with vocab_size, embedding_dim, n_place_cells
        hippocampus: HippocampalFormation instance (for integration)
        
    Example:
        >>> config = Config(vocab_size=50257, embedding_dim=768, n_place_cells=2000)
        >>> hippocampus = HippocampalFormation(...)
        >>> encoder = PlaceCellSemanticEncoder(config, hippocampus)
        >>> input_ids = torch.tensor([[1, 2, 3, 4]])
        >>> semantic_embeds, place_activity = encoder(input_ids)
        >>> semantic_embeds.shape, place_activity.shape
        (torch.Size([1, 4, 768]), torch.Size([1, 4, 2000]))
    """
    
    def __init__(self, config, hippocampus):
        super().__init__()
        self.config = config
        self.hippocampus = hippocampus
        
        # Standard token embedding with proper initialization
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        # Initialize embeddings with smaller variance for stability
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Project to place cell space
        # This creates a "semantic map" like spatial maps in hippocampus
        self.semantic_projection = nn.Linear(config.embedding_dim, config.n_place_cells)
        
        # Reconstruct from place cells back to semantic space
        self.place_to_semantic = nn.Linear(config.n_place_cells, config.embedding_dim)
        
        # Sparsity parameter (top-k selection)
        # Real place cells: ~1-5% active at any given location
        # We use ~3% as a biologically plausible compromise
        self.sparsity = 0.03
        self.k = int(config.n_place_cells * self.sparsity)
        
    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode tokens via sparse place cell activation.
        
        Args:
            input_ids: [batch, seq_len] token indices
            
        Returns:
            semantic_embedding: [batch, seq_len, embedding_dim] reconstructed embeddings
            place_cell_activity: [batch, seq_len, n_place_cells] sparse activation
        """
        # 1. Get token embeddings (dense)
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, embedding_dim]
        
        # 2. Project to place cell space
        place_cell_logits = self.semantic_projection(token_embeds)  # [batch, seq_len, n_place_cells]
        
        # 3. Sparse activation via top-k selection
        # Only the k most responsive place cells fire (like receptive fields)
        batch_size, seq_len, n_cells = place_cell_logits.shape
        
        # Find top-k values and indices
        topk_values, topk_indices = torch.topk(
            place_cell_logits, 
            self.k, 
            dim=-1
        )  # [batch, seq_len, k]
        
        # Create sparse place cell activation
        place_cell_activity = torch.zeros_like(place_cell_logits)
        
        # Apply sigmoid to top-k values (0-1 activation)
        topk_activations = torch.sigmoid(topk_values)
        
        # Scatter sparse activations into full place cell vector
        # Scatter sparse activations into full place cell vector
        # Use out-of-place scatter to avoid DirectML in-place errors
        place_cell_activity = place_cell_activity.scatter(
            dim=-1,
            index=topk_indices,
            src=topk_activations
        )
        
        # 4. Reconstruct semantic embedding from sparse place cell code
        reconstructed = self.place_to_semantic(place_cell_activity)
        
        # 5. Residual connection (preserve original information)
        # This is like the direct pathway in hippocampus (CA3 -> CA1)
        # Scale down the reconstructed signal to avoid overwhelming the original embedding
        semantic_embedding = token_embeds + 0.1 * reconstructed
        
        return semantic_embedding, place_cell_activity
    
    def get_place_cell_pattern(self, token_id: int) -> torch.Tensor:
        """
        Get the place cell activation pattern for a specific token.
        
        Useful for analysis and visualization.
        
        Args:
            token_id: Single token ID
            
        Returns:
            place_pattern: [n_place_cells] sparse activation pattern
        """
        input_ids = torch.tensor([[token_id]], device=next(self.parameters()).device)
        _, place_activity = self.forward(input_ids)
        return place_activity[0, 0]
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'vocab_size={self.config.vocab_size}, '
                f'embedding_dim={self.config.embedding_dim}, '
                f'n_place_cells={self.config.n_place_cells}, '
                f'sparsity={self.sparsity:.1%} (k={self.k})')
