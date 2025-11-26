"""
Hippocampal Transformer Model

Integrates all hippocampal components:
1. ThetaGammaPositionalEncoding
2. PlaceCellSemanticEncoder
3. HippocampalTransformerLayer (stack)
4. Output Head
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
from src.core.language_zone.hippocampal_layer import HippocampalTransformerLayer

class HippocampalTransformer(nn.Module):
    """
    Full Hippocampal Transformer Model.
    
    Args:
        config: Configuration object
        hippocampus: HippocampalFormation instance
    """
    
    def __init__(self, config, hippocampus):
        super().__init__()
        self.config = config
        self.hippocampus = hippocampus
        
        # 1. Embeddings & Encodings
        self.pos_encoder = ThetaGammaPositionalEncoding(config)
        self.semantic_encoder = PlaceCellSemanticEncoder(config, hippocampus)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        
        # 2. Transformer Layers
        self.layers = nn.ModuleList([
            HippocampalTransformerLayer(config, hippocampus)
            for _ in range(config.num_layers)
        ])
        
        # 3. Output Head
        self.output_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Tie weights (optional but standard)
        # self.output_head.weight = self.semantic_encoder.token_embedding.weight
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        prosody: Optional[torch.Tensor] = None,
        use_memory: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            prosody: [batch, seq_len, 4]
            use_memory: Whether to use hippocampal memory
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            place_cell_activity: [batch, seq_len, n_place_cells] (for memory creation)
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Semantic Encoding (Place Cells)
        # Returns: [batch, seq_len, dim], [batch, seq_len, n_place_cells]
        hidden_states, place_cell_activity = self.semantic_encoder(input_ids)
        
        # 2. Positional Encoding (Theta-Gamma)
        # Generate position indices [1, seq_len]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_encoding = self.pos_encoder(positions, seq_length=seq_len)
        hidden_states = hidden_states + pos_encoding
        
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 3. Transformer Layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, 
                prosody=prosody, 
                use_memory=use_memory
            )
            
        # 4. Output Head
        logits = self.output_head(hidden_states)
        
        return logits, place_cell_activity
