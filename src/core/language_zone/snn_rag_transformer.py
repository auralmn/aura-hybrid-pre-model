"""
SNN-RAG Hippocampal Transformer

Full model integrating:
1. Spiking Neural Network FFN layers
2. Retrieval-Augmented Generation via hippocampal memory
3. All existing hippocampal components (place cells, theta-gamma, prosody)
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List

from src.core.language_zone.theta_gamma_encoding import ThetaGammaPositionalEncoding
from src.core.language_zone.place_cell_encoder import PlaceCellSemanticEncoder
from src.core.language_zone.memory_augmented_layer import MemoryAugmentedLayer


class SNNRAGTransformer(nn.Module):
    """
    Hippocampal Transformer with SNN layers and RAG memory.
    
    Architecture:
    - PlaceCellSemanticEncoder: Sparse population coding
    - ThetaGammaPositionalEncoding: Neural oscillation positions
    - MemoryAugmentedLayer stack: Self-attention + RAG + SNN FFN
    - Output head with weight tying
    
    Config options:
    - use_snn_ffn: Replace MLP with SNN in specified layers
    - snn_layers: List of layer indices to use SNN (e.g., [0, 2, 4])
    - memory_injection: "cross_attention", "concat", or "gate"
    - num_retrieved: Number of memories to retrieve per forward pass
    """
    
    def __init__(
        self,
        config,
        hippocampus,
        use_snn_ffn: bool = True,
        snn_layers: Optional[List[int]] = None,
        memory_injection: str = "gate",
        num_retrieved: int = 5
    ):
        super().__init__()
        self.config = config
        self.hippocampus = hippocampus
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        
        # Default: SNN in every other layer
        if snn_layers is None:
            snn_layers = list(range(0, config.num_layers, 2))
        self.snn_layers = set(snn_layers)
        
        # 1. Embeddings & Encodings
        self.pos_encoder = ThetaGammaPositionalEncoding(config)
        self.semantic_encoder = PlaceCellSemanticEncoder(config, hippocampus)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        
        # 2. Memory-Augmented Transformer Layers
        self.layers = nn.ModuleList([
            MemoryAugmentedLayer(
                config=config,
                hippocampus=hippocampus,
                use_snn_ffn=(use_snn_ffn and i in self.snn_layers),
                memory_injection=memory_injection,
                num_retrieved=num_retrieved
            )
            for i in range(config.num_layers)
        ])
        
        # 3. Output Head
        self.output_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.output_head.weight = self.semantic_encoder.token_embedding.weight
        
        # Track which layers use SNN
        snn_count = sum(1 for i in range(config.num_layers) if i in self.snn_layers)
        print(f"SNNRAGTransformer: {snn_count}/{config.num_layers} layers use SNN FFN")
        print(f"Memory injection: {memory_injection}, retrieving {num_retrieved} memories")
    
    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing."""
        self.use_gradient_checkpointing = enable
    
    def _layer_forward(
        self,
        layer: MemoryAugmentedLayer,
        hidden_states: torch.Tensor,
        prosody: Optional[torch.Tensor],
        use_memory: bool,
        store_memory: bool
    ) -> torch.Tensor:
        """Helper for gradient checkpointing."""
        return layer(
            hidden_states,
            prosody=prosody,
            use_memory=use_memory,
            store_memory=store_memory
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        prosody: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        store_memory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [Batch, Seq] token IDs
            prosody: [Batch, Seq, 4] prosody features
            use_memory: Whether to use RAG memory retrieval
            store_memory: Whether to store context in memory (training)
            
        Returns:
            logits: [Batch, Seq, Vocab]
            place_cell_activity: [Batch, Seq, N_place_cells]
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Semantic Encoding (Place Cells)
        hidden_states, place_cell_activity = self.semantic_encoder(input_ids)
        
        # 2. Positional Encoding (Theta-Gamma)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_encoding = self.pos_encoder(positions, seq_length=seq_len)
        hidden_states = hidden_states + pos_encoding
        
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 3. Transformer Layers with RAG and SNN
        for i, layer in enumerate(self.layers):
            # Only store memory in last layer during training
            should_store = store_memory and (i == len(self.layers) - 1)
            
            if self.training and self.use_gradient_checkpointing:
                hidden_states = checkpoint(
                    self._layer_forward,
                    layer,
                    hidden_states,
                    prosody,
                    use_memory,
                    should_store,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    prosody=prosody,
                    use_memory=use_memory,
                    store_memory=should_store
                )
        
        # 4. Output Head
        logits = self.output_head(hidden_states)
        
        return logits, place_cell_activity
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        use_memory: bool = True,
        repetition_penalty: float = 1.2
    ) -> torch.Tensor:
        """
        Generate text with RAG memory augmentation.
        
        Args:
            input_ids: [Batch, Seq] prompt tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            use_memory: Whether to use RAG during generation
            repetition_penalty: Penalty for repeated tokens
            
        Returns:
            generated: [Batch, Seq + max_new_tokens] full sequence
        """
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if exceeds max length
                if generated.shape[1] >= self.config.max_seq_len:
                    context = generated[:, -self.config.max_seq_len:]
                else:
                    context = generated
                
                # Forward pass with memory
                logits, _ = self.forward(context, use_memory=use_memory)
                next_token_logits = logits[:, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for b in range(generated.shape[0]):
                        for token in generated[b].unique():
                            next_token_logits[b, token] /= repetition_penalty
                
                # Temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS (if defined)
                if hasattr(self.config, 'eos_token_id') and self.config.eos_token_id is not None:
                    if (next_token == self.config.eos_token_id).all():
                        break
        
        return generated

