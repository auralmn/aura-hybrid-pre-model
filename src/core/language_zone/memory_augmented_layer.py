"""
Memory-Augmented Transformer Layer

Implements Retrieval-Augmented Generation (RAG) using the hippocampal memory system.
Retrieves relevant memories and injects them into the attention computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import uuid

from src.core.language_zone.hippocampal_attention import HippocampalProsodyAttention
from src.core.language_zone.snn_ffn import SNNFFN, HybridFFN


class MemoryAugmentedLayer(nn.Module):
    """
    Transformer layer with explicit memory retrieval and injection.
    
    Extends HippocampalTransformerLayer with:
    1. Query-based memory retrieval from hippocampal formation
    2. Memory injection via cross-attention or concatenation
    3. Optional SNN-based FFN
    4. Memory storage during training
    """
    
    def __init__(
        self,
        config,
        hippocampus,
        use_snn_ffn: bool = False,
        memory_injection: str = "cross_attention",  # "cross_attention", "concat", "gate"
        num_retrieved: int = 5
    ):
        super().__init__()
        self.config = config
        self.hippocampus = hippocampus
        self.memory_injection = memory_injection
        self.num_retrieved = num_retrieved
        self.use_snn_ffn = use_snn_ffn
        
        # 1. Self-Attention with Prosody
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = HippocampalProsodyAttention(config, hippocampus)
        
        # 2. Memory Cross-Attention (if using cross_attention injection)
        if memory_injection == "cross_attention":
            self.memory_norm = nn.LayerNorm(config.embedding_dim)
            self.memory_attention = nn.MultiheadAttention(
                embed_dim=config.embedding_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True
            )
        elif memory_injection == "gate":
            self.memory_gate = nn.Sequential(
                nn.Linear(config.embedding_dim * 2, config.embedding_dim),
                nn.Sigmoid()
            )
            self.memory_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # 3. Feed-Forward (Standard or SNN)
        self.ffn_norm = nn.LayerNorm(config.embedding_dim)
        if use_snn_ffn:
            self.ffn = HybridFFN(
                input_dim=config.embedding_dim,
                hidden_dim=config.intermediate_size,
                snn_ratio=0.5,
                dropout=config.dropout
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.embedding_dim, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.embedding_dim),
                nn.Dropout(config.dropout)
            )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Memory query projection
        self.query_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
    def retrieve_memories(
        self,
        hidden_states: torch.Tensor,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant memories based on current hidden states.
        
        Args:
            hidden_states: [Batch, Seq, Dim]
            k: Number of memories to retrieve
            
        Returns:
            memory_features: [Batch, K, Dim] retrieved memory features
            memory_scores: [Batch, K] retrieval scores
        """
        batch_size, seq_len, dim = hidden_states.shape
        device = hidden_states.device
        
        # Use mean pooled representation as query
        query = self.query_proj(hidden_states.mean(dim=1))  # [B, D]
        
        # Initialize output tensors
        memory_features = torch.zeros(batch_size, k, dim, device=device, dtype=hidden_states.dtype)
        memory_scores = torch.zeros(batch_size, k, device=device, dtype=hidden_states.dtype)
        
        # Retrieve for each batch item
        for b in range(batch_size):
            query_b = query[b]  # [D]
            
            # Use hippocampal retrieval
            if self.hippocampus is not None and self.hippocampus.memory_count > 0:
                results = self.hippocampus.retrieve_similar_memories(
                    query_features=query_b,
                    k=k
                )
                
                # Extract features for retrieved memories
                for i, (mem_id, score) in enumerate(results):
                    if mem_id in self.hippocampus.id_to_idx:
                        idx = self.hippocampus.id_to_idx[mem_id]
                        memory_features[b, i] = self.hippocampus.memory_features[idx]
                        memory_scores[b, i] = score
        
        return memory_features, memory_scores
    
    def store_memory(self, hidden_states: torch.Tensor):
        """
        Store current context in hippocampal memory.
        
        Args:
            hidden_states: [Batch, Seq, Dim]
        """
        if self.hippocampus is None:
            return
            
        batch_size = hidden_states.shape[0]
        
        # Store mean representation of each batch item
        for b in range(batch_size):
            features = hidden_states[b].mean(dim=0).detach()  # [D]
            memory_id = str(uuid.uuid4())[:8]
            
            self.hippocampus.create_episodic_memory(
                memory_id=memory_id,
                event_id=f"layer_{id(self)}",
                features=features
            )
    
    def inject_memories(
        self,
        hidden_states: torch.Tensor,
        memory_features: torch.Tensor,
        memory_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Inject retrieved memories into hidden states.
        
        Args:
            hidden_states: [Batch, Seq, Dim]
            memory_features: [Batch, K, Dim]
            memory_scores: [Batch, K]
            
        Returns:
            augmented_states: [Batch, Seq, Dim]
        """
        if self.memory_injection == "cross_attention":
            # Cross-attention: hidden_states attend to memories
            normed = self.memory_norm(hidden_states)
            attn_out, _ = self.memory_attention(
                query=normed,
                key=memory_features,
                value=memory_features
            )
            return hidden_states + self.dropout(attn_out)
            
        elif self.memory_injection == "concat":
            # Concatenate memories and project back
            # Mean of memories weighted by scores
            weights = F.softmax(memory_scores, dim=-1).unsqueeze(-1)  # [B, K, 1]
            memory_context = (memory_features * weights).sum(dim=1, keepdim=True)  # [B, 1, D]
            memory_context = memory_context.expand(-1, hidden_states.shape[1], -1)  # [B, S, D]
            return hidden_states + 0.1 * memory_context
            
        elif self.memory_injection == "gate":
            # Gated injection
            weights = F.softmax(memory_scores, dim=-1).unsqueeze(-1)
            memory_context = (memory_features * weights).sum(dim=1, keepdim=True)
            memory_context = memory_context.expand(-1, hidden_states.shape[1], -1)
            memory_context = self.memory_proj(memory_context)
            
            # Compute gate
            gate_input = torch.cat([hidden_states, memory_context], dim=-1)
            gate = self.memory_gate(gate_input)
            
            return hidden_states + gate * memory_context
        
        return hidden_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        prosody: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        store_memory: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with memory retrieval and injection.
        
        Args:
            hidden_states: [Batch, Seq, Dim]
            prosody: [Batch, Seq, 4] prosody features
            use_memory: Whether to retrieve and inject memories
            store_memory: Whether to store current context in memory
        """
        # 1. Self-Attention Block
        normed_hidden = self.attention_norm(hidden_states)
        attn_output, _ = self.attention(
            normed_hidden,
            prosody=prosody,
            use_memory=use_memory
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # 2. Memory Retrieval and Injection (RAG)
        if use_memory and self.hippocampus is not None and self.hippocampus.memory_count > 0:
            memory_features, memory_scores = self.retrieve_memories(
                hidden_states,
                k=self.num_retrieved
            )
            hidden_states = self.inject_memories(
                hidden_states,
                memory_features,
                memory_scores
            )
        
        # 3. Feed-Forward Block (Standard or SNN)
        normed_hidden = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(normed_hidden)
        hidden_states = hidden_states + ffn_output
        
        # 4. Store memory if training
        if store_memory and self.training:
            self.store_memory(hidden_states)
        
        return hidden_states

