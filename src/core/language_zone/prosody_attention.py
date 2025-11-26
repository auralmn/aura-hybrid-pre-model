import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .multi_channel_attention import MultiChannelSpikingAttention, prosody_channels_from_text, AttentionPresets


class ProsodyAttentionBridge(nn.Module):
    """
    Bridge between text input and Multi-Channel Spiking Attention.
    
    Converts tokenized text → prosody channels → attention gains
    that modulate SNN dynamics.
    """
    
    def __init__(
        self,
        attention_preset: str = 'analytical',  # 'analytical', 'emotional', 'historical', 'streaming'
        k_winners: int = 5,
        device: str = 'cpu'
    ):
        super().__init__()
        
        # Initialize spiking attention
        if attention_preset == 'analytical':
            self.attention = AttentionPresets.analytical()
        elif attention_preset == 'emotional':
            self.attention = AttentionPresets.emotional()
        elif attention_preset == 'historical':
            self.attention = AttentionPresets.historical()
        elif attention_preset == 'streaming':
            self.attention = AttentionPresets.streaming()
        else:
            # Custom configuration
            self.attention = MultiChannelSpikingAttention(k_winners=k_winners)
        
        self.device = device
        
        # Cache for efficiency
        self._token_cache = {}
    
    def extract_prosody(
        self,
        token_strings: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract prosody channels from token strings.
        
        Args:
            token_strings: List of token strings (e.g., ["Hello", "world", "!"])
        
        Returns:
            (amplitude, pitch, boundary) channels
        """
        # Use cached channels if available
        cache_key = tuple(token_strings)
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]
        
        # Extract prosody
        amp, pitch, boundary = prosody_channels_from_text(token_strings)
        
        # Cache for reuse
        self._token_cache[cache_key] = (amp, pitch, boundary)
        
        return amp, pitch, boundary
    
    def compute_attention_gains(
        self,
        token_ids: List[int],
        token_strings: List[str],
        feature_size: Optional[int] = None,
        use_stateful: bool = False
    ) -> Dict[str, any]:
        """
        Compute attention gains from prosody channels.
        
        Args:
            token_ids: List of token IDs
            token_strings: List of token strings (for prosody extraction)
            feature_size: Optional feature vector size for per-feature gains
            use_stateful: Use stateful LIF (for streaming)
        
        Returns:
            Dict with attention gains and metadata
        """
        # Extract prosody channels
        amp, pitch, boundary = self.extract_prosody(token_strings)
        
        # Build token-to-feature mapping if needed
        token_to_feature = None
        if feature_size is not None:
            token_to_feature = {tid: i % feature_size for i, tid in enumerate(token_ids)}
        
        # Compute spiking attention
        result = self.attention.compute(
            token_ids=token_ids,
            amp=amp,
            pitch=pitch,
            boundary=boundary,
            feature_size=feature_size,
            token_to_feature=token_to_feature,
            use_stateful=use_stateful
        )
        
        return result
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_strings: Optional[List[List[str]]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute attention gains for a batch of sequences.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            token_strings: Optional list of token string lists per batch
        
        Returns:
            attention_gains: (batch, seq_len) tensor
            metadata: Dict with salience, winners, etc.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize gains
        attention_gains = torch.ones(batch_size, seq_len, device=device)
        
        metadata = {
            'salience': [],
            'winners': [],
            'spikes': []
        }
        
        # Process each sequence in batch
        for b in range(batch_size):
            token_ids_b = input_ids[b].cpu().numpy().tolist()
            
            # Get token strings (or use placeholder)
            if token_strings is not None and b < len(token_strings):
                tokens_b = token_strings[b]
            else:
                # Fallback: use token IDs as strings
                tokens_b = [str(tid) for tid in token_ids_b]
            
            # Compute attention
            result = self.compute_attention_gains(
                token_ids=token_ids_b,
                token_strings=tokens_b,
                use_stateful=False
            )
            
            # Convert salience to tensor
            salience_b = torch.from_numpy(result['salience']).float().to(device)
            
            # Apply mu_scalar as baseline, then modulate by salience
            mu_scalar = result['mu_scalar']
            attention_gains[b] = mu_scalar * (1.0 + salience_b)  # Amplify by salience
            
            # Store metadata
            metadata['salience'].append(salience_b)
            metadata['winners'].append(result['winners_idx'])
            metadata['spikes'].append(result['spikes'])
        
        return attention_gains, metadata
    
    def reset_state(self):
        """Reset LIF state for streaming."""
        self.attention.reset_state()
        self._token_cache.clear()
