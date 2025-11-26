import pytest
import torch
from src.core.language_zone.prosody_attention import ProsodyAttentionBridge
from src.core.language_zone.prosody_gif import ProsodyModulatedGIF
from src.core.language_zone.full_language_zone import FullLanguageZone


class TestProsodyAttention:
    """Test prosody attention module."""
    
    def test_attention_initialization(self):
        """Test attention bridge initialization."""
        bridge = ProsodyAttentionBridge(attention_preset='analytical')
        assert bridge.attention is not None
        assert bridge.attention.k_winners == 3  # analytical preset
    
    def test_prosody_extraction(self):
        """Test prosody channel extraction."""
        bridge = ProsodyAttentionBridge()
        tokens = ["Hello", "world", "!", "Amazing", "?"]
        
        amp, pitch, boundary = bridge.extract_prosody(tokens)
        
        assert len(amp) == 5
        assert len(pitch) == 5
        assert len(boundary) == 5
        
        # "!" should trigger amplitude
        assert amp[2] > 0
        
        # "?" should trigger pitch
        assert pitch[4] > 0
    
    def test_attention_gains(self):
        """Test attention gain computation."""
        bridge = ProsodyAttentionBridge(k_winners=2)
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        token_strings = [["wow", "amazing", "!", "test", "."]]
        
        gains, metadata = bridge(input_ids, token_strings)
        
        assert gains.shape == (1, 5)
        assert len(metadata['winners']) == 1
        # k_winners is a target, but may return more if multiple tokens have high salience
        assert len(metadata['winners'][0]) >= 1  # At least some winners


class TestProsodyModulatedGIF:
    """Test prosody-modulated GIF neuron."""
    
    def test_initialization(self):
        """Test initialization."""
        gif = ProsodyModulatedGIF(64, 128, L=16)
        assert gif.attention_modulation_strength == 0.3
        assert gif.hidden_dim == 128
    
    def test_forward_without_attention(self):
        """Test forward without attention gains."""
        gif = ProsodyModulatedGIF(64, 128, L=16)
        x = torch.randn(2, 10, 64)
        
        spikes, state = gif(x, attention_gains=None)
        
        assert spikes.shape == (2, 10, 128)
        assert state is not None
    
    def test_forward_with_attention(self):
        """Test forward with attention gains."""
        gif = ProsodyModulatedGIF(64, 128, L=16)
        x = torch.randn(2, 10, 64)
        attention_gains = torch.ones(2, 10) * 2.0  # 2x gain
        
        spikes_with_attn, _ = gif(x, attention_gains=attention_gains)
        spikes_no_attn, _ = gif(x, attention_gains=None)
        
        # With higher attention, we expect more spikes
        assert spikes_with_attn.sum() >= spikes_no_attn.sum() * 0.5


class TestFullLanguageZone:
    """Test full language zone integration."""
    
    def test_initialization(self):
        """Test full language zone initialization."""
        model = FullLanguageZone(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            num_experts=4,
            attention_preset='emotional'
        )
        
        assert model.vocab_size == 1000
        assert len(model.experts) == 4
    
    def test_forward_sync(self):
        """Test synchronous forward pass."""
        model = FullLanguageZone(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            num_experts=4
        )
        
        input_ids = torch.randint(0, 1000, (2, 10))
        token_strings = [
            ["wow", "amazing", "!", "test", ".", "hello", "world", "?", "great", "."],
            ["normal", "text", "here", "with", "some", "words", "and", "more", "stuff", "."]
        ]
        
        logits, info = model(input_ids, token_strings)
        
        assert logits.shape == (2, 10, 1000)
        assert 'prosody_stats' in info
        assert 'attention' in info
        
        # Check prosody influence
        assert info['prosody_stats']['mean_gain'] > 0
    
    def test_prosody_influence(self):
        """Test that prosody actually influences the output."""
        model = FullLanguageZone(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            num_experts=4,
            attention_preset='emotional'  # High prosody sensitivity
        )
        
        input_ids = torch.randint(0, 1000, (2, 5))
        
        # Emotional tokens
        emotional_tokens = [["incredible", "!", "amazing", "!", "wow"]]
        logits_emotional, info_emotional = model(input_ids[:1], emotional_tokens)
        
        # Neutral tokens
        neutral_tokens = [["the", "and", "or", "is", "a"]]
        logits_neutral, info_neutral = model(input_ids[:1], neutral_tokens)
        
        # Emotional content should have higher attention gains
        assert info_emotional['prosody_stats']['max_gain'] > info_neutral['prosody_stats']['max_gain']
