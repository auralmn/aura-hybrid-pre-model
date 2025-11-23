#!/usr/bin/env python3
"""
Frequency-based phoneme pattern encoder for acoustic spike generation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

# International Phonetic Alphabet to acoustic features
IPA_ACOUSTICS = {
    # Vowels (F1, F2 frequencies in Hz)
    'i': (270, 2290),   # high front
    'ɪ': (390, 1990),   # near-high near-front
    'e': (530, 1840),   # mid front
    'ɛ': (660, 1720),   # mid-low front
    'æ': (860, 1720),   # low front
    'ɑ': (730, 1090),   # low back
    'ɔ': (570, 840),    # mid-low back
    'o': (450, 880),    # mid back
    'ʊ': (440, 1020),   # near-high back
    'u': (300, 870),    # high back
    'ə': (500, 1500),   # schwa (central)
    
    # Consonants (dominant frequency ranges)
    'p': (100, 500),    # bilabial stop
    'b': (100, 500),
    't': (4000, 8000),  # alveolar stop
    'd': (4000, 8000),
    'k': (2000, 4000),  # velar stop
    'g': (2000, 4000),
    'f': (6000, 12000), # fricatives
    's': (8000, 12000),
    'ʃ': (3000, 6000),  # "sh"
    'h': (500, 2000),   # aspiration
    'l': (200, 400),    # liquids
    'r': (300, 600),
    'm': (200, 300),    # nasals
    'n': (200, 300),
}

class FrequencyPatternEncoder(nn.Module):
    """
    Convert phoneme frequencies to addition-only spike patterns
    Maps IPA phonemes to acoustic formant frequencies to spike trains
    """
    
    def __init__(self, d_model: int = 256, sample_rate: int = 1000, 
                 duration_ms: int = 100, device: Optional[torch.device] = None):
        super().__init__()
        
        self.d_model = d_model
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.device = device or torch.device('cpu')
        
        # Precompute acoustic patterns for all phonemes
        self.phoneme_patterns = self._precompute_patterns()
        
        # Learnable adaptation parameters
        self.amplitude_scale = nn.Parameter(torch.ones(len(IPA_ACOUSTICS)))
        self.frequency_shift = nn.Parameter(torch.zeros(len(IPA_ACOUSTICS)))
        
        # Pattern combination weights
        self.f1_weight = nn.Parameter(torch.tensor(1.0))
        self.f2_weight = nn.Parameter(torch.tensor(0.5))
        
        # Move to device
        self.to(self.device)
    
    def _precompute_patterns(self) -> Dict[str, torch.Tensor]:
        """Precompute spike patterns for all phonemes"""
        patterns = {}
        samples = int(self.duration_ms * self.sample_rate / 1000)
        
        for phoneme, (f1, f2) in IPA_ACOUSTICS.items():
            pattern = self._frequency_to_spikes(f1, f2, samples)
            patterns[phoneme] = pattern
        
        return patterns
    
    def _frequency_to_spikes(self, f1: float, f2: float, samples: int) -> torch.Tensor:
        """Convert formant frequencies to addition-only spike pattern"""
        pattern = torch.zeros(samples)
        
        for i in range(samples):
            # Time step
            t = i / self.sample_rate
            
            # Addition-only sine approximation for F1 (primary formant)
            phase1 = (2 * np.pi * f1 * t) % (2 * np.pi)
            f1_component = self._sign_sine(phase1)
            
            # Addition-only sine approximation for F2 (secondary formant)
            phase2 = (2 * np.pi * f2 * t) % (2 * np.pi)
            f2_component = self._sign_sine(phase2)
            
            # Combine formants with addition only
            pattern[i] = f1_component + 0.5 * f2_component
        
        # Normalize to spike-like values
        pattern = torch.sign(pattern)
        
        # Pad or truncate to d_model size
        if samples < self.d_model:
            # Repeat pattern to fill d_model
            repeats = self.d_model // samples + 1
            pattern = pattern.repeat(repeats)[:self.d_model]
        else:
            pattern = pattern[:self.d_model]
        
        return pattern
    
    def _sign_sine(self, phase: float) -> float:
        """Addition-only sine approximation using sign patterns"""
        # Map phase to [-1, 1] using only addition/subtraction
        normalized = (phase / np.pi) - 1
        
        # Piecewise linear approximation of sine
        if -1 <= normalized <= -0.5:
            return -1 + 2 * (normalized + 1)  # Rising from -1 to 0
        elif -0.5 < normalized <= 0:
            return 2 * (normalized + 0.5)     # Rising from 0 to 1
        elif 0 < normalized <= 0.5:
            return 1 - 2 * normalized         # Falling from 1 to 0
        else:  # 0.5 < normalized <= 1
            return -2 * (normalized - 0.5)    # Falling from 0 to -1
    
    def encode_phoneme_sequence(self, phonemes: List[str]) -> torch.Tensor:
        """
        Encode a sequence of IPA phonemes to spike patterns
        Returns: [seq_len, d_model] tensor
        """
        if not phonemes:
            return torch.zeros(1, self.d_model, device=self.device)
        
        patterns = []
        
        for i, phoneme in enumerate(phonemes):
            if phoneme in self.phoneme_patterns:
                # Get base pattern
                base_pattern = self.phoneme_patterns[phoneme].to(self.device)
                
                # Apply learnable adaptations
                phoneme_idx = list(IPA_ACOUSTICS.keys()).index(phoneme)
                
                # Scale amplitude
                scaled_pattern = base_pattern * self.amplitude_scale[phoneme_idx]
                
                # Apply frequency shift (temporal shift approximation)
                shift = int(self.frequency_shift[phoneme_idx].item() * 10)  # Convert to samples
                if shift != 0:
                    scaled_pattern = torch.roll(scaled_pattern, shift)
                
                patterns.append(scaled_pattern)
            else:
                # Unknown phoneme - use neutral pattern
                patterns.append(torch.zeros(self.d_model, device=self.device))
        
        return torch.stack(patterns)  # [seq_len, d_model]
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Simple orthographic to IPA conversion
        In production, would use proper G2P (grapheme-to-phoneme) conversion
        """
        # Very basic mapping for demonstration
        simple_mapping = {
            'a': 'ɑ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɔ', 'u': 'ʊ',
            'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
            'f': 'f', 's': 's', 'h': 'h', 'l': 'l', 'r': 'r', 'm': 'm', 'n': 'n'
        }
        
        phonemes = []
        for char in text.lower():
            if char in simple_mapping:
                phonemes.append(simple_mapping[char])
            elif char.isalpha():
                phonemes.append('ə')  # Default to schwa
        
        return phonemes
    
    def forward(self, input_data) -> torch.Tensor:
        """
        Forward pass - accepts either phoneme list or text string
        """
        if isinstance(input_data, str):
            # Convert text to phonemes first
            phonemes = self.text_to_phonemes(input_data)
            return self.encode_phoneme_sequence(phonemes)
        elif isinstance(input_data, list):
            # Direct phoneme sequence
            return self.encode_phoneme_sequence(input_data)
        else:
            raise ValueError("Input must be string (text) or list (phonemes)")
    
    def get_acoustic_features(self, phoneme: str) -> Optional[Tuple[float, float]]:
        """Get the acoustic features (F1, F2) for a phoneme"""
        return IPA_ACOUSTICS.get(phoneme)
    
    def get_supported_phonemes(self) -> List[str]:
        """Get list of supported IPA phonemes"""
        return list(IPA_ACOUSTICS.keys())

def ipa_to_frequencies(ipa_string: str) -> List[Tuple[float, float]]:
    """Convert IPA string to frequency components (utility function)"""
    frequencies = []
    for char in ipa_string:
        if char in IPA_ACOUSTICS:
            frequencies.append(IPA_ACOUSTICS[char])
    return frequencies
