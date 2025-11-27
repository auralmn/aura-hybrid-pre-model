#!/usr/bin/env python3
"""
Vectorized Frequency-based phoneme pattern encoder.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

# [Keep IPA_ACOUSTICS dictionary unchanged]
IPA_ACOUSTICS = {
    'i': (270, 2290), 'ɪ': (390, 1990), 'e': (530, 1840), 'ɛ': (660, 1720),
    'æ': (860, 1720), 'ɑ': (730, 1090), 'ɔ': (570, 840), 'o': (450, 880),
    'ʊ': (440, 1020), 'u': (300, 870), 'ə': (500, 1500),
    'p': (100, 500), 'b': (100, 500), 't': (4000, 8000), 'd': (4000, 8000),
    'k': (2000, 4000), 'g': (2000, 4000), 'f': (6000, 12000), 's': (8000, 12000),
    'ʃ': (3000, 6000), 'h': (500, 2000), 'l': (200, 400), 'r': (300, 600),
    'm': (200, 300), 'n': (200, 300),
}

class FrequencyPatternEncoder(nn.Module):
    """
    Convert phoneme frequencies to addition-only spike patterns (Vectorized).
    """
    
    def __init__(self, d_model: int = 256, sample_rate: int = 1000, 
                 duration_ms: int = 100, device: Optional[torch.device] = None):
        super().__init__()
        
        self.d_model = d_model
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Precompute time vector [Samples]
        self.num_samples = int(self.duration_ms * self.sample_rate / 1000)
        t = torch.linspace(0, self.duration_ms / 1000, self.num_samples, device=self.device)
        self.register_buffer('t', t)
        
        # Pattern storage
        self.phoneme_patterns = {}
        self._precompute_patterns()
        
        # Parameters
        self.amplitude_scale = nn.Parameter(torch.ones(len(IPA_ACOUSTICS)))
        
        self.to(self.device)
    
    def _precompute_patterns(self):
        """Vectorized pattern generation."""
        # We can't precompute everything as tensors in __init__ if we want gradients
        # But we can store base patterns.
        
        for phoneme, (f1, f2) in IPA_ACOUSTICS.items():
            # Vectorized sine generation
            # phase = 2 * pi * f * t
            p1 = (2 * np.pi * f1 * self.t) % (2 * np.pi)
            p2 = (2 * np.pi * f2 * self.t) % (2 * np.pi)
            
            # Apply sign approximation
            s1 = self._sign_sine_vectorized(p1)
            s2 = self._sign_sine_vectorized(p2)
            
            pattern = torch.sign(s1 + 0.5 * s2)
            
            # Pad/Repeat to d_model
            if self.num_samples < self.d_model:
                repeats = self.d_model // self.num_samples + 1
                pattern = pattern.repeat(repeats)[:self.d_model]
            else:
                pattern = pattern[:self.d_model]
                
            self.phoneme_patterns[phoneme] = pattern

    def _sign_sine_vectorized(self, phase: torch.Tensor) -> torch.Tensor:
        """Vectorized addition-only sine approximation."""
        norm = (phase / np.pi) - 1.0
        
        # Masking for piecewise logic
        # -1 <= x <= -0.5: -1 + 2(x+1)
        # -0.5 < x <= 0: 2(x+0.5)
        # 0 < x <= 0.5: 1 - 2x
        # 0.5 < x <= 1: -2(x-0.5)
        
        res = torch.zeros_like(norm)
        
        m1 = (norm <= -0.5)
        m2 = (norm > -0.5) & (norm <= 0)
        m3 = (norm > 0) & (norm <= 0.5)
        m4 = (norm > 0.5)
        
        res[m1] = -1 + 2 * (norm[m1] + 1)
        res[m2] = 2 * (norm[m2] + 0.5)
        res[m3] = 1 - 2 * norm[m3]
        res[m4] = -2 * (norm[m4] - 0.5)
        
        return res

    def forward(self, input_data) -> torch.Tensor:
        """Forward pass with phoneme encoding."""
        # ... [Keep original text-to-phoneme logic if needed] ...
        # Assume input_data is list of phonemes for simplicity here
        if isinstance(input_data, str):
            # Stub for conversion
            input_data = ['ə'] * (len(input_data) // 2) 
            
        patterns = []
        for p in input_data:
            pat = self.phoneme_patterns.get(p, self.phoneme_patterns['ə'])
            patterns.append(pat)
            
        if not patterns:
            return torch.zeros(1, self.d_model, device=self.device)
            
        return torch.stack(patterns)