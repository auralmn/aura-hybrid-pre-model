#!/usr/bin/env python3
"""
Dual-layer Addition-Only SRFFN with semantic and phonetic processing
"""

from encoders.fast_event_encoder import FastEventPatternEncoder
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maths.addition_linear import AdditionLinear
from maths.additive_receptance import AdditiveReceptance
from maths.sign_activation import SignActivation

from maths.frequency_encoder import FrequencyPatternEncoder

class DualLayerSRFFN(nn.Module):
    """
    Dual-layer Addition-Only SRFFN combining:
    - Semantic layer: Event-based keyword patterns (war, cultural, etc.)
    - Phonetic layer: Frequency-based acoustic patterns (IPA phonemes)
    
    Enables "voice-aware" text processing where content meaning affects
    acoustic realization, mimicking human reading comprehension.
    """

    previous_semantic_activation: torch.Tensor
    
    def __init__(self, module_id: str, config: Dict[str, Any], 
                 keyword_file: Optional[str] = None):
        super().__init__()
        
        # Core dimensions
        self.d_model = config.get('d_model', 256)
        self.d_ff = config.get('d_ff', 1024)
        self.dropout_rate = config.get('dropout', 0.1)
        
        # Spiking parameters
        self.spike_threshold = config.get('spike_threshold', 0.0)
        self.decay_factor = config.get('decay_factor', 0.9)
        
        # Voice-aware processing
        self.enable_voice_synthesis = config.get('enable_voice_synthesis', True)
        
        # Device management
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Semantic processing layer (your existing system)
        self.semantic_encoder = None
        if keyword_file:
            self.semantic_encoder = FastEventPatternEncoder(keyword_file, self.device)
        
        # Phonetic processing layer (new acoustic layer)
        self.phonetic_encoder = FrequencyPatternEncoder(
            d_model=self.d_model,
            device=self.device
        )
        
        # Addition-only network layers
        self.w1 = AdditionLinear(self.d_model, self.d_ff, bias=False)
        self.w2 = AdditionLinear(self.d_ff, self.d_model, bias=False)
        
        # Dual receptance gating (semantic + phonetic)
        self.semantic_receptance = AdditiveReceptance(self.d_model, self.d_ff)
        self.phonetic_receptance = AdditiveReceptance(self.d_model, self.d_ff)
        
        # Cross-modal fusion layer
        self.fusion_layer = AdditionLinear(self.d_ff * 2, self.d_model, bias=False)
        # Sign-based activations
        self.semantic_activation = SignActivation(threshold=self.spike_threshold)
        self.phonetic_activation = SignActivation(threshold=self.spike_threshold * 0.5)
        
        # Voice synthesis parameters (learnable)
        if self.enable_voice_synthesis:
            self.voice_pitch_scale = nn.Parameter(torch.tensor(1.0))
            self.voice_speed_scale = nn.Parameter(torch.tensor(1.0))
            self.voice_intensity_scale = nn.Parameter(torch.tensor(1.0))
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Temporal state buffers for both streams
        self.register_buffer('prev_semantic_activation', torch.zeros(1, self.d_ff))
        self.register_buffer('prev_phonetic_activation', torch.zeros(1, self.d_ff))
        self.register_buffer('prev_semantic_receptance', torch.zeros(1, self.d_ff))
        self.register_buffer('prev_phonetic_receptance', torch.zeros(1, self.d_ff))
        self.register_buffer('adaptation_state', torch.zeros(1))
        
        # Move everything to device
        self.to(self.device)
    
    def temporal_mixing_additive(self, current: torch.Tensor, 
                                previous: torch.Tensor) -> torch.Tensor:
        """Addition-only temporal mixing for both streams"""
        diff = current - previous
        mix_signs = torch.sign(diff)
        
        # Addition-only mixing based on sign patterns
        mixed = torch.where(mix_signs > 0, current, previous)
        complement = torch.where(mix_signs > 0, previous, current)
        result = mixed + 0.1 * complement
        
        return result
    
    def cross_modal_fusion(self, semantic_features: torch.Tensor, 
                          phonetic_features: torch.Tensor) -> torch.Tensor:
        """Fuse semantic and phonetic features using addition only"""
        # Concatenate features
        combined = torch.cat([semantic_features, phonetic_features], dim=-1)

        # The fusion layer should map from d_ff*2 -> d_model
        fused = self.fusion_layer(combined)

        return fused
    
    def extract_voice_characteristics(self, text: str, 
                                    semantic_pattern: torch.Tensor) -> Dict[str, float]:
        """Extract voice characteristics from semantic content"""
        if not self.enable_voice_synthesis:
            return {}
        
        voice_params = {
            'pitch_factor': 1.0,
            'speed_factor': 1.0,
            'intensity_factor': 1.0,
            'timbre': 'neutral'
        }
        
        if self.semantic_encoder:
            analysis = self.semantic_encoder.get_event_analysis(text)
            detected_events = analysis.get('detected_events', [])
            
            for event in detected_events[:3]:  # Top 3 events
                event_type = event['event_type']
                score = event['normalized_score']
                
                # Adjust voice based on event type
                if event_type == 'war':
                    voice_params['pitch_factor'] -= 0.2 * score  # Lower pitch
                    voice_params['intensity_factor'] += 0.3 * score  # More intense
                    voice_params['timbre'] = 'harsh'
                elif event_type == 'cultural':
                    voice_params['pitch_factor'] += 0.1 * score  # Slightly higher
                    voice_params['speed_factor'] -= 0.1 * score  # Slower, expressive
                    voice_params['timbre'] = 'melodic'
                elif event_type == 'religious':
                    voice_params['pitch_factor'] -= 0.1 * score  # Reverent tone
                    voice_params['speed_factor'] -= 0.2 * score  # Slower pace
                    voice_params['timbre'] = 'solemn'
                elif event_type == 'technological':
                    voice_params['speed_factor'] += 0.1 * score  # Precise delivery
                    voice_params['timbre'] = 'precise'
        
        return voice_params
    
    def forward(self, text: str, phonemes: Optional[List[str]] = None, 
                include_voice_params: bool = False) -> Dict[str, torch.Tensor]:
        """
        Dual-stream forward pass with semantic and phonetic processing
        
        Args:
            text: Input text string
            phonemes: Optional IPA phoneme sequence
            include_voice_params: Whether to extract voice characteristics
            
        Returns:
            Dictionary containing processed outputs and voice parameters
        """
        batch_size = 1
        
        # Resize state buffers if needed
        if self.prev_semantic_activation.size(0) != batch_size:
            self.prev_semantic_activation = torch.zeros(batch_size, self.d_ff, device=self.device)
            self.prev_phonetic_activation = torch.zeros(batch_size, self.d_ff, device=self.device)
            self.prev_semantic_receptance = torch.zeros(batch_size, self.d_ff, device=self.device)
            self.prev_phonetic_receptance = torch.zeros(batch_size, self.d_ff, device=self.device)
        
        # === SEMANTIC STREAM ===
        semantic_pattern = torch.zeros(1, self.d_model, device=self.device)
        if self.semantic_encoder and text:
            semantic_pattern = self.semantic_encoder.encode_text_to_patterns(text)
        
        # === PHONETIC STREAM ===
        phonetic_pattern = torch.zeros(1, self.d_model, device=self.device)
        if phonemes:
            phonetic_sequence = self.phonetic_encoder.encode_phoneme_sequence(phonemes)
            # Average across sequence for single pattern
            phonetic_pattern = torch.mean(phonetic_sequence, dim=0, keepdim=True)
        elif text:
            # Convert text to phonemes and process
            phonetic_pattern = self.phonetic_encoder(text)
            if phonetic_pattern.dim() > 2:
                phonetic_pattern = torch.mean(phonetic_pattern, dim=0, keepdim=True)
        
        # === DUAL PROCESSING ===
        
        # Semantic receptance gating
        semantic_receptance = self.semantic_receptance(semantic_pattern)
        mixed_sem_recept = self.temporal_mixing_additive(
            semantic_receptance, self.prev_semantic_receptance
        )
        
        # Phonetic receptance gating
        phonetic_receptance = self.phonetic_receptance(phonetic_pattern)
        mixed_phon_recept = self.temporal_mixing_additive(
            phonetic_receptance, self.prev_phonetic_receptance
        )
        
        # Feed-forward processing for both streams
        semantic_hidden = self.w1(semantic_pattern)
        phonetic_hidden = self.w1(phonetic_pattern)
        
        # Temporal mixing
        mixed_sem_hidden = self.temporal_mixing_additive(
            semantic_hidden, self.prev_semantic_activation
        )
        mixed_phon_hidden = self.temporal_mixing_additive(
            phonetic_hidden, self.prev_phonetic_activation
        )
        
        # Spike activations
        semantic_spikes = self.semantic_activation(mixed_sem_hidden)
        phonetic_spikes = self.phonetic_activation(mixed_phon_hidden)
        
        # Apply receptance gating (addition-only)
        sem_gating_signs = torch.sign(mixed_sem_recept)
        phon_gating_signs = torch.sign(mixed_phon_recept)
        
        gated_semantic = torch.where(sem_gating_signs > 0, semantic_spikes, -semantic_spikes)
        gated_phonetic = torch.where(phon_gating_signs > 0, phonetic_spikes, -phonetic_spikes)
        
        # Cross-modal fusion
        semantic_squeezed = gated_semantic.squeeze(0)  # [d_ff]
        phonetic_squeezed = gated_phonetic.squeeze(0)  # [d_ff]
        fused_features = semantic_squeezed + phonetic_squeezed  # [d_ff]
        
        # Dropout
        if self.training:
            fused_features = self.dropout(fused_features)
        
        # Final output projection
        output = self.w2(fused_features.unsqueeze(0))
        
        # Update states
        self.prev_semantic_activation = mixed_sem_hidden.detach()
        self.prev_phonetic_activation = mixed_phon_hidden.detach()
        self.prev_semantic_receptance = mixed_sem_recept.detach()
        self.prev_phonetic_receptance = mixed_phon_recept.detach()
        
        # Update adaptation state
        processing_intensity = torch.sum(torch.abs(output)) / output.numel()
        self.adaptation_state = (self.adaptation_state * 0.9 + 
                               processing_intensity.detach() * 0.1)
        
        # Prepare output dictionary
        result = {
            'output': output,
            'semantic_pattern': semantic_pattern,
            'phonetic_pattern': phonetic_pattern,
            'fused_features': fused_features.unsqueeze(0)
        }
        
        # Add voice characteristics if requested
        if include_voice_params:
            voice_params = self.extract_voice_characteristics(text, semantic_pattern)
            result['voice_parameters'] = voice_params
        
        return result
    
    def read_with_voice(self, text: str, phonemes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process text and return both neural output and voice synthesis parameters
        This is the main interface for "voice-aware reading"
        """
        with torch.no_grad():
            result = self.forward(text, phonemes, include_voice_params=True)
            
            # Extract voice characteristics
            voice_params = result.get('voice_parameters', {})
            
            # Add neural processing statistics
            output_stats = {
                'semantic_intensity': float(torch.sum(torch.abs(result['semantic_pattern']))),
                'phonetic_intensity': float(torch.sum(torch.abs(result['phonetic_pattern']))),
                'fusion_intensity': float(torch.sum(torch.abs(result['fused_features']))),
                'overall_intensity': float(torch.sum(torch.abs(result['output'])))
            }
            
            return {
                'text': text,
                'voice_parameters': voice_params,
                'neural_output': result['output'],
                'processing_stats': output_stats,
                'phonemes_used': phonemes if phonemes else self.phonetic_encoder.text_to_phonemes(text)
            }
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get comprehensive network information"""
        return {
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'spike_threshold': self.spike_threshold,
            'decay_factor': self.decay_factor,
            'enable_voice_synthesis': self.enable_voice_synthesis,
            'semantic_encoder_active': self.semantic_encoder is not None,
            'phonetic_encoder_active': True,
            'supported_phonemes': len(self.phonetic_encoder.get_supported_phonemes()),
            'operation_type': 'addition_only_dual_stream',
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'multiplication_operations': 0
        }
