"""
Natural Brain Architecture.

Orchestrates:
1. Global Sensory Embedding
2. Endocrine System (Homeostasis)
3. Thalamus (Gating/Routing)
4. Limbic System (Emotion/Memory)
5. Neocortex (Specialized Regions including Language Zone)
6. Basal Ganglia (Integration)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from src.core.thalamus import Thalamus
from src.core.cortical_region import CorticalRegion
from src.core.language_zone.full_language_zone import FullLanguageZone
from src.core.limbic_system import LimbicSystem
from src.core.basal_ganglia import BasalGanglia
from src.core.endocrine import EndocrineSystem
from src.base.snn_brain_zones import BrainZoneConfig, BrainZoneType
from src.core.hippocampal import HippocampalFormation

class NaturalBrain(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 vocab_size: int,
                 zone_configs: Dict[str, BrainZoneConfig],
                 device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        region_names = list(zone_configs.keys())
        
        # 0. Sensory Cortex (Embeddings)
        # Converts raw tokens to neural signals
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 1. Homeostatic Core
        self.endocrine = EndocrineSystem()
        
        # 2. Thalamus
        self.thalamus = Thalamus(d_model, region_names).to(self.device)
        
        # 3. Limbic System
        self.hippocampus = HippocampalFormation(
            spatial_dimensions=2,
            n_place_cells=2000,
            feature_dim=d_model,
            device=self.device
        )
        self.limbic_system = LimbicSystem(d_model, self.hippocampus).to(self.device)
        
        # 4. Neocortex (Specialized Regions)
        self.cortex = nn.ModuleDict()
        for name, config in zone_configs.items():
            config.d_model = d_model
            config.name = name
            
            # Specialized Instantiation
            if config.zone_type == BrainZoneType.TEMPORAL_CORTEX:
                # Use the sophisticated Language Zone for Temporal lobe
                print(f"🧠 Initializing {name} as FullLanguageZone (Temporal Cortex)")
                self.cortex[name] = FullLanguageZone(config, vocab_size).to(self.device)
            else:
                # Use generic Neuromorphic Region for others
                self.cortex[name] = CorticalRegion(config).to(self.device)
            
        # 5. Basal Ganglia
        self.basal_ganglia = BasalGanglia(d_model, region_names).to(self.device)
        
        # 6. Output Head (Motor Cortex equivalent)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def update_homeostasis(self, metrics: Dict[str, float]):
        stats = {
            'accuracy': metrics.get('accuracy', 0.5),
            'gate_diversity': 0.8,
            'energy': 0.2
        }
        self.current_hormones = self.endocrine.step(stats)
        
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Full Brain Forward Pass.
        Args:
            input_ids: [Batch, Seq_Len]
        """
        if input_ids.device != self.device: input_ids = input_ids.to(self.device)
        
        # 0. Sensory Processing
        x = self.embedding(input_ids)
        
        # A. Get Hormone State
        hormones = getattr(self, 'current_hormones', {})
        
        # B. Limbic Assessment (Fast Path)
        limbic_out = self.limbic_system(x)
        emotional_state = limbic_out['emotional_state']
        
        # C. Thalamic Gating & Routing
        thalamus_mod = {
            'arousal': emotional_state['arousal'],
            'cortisol': hormones.get('cortisol', 0.0),
            'norepinephrine': hormones.get('norepinephrine', 0.0)
        }
        
        # routed_signals: Dict[region_name, Tensor[Batch, Seq, Dim]]
        routed_signals, routing_probs = self.thalamus(x, limbic_state=thalamus_mod)
        
        # D. Cortical Processing
        cortical_outputs = {}
        for region_name, region_input in routed_signals.items():
            dopamine = hormones.get('dopamine', 0.0)
            modulated_input = region_input * (1.0 + dopamine * 0.5)
            
            # Dispatch based on region type
            if isinstance(self.cortex[region_name], FullLanguageZone):
                # Pass input_ids for Prosody extraction
                out = self.cortex[region_name](modulated_input, input_ids=input_ids)
            else:
                # Standard region
                out = self.cortex[region_name](modulated_input)
                
            cortical_outputs[region_name] = out
            
        # E. Action Selection (Basal Ganglia)
        integrated_signal = self.basal_ganglia(cortical_outputs)
        
        # Skip connection
        if integrated_signal is not None:
            brain_output = x + integrated_signal
        else:
            brain_output = x
            
        # F. Motor Output
        logits = self.output_head(brain_output)
            
        return logits, {
            'routing': routing_probs,
            'emotion': emotional_state,
            'hormones': hormones
        }