"""
Natural Brain Architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import torch.utils.checkpoint as checkpoint

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
                 device: str = 'cuda',
                 use_checkpointing: bool = False): # <--- NEW FLAG
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_checkpointing = use_checkpointing
        
        region_names = list(zone_configs.keys())
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.endocrine = EndocrineSystem()
        self.thalamus = Thalamus(d_model, region_names).to(self.device)
        
        self.hippocampus = HippocampalFormation(
            spatial_dimensions=2, 
            n_place_cells=2000, 
            feature_dim=d_model, 
            device=self.device
        )
        self.limbic_system = LimbicSystem(d_model, self.hippocampus).to(self.device)
        
        self.cortex = nn.ModuleDict()
        for name, config in zone_configs.items():
            config.d_model = d_model
            config.name = name
            if config.zone_type == BrainZoneType.TEMPORAL_CORTEX:
                self.cortex[name] = FullLanguageZone(config, vocab_size).to(self.device)
            else:
                self.cortex[name] = CorticalRegion(config).to(self.device)
            
        self.basal_ganglia = BasalGanglia(d_model, region_names).to(self.device)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def update_homeostasis(self, metrics: Dict[str, float]):
        stats = {'accuracy': metrics.get('accuracy', 0.5), 'gate_diversity': 0.8, 'energy': 0.2}
        self.current_hormones = self.endocrine.step(stats)
        
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if input_ids.device != self.device: 
            input_ids = input_ids.to(self.device)
        
        # 0. Sensory
        x = self.embedding(input_ids)
        
        # A. Limbic
        hormones = getattr(self, 'current_hormones', {})
        limbic_out = self.limbic_system(x)
        emotional_state = limbic_out['emotional_state']
        
        # B. Thalamus
        thalamus_mod = {
            'arousal': emotional_state['arousal'],
            'cortisol': hormones.get('cortisol', 0.0),
            'norepinephrine': hormones.get('norepinephrine', 0.0)
        }
        routed_signals, routing_probs = self.thalamus(x, limbic_state=thalamus_mod)
        
        # C. Cortex
        cortical_outputs = {}
        dopamine = hormones.get('dopamine', 0.0)
        
        for region_name, region_input in routed_signals.items():
            modulated_input = region_input * (1.0 + dopamine * 0.5)
            module = self.cortex[region_name]
            
            # Checkpointing Logic
            if self.use_checkpointing and modulated_input.requires_grad:
                if isinstance(module, FullLanguageZone):
                    out = checkpoint.checkpoint(module, modulated_input, input_ids, use_reentrant=False)
                else:
                    out = checkpoint.checkpoint(module, modulated_input, use_reentrant=False)
            else:
                # Standard execution
                if isinstance(module, FullLanguageZone):
                    out = module(modulated_input, input_ids=input_ids)
                else:
                    out = module(modulated_input)
                    
            cortical_outputs[region_name] = out
            
        # D. Basal Ganglia
        final_output = self.basal_ganglia(cortical_outputs)
        
        # Residual connection with scaling for stability
        if final_output is not None:
            output = x + 0.1 * final_output
        else:
            output = x
        
        logits = self.output_head(output)
            
        return logits, {
            'routing': routing_probs,
            'emotion': emotional_state,
            'hormones': hormones
        }