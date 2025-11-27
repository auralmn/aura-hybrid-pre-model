#!/usr/bin/env python3
"""
Enhanced brain.py integration example
Shows how to integrate all the enhanced components with your existing Brain class
"""

from typing import Dict, List, Optional, Any, Tuple, Literal
import torch
import torch.nn as nn
import numpy as np
from enum import Enum
from dataclasses import dataclass
import logging
from scipy.signal import hilbert
from scipy.linalg import expm
import glob
import time
import os
import asyncio

# Your existing imports
from .layers_factory import LayersFactory
from .neuron_factory import NeuronFactory
from base.snn_layers import BaseLayerConfig, BaseLayerContainer, BaseLayerContainerConfig
from base.neuron import NeuronalState
from base.snn_brain_zones import BrainZone, BrainZoneConfig
from base.brain_zone_factory import BrainZoneFactory

# Enhanced imports
from base.snn_brain_zones import (
    NeuromorphicBrainZone,
    BrainZoneType,
)
from base.snn_brain_stats import BrainStats, StatsCollector
from base.snn_processor import NeuromorphicProcessor, ContentRouter, ProcessingMode
from base.snn_layers import BaseLayerFactory, create_neuromorphic_layer_stack
from base.events import EventBus

# Liquid MoE and Learning Components
from core.liquid_moe import LiquidMoERouter
from core.experts import ExpertHead, NLMSExpertAdapter
from training.hebbian_layer import OjaLayer
from training.optimized_whitener import OptimizedWhitener
from encoders.fast_hash_embedder import FastHashEmbedder
from training.memory_manager import maybe_empty_cuda_cache

# Optional continuous learning orchestrator
try:
    from services.continuous_learning import (
        ContinuousLearningOrchestrator,
        create_default_feeds,
        RSSFeedConfig,
    )
except Exception:
    ContinuousLearningOrchestrator = None  # type: ignore
    create_default_feeds = None  # type: ignore
    RSSFeedConfig = None  # type: ignore

"""
Enhanced Brain (GPU-Optimized Integration).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# Use optimized components
from core.liquid_moe import LiquidMoERouter
from base.snn_brain_zones import NeuromorphicBrainZone, BrainZoneConfig, BrainZoneType
from base.events import EventBus

class EnhancedBrain(nn.Module):
    """
    Main Brain Module.
    Integrates multiple Brain Zones (Cortical, Hippocampal, etc.) on GPU.
    """
    def __init__(self, 
                 d_model: int = 1024, 
                 zones_config: Dict[str, BrainZoneConfig] = None,
                 device: str = 'cuda'):
        super().__init__()
        self.d_model = d_model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.event_bus = EventBus()
        
        # Initialize Zones as ModuleDict (automatically registered)
        self.zones = nn.ModuleDict()
        
        if zones_config:
            for name, config in zones_config.items():
                # Ensure config matches model dim
                config.d_model = d_model
                config.name = name
                # Create GPU-native zone
                self.zones[name] = NeuromorphicBrainZone(config).to(self.device)
                
        # Global Router (Liquid State Machine)
        # Routes signals between zones based on content
        self.global_router = LiquidMoERouter(
            in_dim=d_model,
            hidden_dim=256,
            num_experts=len(self.zones) if self.zones else 1,
            top_k=min(3, len(self.zones))
        ).to(self.device)
        
        self.zone_names = list(self.zones.keys())

    def process_input(self, 
                      x: torch.Tensor, 
                      text_context: Optional[str] = None,
                      content_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        API-Compatible Processing Wrapper.
        
        Args:
            x: Input embeddings [Batch, Seq_Len, Dim]
            text_context: Optional text string (for legacy logging/routing fallback)
            content_context: Optional dict with metadata (e.g. 'source', 'timestamp')
            
        Returns:
            output: Processed tensor
            info: Dictionary containing routing info, activity stats, etc.
        """
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Run optimized forward pass
        output, internal_info = self.forward(x, context=content_context)
        
        # Augment info with context for API compatibility
        info = {
            'mode': 'neuromorphic_gpu',
            'routing': internal_info['routing'],
            'zone_activities': internal_info['activity'],
            'text_context': text_context,
            'content_context': content_context or {}
        }
        
        return output, info

    def forward(self, x: torch.Tensor, context: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through the brain (GPU Optimized).
        Args:
            x: Input embeddings [Batch, Seq_Len, Dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Global Routing (Per-token or Per-Sequence?)
        # For efficiency, we pool the sequence to route
        # [Batch, Seq, Dim] -> [Batch, Dim]
        pooled_x = x.mean(dim=1)
        
        routing_out = self.global_router(pooled_x)
        topk_indices = routing_out['indices'] # [Batch, k]
        topk_weights = routing_out['weights'] # [Batch, k]
        
        # 2. Dispatch to Zones
        zone_outputs = torch.zeros_like(x)
        total_activity = {}
        
        # Iterate unique active zones in this batch to save compute
        active_indices = torch.unique(topk_indices)
        
        for idx in active_indices:
            zone_idx = idx.item()
            if zone_idx >= len(self.zone_names): continue
            
            zone_name = self.zone_names[zone_idx]
            zone_module = self.zones[zone_name]
            
            # Forward pass through zone
            # [Batch, Seq, Dim]
            z_out, z_stats = zone_module(x)
            
            # Masking: Apply contribution only where this zone was selected
            # Mask: [Batch, 1, 1]
            mask = (topk_indices == idx).any(dim=1).float().view(batch_size, 1, 1)
            
            # Weighted combination
            # Sum weights where selected (usually selected once)
            # w: [Batch, 1]
            is_selected = (topk_indices == idx) # [Batch, k]
            w = (topk_weights * is_selected.float()).sum(dim=1).view(batch_size, 1, 1)
            
            zone_outputs += z_out * w
            total_activity[zone_name] = z_stats.get('avg_firing_rate', 0.0)
            
        # 3. Residual / Skip Connection
        output = x + zone_outputs
        
        # Detach routing info for CPU return to avoid blocking
        return output, {
            "routing": routing_out['probs'].detach().cpu().numpy(),
            "activity": total_activity
        }

# Factory for convenience
def create_aura_brain(d_model=1024, device='cuda') -> EnhancedBrain:
    zones = {
        'prefrontal': BrainZoneConfig(max_neurons=512, zone_type=BrainZoneType.PREFRONTAL_CORTEX),
        'hippocampus': BrainZoneConfig(max_neurons=512, zone_type=BrainZoneType.HIPPOCAMPUS),
        'temporal': BrainZoneConfig(max_neurons=512, zone_type=BrainZoneType.TEMPORAL_CORTEX),
        'amygdala': BrainZoneConfig(max_neurons=256, zone_type=BrainZoneType.AMYGDALA),
    }
    return EnhancedBrain(d_model=d_model, zones_config=zones, device=device)

# Backwards-compatible alias for tests expecting `Brain`
from cli.config import BrainConfig, Config

class Brain(EnhancedBrain):
    """Backwards compatible Brain class.

    Accepts a ``BrainConfig`` (or ``Config``) instance as used in older code and
    forwards the relevant parts to ``EnhancedBrain``.
    """

    def __init__(self, config: Config):
        # ``config`` may be a ``BrainConfig`` alias; both have the same fields.
        # Extract layer configuration and brain zone configurations.
        layers_cfg = getattr(config, "layers_config", None)
        zones_cfg = getattr(config, "brain_zones_config", [])
        # Convert list of BrainZoneConfig objects to a dict keyed by name if needed.
        zones_dict = {}
        for zone in zones_cfg:
            # Assume each zone has a ``name`` attribute; fallback to its class name.
            zone_name = getattr(zone, "name", zone.__class__.__name__)
            zones_dict[zone_name] = zone
        # Initialise the parent class with defaults for other parameters.
        super().__init__(
            config=layers_cfg,
            zones=zones_dict,
            d_model=1024,
            use_neuromorphic=True,
            processing_mode=ProcessingMode.NEUROMORPHIC,
        )

# --- Liquid MoE Integration Components ---

class ConsciousnessLevel(Enum):
    DEEP_SLEEP = 0; ASLEEP = 1; ALERT = 2; FOCUSED = 3; HYPERVIGILANT = 4

class CentralNervousSystem:
    def __init__(self):
        self.consciousness_level = ConsciousnessLevel.ALERT
        self.stress_level = 0.0 # 'cortisol'
        self.consolidation_factor = 1.0 # Reduces impact of errors after dreaming
        print("CREATED: CentralNervousSystem (CNS)")
    def set_consciousness(self, level: ConsciousnessLevel):
        if self.consciousness_level != level:
            self.consciousness_level = level; print(f"CNS: Consciousness set to {level.name}")
    def update_stress(self, error: float):
        new_stress = abs(error) * 1.5 * self.consolidation_factor
        self.stress_level = (self.stress_level * 0.5) + (new_stress * 0.5)
        self.stress_level = max(0.0, self.stress_level - 0.1)
        if self.stress_level > 1.0: self.set_consciousness(ConsciousnessLevel.HYPERVIGILANT)
        else: self.set_consciousness(ConsciousnessLevel.ALERT)
    def apply_consolidation(self, factor: float = 0.7):
        """Apply memory consolidation effect that reduces impact of future errors"""
        self.consolidation_factor = factor

    def get_endocrine_levels(self) -> Dict[str, float]:
        """Compute endocrine hormone levels from CNS state"""
        cortisol = min(2.0, self.stress_level * 2.0)
        gh = 0.5 if self.consciousness_level == ConsciousnessLevel.ALERT else 0.0
        thyroid = 1.0 - (self.stress_level * 0.3)
        dopamine = max(0.0, 1.0 - self.stress_level) if self.consciousness_level == ConsciousnessLevel.ALERT else 0.0
        return {
            'cortisol': cortisol, 'gh': gh, 'thyroid': thyroid, 'dopamine': dopamine
        }

class TemporalMemoryInterpolator:
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon
        print("CREATED: TemporalMemoryInterpolator")
    def interpolate(self, M0: np.ndarray, M1: np.ndarray, t: float,
                    mode: Literal['linear', 'fourier', 'hilbert', 'hamiltonian'] = 'hilbert'
                   ) -> np.ndarray:
        alpha = np.clip(t, 0.0, 1.0)
        if mode == 'linear': return (1.0 - alpha) * M0 + alpha * M1
        elif mode == 'fourier':
            F0 = np.fft.fft(M0); F1 = np.fft.fft(M1)
            F_interp = (1.0 - alpha) * F0 + alpha * F1
            return np.real(np.fft.ifft(F_interp))
        A0 = hilbert(M0, axis=0); A1 = hilbert(M1, axis=0)
        if mode == 'hilbert':
            A_interp = (1.0 - alpha) * A0 + alpha * A1
            return np.real(A_interp)
        else: raise ValueError(f"Unknown interpolation mode: {mode}")

class LiquidBrain:
    """
    Aura 7.0: Liquid Brain Architecture integrating STDP, Hebbian Learning, and Liquid MoE.
    Parallel implementation to EnhancedBrain for advanced learning capabilities.
    """
    def __init__(self, n_experts: int = 15, hebbian_components: int = 64, d_model: int = 1024):
        print("--- 🧠 Initializing Liquid Brain (Hebbian-MoE-Temporal) ---")
        
        if torch.cuda.is_available():
            maybe_empty_cuda_cache("startup", min_free_ratio=0.05)
            
        self.d_model = d_model
        self.n_experts = n_experts
        self.hebbian_components = hebbian_components
        
        # Components
        self.cns = CentralNervousSystem()
        self.interpolator = TemporalMemoryInterpolator()
        
        # Input processing (replaces FeatureGenerator)
        self.embedder = FastHashEmbedder(dim=d_model)
        self.whitener = OptimizedWhitener(dim=d_model)
        
        # Hebbian Cortex (OjaLayer)
        self.hippocampus = OjaLayer(
            input_dim=d_model, # Note: OjaLayer expects input_dim
            n_components=hebbian_components,
            # mode='nonlinear', # OjaLayer in codebase uses standard Oja's rule, check implementation
            # lr=5e-4, max_components=128 # Check constructor args
        )
        
        # Liquid MoE Cortex
        self.cortex: Optional[LiquidMoERouter] = None
        self._create_moe_cortex(hebbian_components)
        
        # Stats
        self.last_run_final_stress = 0.0
        
        print("--- Liquid Brain Initialized ---")

    def _create_moe_cortex(self, input_dim: int):
        """Dynamically creates the MoE cortex"""
        experts: Dict[str, NLMSExpertAdapter] = {}
        attention_config = {
            'decay': 0.7, 'theta': 1.0, 'k_winners': 3, 'gain_up': 1.5, 'gain_down': 0.7,
            'multi_channel': {'k_winners': 5, 'w_amp': 1.0, 'w_pitch': 1.4, 'w_bound': 0.8}
        }
        
        # Mock emotion range for initialization
        emotion_range_per_expert = 28.0 / self.n_experts 
        
        for i in range(self.n_experts):
            name = f"expert__{i}"
            target_emotion_idx = i * emotion_range_per_expert + emotion_range_per_expert / 2
            head = ExpertNLMSHead(
                n_features=input_dim,
                vocab_size=10000, # Placeholder vocab size
                attention_config=attention_config,
                mu=0.5, mu_decay=0.99995, mu_min=0.1,
                initial_bias=target_emotion_idx
            )
            # Random initialization
            head.w = np.random.normal(0, 1.0, input_dim)
            head.bias = float(target_emotion_idx)
            experts[name] = NLMSExpertAdapter(neuron=head)
            
        self.cortex = LiquidMoERouter(
            experts=experts, in_dim=input_dim,
            hidden_dim=128, top_k=3
        )

    async def process_query(self, query: str, target_signal: float = 0.0, update_stress: bool = True) -> Dict[str, Any]:
        """
        Process a text query through the Liquid Brain pipeline.
        
        Args:
            query: Input text
            target_signal: Optional target signal (e.g., emotion index) for learning.
                           If 0.0 (default), it implies unsupervised or inference mode.
            update_stress: Whether to update CNS stress level based on prediction error.
        """
        # 1. Embed and Whiten
        x_emb, token_indices = self.embedder.encode_with_indices(query)
        x_np = x_emb.numpy()
        xw = self.whitener.transform(x_np)
        
        # 2. Hebbian Cortex (Hippocampus)
        oja_out = self.hippocampus.step(xw)
        y_abstract = oja_out.y
        
        # Check for neurogenesis
        if oja_out.grew:
            # If hippocampus grew, we might need to adapt MoE input dim
            # For now, we'll just log it, real implementation would require dynamic resizing of MoE weights
            print(f"Neurogenesis: Hippocampus grew to {self.hippocampus.K} components")
            # In a full implementation, we would rebuild/expand the MoE cortex here
        
        # 3. MoE Cortex (Liquid Router)
        # Learn mode if we have a meaningful target, otherwise just route (inference)
        # For this implementation, we assume target_signal is provided or 0
        
        # Note: LiquidMoERouter.learn does routing + update
        moe_out = await self.cortex.learn(
            x=y_abstract, 
            token_ids=token_indices, 
            y_true=target_signal,
            attention_bundle={} # Placeholder
        )
        
        raw_prediction = moe_out['y_hat']
        
        # 4. CNS Update
        if update_stress:
            error = abs(target_signal - raw_prediction)
            self.cns.update_stress(error)
            endocrine = self.cns.get_endocrine_levels()
            await self.cortex.gating.apply_endocrine(**endocrine)
            self.last_run_final_stress = self.cns.stress_level
            
        return {
            "prediction": raw_prediction,
            "moe_output": moe_out,
            "hebbian_residual": oja_out.residual_ema,
            "stress_level": self.cns.stress_level
        }

# Factory for convenience
def create_aura_brain(d_model=1024, device='cuda') -> EnhancedBrain:
    zones = {
        'prefrontal': BrainZoneConfig(max_neurons=512, zone_type=BrainZoneType.PREFRONTAL_CORTEX),
        'hippocampus': BrainZoneConfig(max_neurons=512, zone_type=BrainZoneType.HIPPOCAMPUS),
        'temporal': BrainZoneConfig(max_neurons=512, zone_type=BrainZoneType.TEMPORAL_CORTEX),
        'amygdala': BrainZoneConfig(max_neurons=256, zone_type=BrainZoneType.AMYGDALA),
    }
    return EnhancedBrain(d_model=d_model, zones_config=zones, device=device)

def fix_neuromorphic_crisis(brain):
    """Emergency fix for 511+ firing rates"""
    import torch
    
    if hasattr(brain, 'enhanced_zones'):
        for zone_name, zone in brain.enhanced_zones.items():
            if hasattr(zone, 'neuron_groups'):
                for neuron_type, neuron_group in zone.neuron_groups.items():
                    # Fix biological → normalized scale
                    if hasattr(neuron_group, 'threshold'):
                        current = float(neuron_group.threshold.data.item())
                        if current < 0:  # Biological scale detected
                            neuron_group.threshold.data.fill_(0.6)
                            print(f"✅ Fixed {zone_name}.{neuron_type} threshold: {current:.1f} → 0.6")
                    
                    # Fix impossible target  
                    if hasattr(neuron_group, 'target_firing_rate'):
                        neuron_group.target_firing_rate = 2.0
                        print(f"✅ Fixed {zone_name}.{neuron_type} target: 0.1 → 2.0")

                    if hasattr(neuron_group, 'homeostasis_lr'):
                        neuron_group.homeostasis_lr = 1e-4  # Gentler than 1e-3
    
    print("🎊 Crisis resolved - firing rates should normalize!")
    return brain

# Example usage
if __name__ == "__main__":
    print("Creating Brain for Aura AI...")
    
    # Create brain
    
    brain = create_aura_brain(d_model=1024)
    # Attempt to load saved homeostasis state per zone (if available)
    try:
        import os
        if hasattr(brain, 'enhanced_zones'):
            for zname, z in brain.enhanced_zones.items():
                neu = getattr(z, 'neuromorphic_processor', None)
                if neu and hasattr(neu, 'load_homeostasis_state'):
                    neu.load_homeostasis_state(os.path.join('brain_states', f'{zname}_homeostasis.json'))
    except Exception:
        pass
    
    # ✅ ADD THIS LINE - THIS IS THE KEY FIX!
    # brain = fix_neuromorphic_crisis(brain)
    # Ensure deterministic inference behavior (disable dropout etc.)
    try:
        if hasattr(brain, 'enhanced_zones'):
            for _z in brain.enhanced_zones.values():
                print(f"Evaluating {_z.config.name}")
                if hasattr(_z, 'eval'):
                    _z.eval()
    except Exception:
        pass
    
   
    
    
    # Test processing (add positive bias to ensure spikes under normalized thresholds)
    sample_input = (torch.randn(2, 10, 512) * 1.0).clamp(-3, 3) + 0.6  # [batch, seq_len, d_model]
    sample_text = "Analyze the scientific discovery of Einstein's relativity theory"
    
    # Longer adaptation loop to let per-group homeostasis converge
    for _ in range(20):
        _ = brain.process_input(
            sample_input,
            sample_text,
            content_context={'dc_bias': 0.05, 'boost': 1.0, 'noise_std': 0.02, 'force_spike': True, 'debug_spike': False, 'rectify': True}
        )
    output, processing_info = brain.process_input(
        sample_input,
        sample_text,
        content_context={'dc_bias': 0.05, 'boost': 1.0, 'noise_std': 0.02, 'force_spike': True, 'debug_spike': True, 'rectify': True}
    )
    
    print(f"\nProcessing Test:")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Processing mode: {processing_info['mode']}")
    # Print per-zone average firing rates after run
    if 'zone_activities' in processing_info:
        print("\nPer-zone activity (avg_firing_rate):")
        for zone_name, activity in processing_info['zone_activities'].items():
            afr = float(activity.get('avg_firing_rate', 0.0)) if isinstance(activity, dict) else 0.0
            print(f"  - {zone_name}: {afr:.3f}")
    
    
    # Print summary
    # Save per-zone homeostasis state for continuous runs
    try:
        import os
        os.makedirs('brain_states', exist_ok=True)
        if hasattr(brain, 'enhanced_zones'):
            for zname, z in brain.enhanced_zones.items():
                neu = getattr(z, 'neuromorphic_processor', None)
                if neu and hasattr(neu, 'save_homeostasis_state'):
                    neu.save_homeostasis_state(os.path.join('brain_states', f'{zname}_homeostasis.json'))
    except Exception:
        pass
    
    if 'content_routing' in processing_info:
        print("Content analysis:", processing_info['content_routing'])
    
    # Check stability
    stability_report = brain.check_training_stability()
    print(f"Stability check: {stability_report.get('overall', 'unknown')}")
    
    print("Enhanced brain integration complete!")

    # Fallback stimulation if no activity observed
    try:
        total_avg = 0.0
        if hasattr(brain, 'enhanced_zones'):
            for zname, z in brain.enhanced_zones.items():
                st = z.get_activity_stats()
                if isinstance(st, dict):
                    total_avg += float(st.get('avg_firing_rate', 0.0))
        if total_avg == 0.0:
            print("\nNo activity detected, running fallback positive-biased stimulation...")
            fb_input = torch.ones(2, 10, 512) * 0.6
            fb_text = "please calculate this quickly"
            fb_out, fb_info = brain.process_input(
                fb_input,
                fb_text,
                content_context={'dc_bias': 0.05, 'boost': 1.0, 'noise_std': 0.02, 'force_spike': True, 'debug_spike': True, 'rectify': True}
            )
            print(f"Fallback processing mode: {fb_info.get('mode')}")
            if 'zone_activities' in fb_info:
                print("Per-zone activity after fallback:")
                for zone_name, activity in fb_info['zone_activities'].items():
                    afr = float(activity.get('avg_firing_rate', 0.0)) if isinstance(activity, dict) else 0.0
                    print(f"  - {zone_name}: {afr:.3f}")
            # If still no activity, directly stimulate each zone more strongly
            try:
                if hasattr(brain, 'enhanced_zones'):
                    direct_total = 0.0
                    for zname, z in brain.enhanced_zones.items():
                        with torch.no_grad():
                            _out, _act = z(
                                torch.ones(1, 10, 512) * 0.8,
                                context={'source': 'direct_fallback', 'dc_bias': 0.8, 'boost': 4.0, 'noise_std': 0.07, 'force_spike': True}
                            )
                        afr = float(_act.get('avg_firing_rate', 0.0)) if isinstance(_act, dict) else 0.0
                        direct_total += afr
                        print(f"  • Direct {zname}: {afr:.3f}")
                    if direct_total == 0.0:
                        print("  • Still no spikes; try increasing thresholds downward (e.g., 0.5) or input to 1.0")
            except Exception:
                pass
    except Exception:
        pass
    
    brain.print_brain_summary()
    
    # --- Liquid Brain Test ---
    print("\n" + "="*60)
    print("LIQUID BRAIN TEST")
    print("="*60)
    
    # Initialize Liquid Brain
    liquid_brain = LiquidBrain(n_experts=5, hebbian_components=32, d_model=512)
    
    # Test Query
    test_query = "I am feeling very curious about how this works."
    print(f"Processing query: '{test_query}'")
    
    async def run_liquid_test():
        # Simulate target signal (e.g., emotion 'curiosity' mapped to index 2.5)
        result = await liquid_brain.process_query(test_query, target_signal=2.5)
        print("\nLiquid Brain Result:")
        print(f"  Prediction: {result['prediction']:.4f}")
        print(f"  Hebbian Residual: {result['hebbian_residual']:.4f}")
        print(f"  Stress Level: {result['stress_level']:.4f}")
        
        topk = result['moe_output'].get('topk', [])
        print("  Top Experts:")
        for name, gate in topk:
            print(f"    - {name}: {gate:.4f}")

    # Run async test
    try:
        asyncio.run(run_liquid_test())
    except Exception as e:
        print(f"Liquid Brain Test Failed: {e}")
        import traceback
        traceback.print_exc()
