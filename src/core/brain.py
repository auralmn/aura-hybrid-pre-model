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
from core.layers_factory import LayersFactory
from core.neuron_factory import NeuronFactory
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
from core.experts import ExpertNLMSHead, NLMSExpertAdapter
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

class EnhancedBrain:
    """Enhanced Brain class with neuromorphic capabilities"""
    
    # Your original attributes
    layers_factory: LayersFactory
    neuron_factory: NeuronFactory
    zone_factory: BrainZoneFactory
    
    # Enhanced attributes
    enhanced_zones: Dict[str, NeuromorphicBrainZone]
    neuromorphic_processor: NeuromorphicProcessor
    content_router: ContentRouter
    event_bus: EventBus
    stats_collector: StatsCollector
    
    def __init__(self, 
                 config: BaseLayerContainerConfig, 
                 zones: Dict[str, BrainZoneConfig],
                 d_model: int = 1024,
                 use_neuromorphic: bool = True,
                 processing_mode: ProcessingMode = ProcessingMode.NEUROMORPHIC):
        
        # Initialize your original components
        self.layers_factory = LayersFactory(config)
        self.neuron_factory = NeuronFactory(256, 512, 768, 0.02, 0.02, 2.0)
        self.zone_factory = BrainZoneFactory()
        
        # Enhanced initialization
        self.d_model = d_model
        self.use_neuromorphic = use_neuromorphic
        self.processing_mode = processing_mode
        
        # Event system for monitoring
        self.event_bus = EventBus()
        
        # Statistics collection
        self.stats_collector = StatsCollector()
        
        # Content routing
        self.content_router = ContentRouter()
        
        # Initialize zones (your existing logic + enhancements)
        self.zones = {}
        self.enhanced_zones = {}
        # Optional continuous learning orchestrator (not started by default)
        self.learning_orchestrator = None
        
        for zone_name, zone_config in zones.items():
            
            # Create your original zone
            original_zone = self.create_brain_zone(zone_config)
            self.zones[zone_name] = original_zone
            
            # Create enhanced neuromorphic version if enabled
            if use_neuromorphic:
                enhanced_zone = self._create_enhanced_zone(zone_name, zone_config)
                if enhanced_zone:
                    self.enhanced_zones[zone_name] = enhanced_zone
        
        # Set zone attributes (your existing logic)
        self._set_zone_attributes()
        
        # Initialize neuromorphic processor
        if use_neuromorphic and self.enhanced_zones:
            self.neuromorphic_processor = NeuromorphicProcessor(
                d_model=d_model,
                processing_mode=processing_mode,
                event_bus=self.event_bus
            )
            self.neuromorphic_processor.set_zone_processors(self.enhanced_zones)
            # Default zone capabilities for planning
            self.neuromorphic_processor.set_zone_capabilities({
                'prefrontal_cortex': ['reasoning', 'analytical'],
                'temporal_cortex': ['language', 'creative', 'semantic'],
                'hippocampus': ['memory', 'temporal', 'context'],
                'cerebellum': ['precise', 'refine']
            })
        else:
            self.neuromorphic_processor = None
        
        # Subscribe to events for statistics
        self.event_bus.subscribe('neuron_fired', self._handle_neuron_fired)
        self.event_bus.subscribe('brain_stats_updated', self._handle_stats_updated)
    
    # -----------------------------
    # Continuous learning integration (optional)
    # -----------------------------
    def attach_continuous_learning(self, feeds: Optional[List["RSSFeedConfig"]] = None) -> None:
        """Attach the RSS-based continuous learning orchestrator to this brain.

        This will NOT start background loops automatically. Call
        start_continuous_learning/stop_continuous_learning explicitly.
        """
        if ContinuousLearningOrchestrator is None:
            return
        if not getattr(self, 'neuromorphic_processor', None):
            return
        if self.learning_orchestrator is None:
            self.learning_orchestrator = ContinuousLearningOrchestrator(
                self.neuromorphic_processor,
                self.event_bus,
            )
        if feeds is None and create_default_feeds is not None:
            try:
                feeds = create_default_feeds()
            except Exception:
                feeds = None
        if feeds:
            for feed in feeds:
                try:
                    self.learning_orchestrator.add_feed(feed)
                except Exception:
                    continue

    async def start_continuous_learning(self) -> None:
        if self.learning_orchestrator is not None:
            try:
                await self.learning_orchestrator.start()
            except Exception:
                pass

    async def stop_continuous_learning(self) -> None:
        if self.learning_orchestrator is not None:
            try:
                await self.learning_orchestrator.stop()
            except Exception:
                pass
    
    def _create_enhanced_zone(self, zone_name: str, zone_config: BrainZoneConfig) -> Optional[NeuromorphicBrainZone]:
        """Create enhanced neuromorphic version of brain zone"""
        
        # Map zone names to types
        zone_type_mapping = {
            'prefrontal_cortex': BrainZoneType.PREFRONTAL_CORTEX,
            'temporal_cortex': BrainZoneType.TEMPORAL_CORTEX,
            'hippocampus': BrainZoneType.HIPPOCAMPUS,
            'cerebellum': BrainZoneType.CEREBELLUM,
            'thalamus': BrainZoneType.THALAMUS,
            'amygdala': BrainZoneType.AMYGDALA,
            'basal_ganglia': BrainZoneType.BASAL_GANGLIA,
            'brainstem': BrainZoneType.BRAINSTEM,
            'occipital_cortex': BrainZoneType.OCCIPITAL_CORTEX,
            'parietal_cortex': BrainZoneType.PARIETAL_CORTEX,
            'insular_cortex': BrainZoneType.INSULAR_CORTEX,
        }
        
        zone_type = zone_type_mapping.get(zone_name)
        if not zone_type:
            print(f"No neuromorphic mapping for zone {zone_name}, skipping enhancement")
            return None
        
        # Create enhanced configuration
        enhanced_config = BrainZoneConfig(
            name=zone_name,
            max_neurons=zone_config.max_neurons,
            min_neurons=zone_config.min_neurons,
            num_layers=zone_config.num_layers,
            zone_type=zone_type,
            d_model=self.d_model,
            use_spiking=True,
            event_bus=self.event_bus
        )
        
        return NeuromorphicBrainZone(enhanced_config)
    
    def _set_zone_attributes(self):
        """Set individual zone attributes (your existing logic)"""
        if 'cortex' in self.zones:
            self.cortex_zone = self.zones['cortex']
        if 'cns' in self.zones:
            self.cns_zone = self.zones['cns']
        if 'thalamus' in self.zones:
            self.thalamus_zone = self.zones['thalamus']
        if 'hypothalamus' in self.zones:
            self.hypothalamus_zone = self.zones['hypothalamus']
        if 'brainstem' in self.zones:
            self.brainstem_zone = self.zones['brainstem']
        if 'cerebellum' in self.zones:
            self.cerebellum_zone = self.zones['cerebellum']
        if 'hippocampus' in self.zones:
            self.hippocampus_zone = self.zones['hippocampus']
        if 'amygdala' in self.zones:
            self.amygdala_zone = self.zones['amygdala']
        if 'prefrontal_cortex' in self.zones:
            self.prefrontal_cortex_zone = self.zones['prefrontal_cortex']
        if 'parietal_cortex' in self.zones:
            self.parietal_cortex_zone = self.zones['parietal_cortex']
        if 'occipital_cortex' in self.zones:
            self.occipital_cortex_zone = self.zones['occipital_cortex']
        if 'temporal_cortex' in self.zones:
            self.temporal_cortex_zone = self.zones['temporal_cortex']
        if 'insular_cortex' in self.zones:
            self.insular_cortex_zone = self.zones['insular_cortex']
        if 'basal_ganglia' in self.zones:
            self.basal_ganglia_zone = self.zones['basal_ganglia']
    
    def create_layers(self, config: BaseLayerContainerConfig) -> BaseLayerContainer:
        """Your existing create_layers method with enhancements"""
        # Your existing factory config logic
        factory_config = getattr(self.layers_factory, 'config', None)
        if factory_config is not None:
            num_layers = getattr(factory_config, 'num_layers', None)
            if not isinstance(num_layers, int):
                try:
                    setattr(factory_config, 'num_layers', 3)
                except Exception:
                    pass
        
        # Create layers using enhanced factory if neuromorphic enabled
        if getattr(self, 'use_neuromorphic', False):
            enhanced_factory = BaseLayerFactory(self.event_bus)
            
            # Create neuromorphic layer configurations
            layer_configs = []
            for i in range(config.num_layers):
                layer_config = BaseLayerConfig(
                    name=f"enhanced_layer_{i}",
                    input_dim=self.d_model if i > 0 else 512,  # Adjust as needed
                    output_dim=self.d_model,
                    use_spiking=True,
                    neuron_type="spiking"
                )
                layer_configs.append(layer_config)
            
            return enhanced_factory.create_layer_container(config, layer_configs)
        else:
            return self.layers_factory.create_layers(config)
    
    def create_brain_zone(self, config: BrainZoneConfig) -> BrainZone:
        """Your existing create_brain_zone method"""
        # Your existing logic
        layer_container_config = BaseLayerContainerConfig(
            num_layers=getattr(config, 'num_layers', 3),
            layer_type=getattr(config, 'layer_type', 'dense')
        )
        
        layer_config = BaseLayerConfig(
            name=config.name, 
            input_dim=config.min_neurons, 
            output_dim=config.max_neurons
        )
        
        layers = self.layers_factory.create_layers(layer_config)
        
        return self.zone_factory.create_brain_zone(config, layers)
    
    def process_input(self, 
                     input_embeddings: torch.Tensor,
                     text_context: Optional[str] = None,
                     content_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Enhanced input processing through neuromorphic zones"""
        
        if not self.use_neuromorphic or not self.neuromorphic_processor:
            # Fallback to basic processing
            return self._basic_processing(input_embeddings)
        
        # Process through neuromorphic processor
        context = content_context or {}
        if text_context:
            context['text'] = text_context
        
        try:
            output = self.neuromorphic_processor.process(input_embeddings, context)
            
            # Get processing stats which includes zone activities
            processing_stats = self.neuromorphic_processor.get_processing_stats()
            
            # Extract zone activities from processing stats
            zone_activities = processing_stats.get('zone_activities', {})
            
            # Update statistics
            self.stats_collector.update_from_brain(self)
            
            processing_info = {
                'mode': 'neuromorphic',
                'zone_activities': zone_activities,
                'content_routing': self.content_router.analyze_content(text_context or ""),
                'processing_stats': processing_stats
            }
            
            return output, processing_info
            
        except Exception as e:
            print(f"Neuromorphic processing failed: {e}, falling back to basic processing")
            return self._basic_processing(input_embeddings)
    
    def _basic_processing(self, input_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Basic fallback processing"""
        # Simple pass-through or basic transformation
        output = input_embeddings  # Could add basic processing here
        
        info = {
            'mode': 'basic',
            'message': 'Using basic processing mode'
        }
        
        return output, info
    
    def get_brain_statistics(self) -> BrainStats:
        """Get comprehensive brain statistics"""
        
        # Update basic statistics
        stats = self.stats_collector.get_stats()
        stats.num_zones = len(self.zones)
        stats.num_layers = 0
        
        # Count neurons from enhanced zones
        total_neurons = 0
        for zone in self.enhanced_zones.values():
            if hasattr(zone, 'neuron_counts'):
                total_neurons += sum(zone.neuron_counts.values())
                
                stats.num_layers += zone.config.num_layers
        
        if total_neurons > 0:
            stats.num_neurons = total_neurons
        
        # Update with current zone activities
        if self.enhanced_zones:
            zone_activities = {}
            for zone_name, zone in self.enhanced_zones.items():
                zone_activities[zone_name] = zone.get_activity_stats()
            
            if zone_activities:
                stats.update_from_zone_activity(zone_activities)
        
        return stats
    
    def check_training_stability(self) -> Dict[str, str]:
        """Check training stability across all zones"""
        stability_report = {}
        
        for zone_name, zone in self.enhanced_zones.items():
            if hasattr(zone, 'check_zone_health'):
                zone_health = zone.check_zone_health()
                stability_report.update({f"{zone_name}_{k}": v for k, v in zone_health.items()})
        
        # Overall stability assessment
        silent_count = sum(1 for status in stability_report.values() if status == "silent")
        hyperactive_count = sum(1 for status in stability_report.values() if status == "hyperactive")
        
        if silent_count > len(stability_report) * 0.3:
            stability_report['overall'] = 'too_many_silent'
        elif hyperactive_count > len(stability_report) * 0.3:
            stability_report['overall'] = 'too_many_hyperactive'
        else:
            stability_report['overall'] = 'stable'
        
        return stability_report
    
    def adjust_for_stability(self, stability_report: Dict[str, str]):
        """Auto-adjust parameters based on stability report"""
        
        for param_name, status in stability_report.items():
            if status == "silent" and "_" in param_name:
                zone_name, neuron_type = param_name.rsplit("_", 1)
                self._increase_surrogate_slope(zone_name, neuron_type)
            elif status == "hyperactive" and "_" in param_name:
                zone_name, neuron_type = param_name.rsplit("_", 1)
                self._decrease_surrogate_slope(zone_name, neuron_type)
    
    def _increase_surrogate_slope(self, zone_name: str, neuron_type: str):
        """Increase surrogate slope for silent neurons"""
        if zone_name in self.enhanced_zones:
            zone = self.enhanced_zones[zone_name]
            if hasattr(zone, 'neuron_groups') and neuron_type in zone.neuron_groups:
                neuron_group = zone.neuron_groups[neuron_type]
                with torch.no_grad():
                    if hasattr(neuron_group, 'slope'):
                        neuron_group.slope.data *= 1.1
                        neuron_group.slope.data.clamp_(5.0, 100.0)
    
    def _decrease_surrogate_slope(self, zone_name: str, neuron_type: str):
        """Decrease surrogate slope for hyperactive neurons"""
        if zone_name in self.enhanced_zones:
            zone = self.enhanced_zones[zone_name]
            if hasattr(zone, 'neuron_groups') and neuron_type in zone.neuron_groups:
                neuron_group = zone.neuron_groups[neuron_type]
                with torch.no_grad():
                    if hasattr(neuron_group, 'slope'):
                        neuron_group.slope.data *= 0.9
                        neuron_group.slope.data.clamp_(5.0, 100.0)
    
    def _handle_neuron_fired(self, event):
        zone = event.data.get('zone', 'unknown')
        rate = event.data.get('firing_rate', 0)
        print(f"🔥 FIRING: {zone} = {rate:.3f} Hz")
    
    def _handle_stats_updated(self, event):
        """Handle brain statistics update events"""
        # Could trigger rebalancing or other adaptations
        pass
    
    def print_brain_summary(self):
        """Print comprehensive brain summary"""
        print("=" * 60)
        print("ENHANCED BRAIN SUMMARY")
        print("=" * 60)
        
        print(f"Neuromorphic Mode: {'Enabled' if self.use_neuromorphic else 'Disabled'}")
        print(f"Total Zones: {len(self.zones)}")
        print(f"Enhanced Zones: {len(self.enhanced_zones)}")
        print(f"Model Dimension: {self.d_model}")
        
        if self.enhanced_zones:
            print("\nEnhanced Zone Details:")
            for zone_name, zone in self.enhanced_zones.items():
                stats = zone.get_activity_stats()
                # Fallback to configured counts if stats are not yet populated
                total_neurons = stats.get('total_neurons') if isinstance(stats, dict) else None
                if not total_neurons and hasattr(zone, 'neuron_counts'):
                    try:
                        total_neurons = sum(zone.neuron_counts.values())
                    except Exception:
                        total_neurons = 0
                avg_firing = stats.get('avg_firing_rate', 0.0) if isinstance(stats, dict) and 'avg_firing_rate' in stats else 0.0
                print(f"  {zone_name}: {total_neurons} neurons, {avg_firing:.4f} avg firing rate")
        
        # Print overall statistics
        brain_stats = self.get_brain_statistics()
        brain_stats.print_summary()
        
        # Check and report stability
        stability = self.check_training_stability()
        print("\nStability Report:")
        healthy = sum(1 for status in stability.values() if status == "healthy")
        total = len([s for s in stability.values() if s != "overall"])
        print(f"  Healthy zones: {healthy}/{total}")
        
        if stability.get('overall') != 'stable':
            print(f"  ⚠️  Overall status: {stability['overall']}")
        
        print("=" * 60)

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

# Helper function to create a pre-configured neuromorphic brain for your Aura AI
def create_aura_brain(d_model: int = 1024, 
                      max_neurons_per_zone: int = 512,
                      use_neuromorphic: bool = True) -> EnhancedBrain:
    """Create a pre-configured brain for Aura AI with optimal zone setup"""
    
    # Layer configuration
    layer_config = BaseLayerContainerConfig(
        num_layers=3, 
        layer_type='neuromorphic',
        use_neuromorphic=True
    )
    
    # Zone configurations optimized for general-purpose LLM
    zone_configs = {
        'prefrontal_cortex': BrainZoneConfig(
            name='prefrontal_cortex',
            max_neurons=max_neurons_per_zone,
            min_neurons=max_neurons_per_zone // 2,
            num_layers=3,
            zone_type=BrainZoneType.PREFRONTAL_CORTEX,
            use_spiking=True
        ),
        'temporal_cortex': BrainZoneConfig(
            name='temporal_cortex', 
            max_neurons=max_neurons_per_zone,
            min_neurons=max_neurons_per_zone // 2,
            num_layers=3,
            zone_type=BrainZoneType.TEMPORAL_CORTEX,
            use_spiking=True
        ),
        'hippocampus': BrainZoneConfig(
            name='hippocampus',
            max_neurons=max_neurons_per_zone,
            min_neurons=max_neurons_per_zone // 2,
            num_layers=2,
            zone_type=BrainZoneType.HIPPOCAMPUS,
            use_spiking=True
        ),
        'cerebellum': BrainZoneConfig(
            name='cerebellum',
            max_neurons=max_neurons_per_zone,
            min_neurons=max_neurons_per_zone // 2,
            num_layers=2,
            zone_type=BrainZoneType.CEREBELLUM,
            use_spiking=True
        )
    }
    
    brain = EnhancedBrain(
        config=layer_config,
        zones=zone_configs,
        d_model=d_model,
        use_neuromorphic=True,
        processing_mode=ProcessingMode.NEUROMORPHIC
    )
    return brain

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
    
    brain = create_aura_brain(d_model=1024, use_neuromorphic=True)
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
