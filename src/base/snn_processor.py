#!/usr/bin/env python3
"""
Enhanced processors.py with neuromorphic content routing
Extends your BaseProcessor with brain zone-specific processing
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import torch
import torch.nn as nn

from base.events import EventBus
from maths.softmax import softmax as np_softmax
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum as _EnumAlias
from datetime import datetime

class ProcessingMode(Enum):
    """Different processing modes for neuromorphic processors"""
    BASIC = "basic"
    NEUROMORPHIC = "neuromorphic"
    HYBRID = "hybrid"

class ContentType(Enum):
    """Types of content for specialized processing"""
    REASONING = "reasoning"
    MEMORY = "memory"
    LANGUAGE = "language"
    EMOTION = "emotion"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    PATTERN = "pattern"
    TEMPORAL = "temporal"

class BaseProcessor(ABC):
    """Base processor interface - maintains your original interface"""
    
    @abstractmethod
    def process(self, input: Any) -> Any:
        pass

class ContentRouter:
    """Routes content to appropriate brain zones based on analysis"""
    
    def __init__(self):
        # Content type to brain zone mapping
        self.content_to_zones = {
            ContentType.REASONING: ['prefrontal_cortex', 'parietal_cortex'],
            ContentType.MEMORY: ['hippocampus', 'temporal_cortex'], 
            ContentType.LANGUAGE: ['temporal_cortex', 'prefrontal_cortex'],
            ContentType.EMOTION: ['amygdala', 'insular_cortex'],
            ContentType.CREATIVE: ['temporal_cortex', 'prefrontal_cortex'],
            ContentType.ANALYTICAL: ['prefrontal_cortex'],
            ContentType.PATTERN: ['occipital_cortex', 'parietal_cortex'],
            ContentType.TEMPORAL: ['hippocampus', 'cerebellum']
        }
        
        # Keywords for content type detection
        self.keyword_mapping = {
            # Reasoning keywords
            'analyze': ContentType.REASONING,
            'logic': ContentType.REASONING,
            'reason': ContentType.REASONING,
            'conclude': ContentType.REASONING,
            'deduce': ContentType.REASONING,
            'infer': ContentType.REASONING,
            
            # Memory keywords
            'remember': ContentType.MEMORY,
            'recall': ContentType.MEMORY,
            'history': ContentType.MEMORY,
            'past': ContentType.MEMORY,
            'memory': ContentType.MEMORY,
            'learned': ContentType.MEMORY,
            
            # Language keywords
            'language': ContentType.LANGUAGE,
            'grammar': ContentType.LANGUAGE,
            'syntax': ContentType.LANGUAGE,
            'semantic': ContentType.LANGUAGE,
            'linguistic': ContentType.LANGUAGE,
            'word': ContentType.LANGUAGE,
            
            # Emotional keywords
            'emotion': ContentType.EMOTION,
            'feel': ContentType.EMOTION,
            'happy': ContentType.EMOTION,
            'sad': ContentType.EMOTION,
            'angry': ContentType.EMOTION,
            'afraid': ContentType.EMOTION,
            
            # Creative keywords
            'create': ContentType.CREATIVE,
            'art': ContentType.CREATIVE,
            'design': ContentType.CREATIVE,
            'imagine': ContentType.CREATIVE,
            'creative': ContentType.CREATIVE,
            'novel': ContentType.CREATIVE,
            
            # Analytical keywords
            'calculate': ContentType.ANALYTICAL,
            'compute': ContentType.ANALYTICAL,
            'solve': ContentType.ANALYTICAL,
            'mathematical': ContentType.ANALYTICAL,
            'statistical': ContentType.ANALYTICAL,
            
            # Pattern keywords
            'pattern': ContentType.PATTERN,
            'visual': ContentType.PATTERN,
            'image': ContentType.PATTERN,
            'recognize': ContentType.PATTERN,
            'classify': ContentType.PATTERN,
            
            # Temporal keywords
            'time': ContentType.TEMPORAL,
            'sequence': ContentType.TEMPORAL,
            'order': ContentType.TEMPORAL,
            'temporal': ContentType.TEMPORAL,
            'timeline': ContentType.TEMPORAL,
        }
        # External lexicon allowing direct word->zone mapping (from vocab_src)
        self.external_lexicon: Dict[str, str] = {}
    
    def analyze_content(self, text: str) -> Dict[ContentType, float]:
        """Analyze text to determine content types and their weights"""
        if not text:
            return {ContentType.REASONING: 1.0}  # Default
        
        text_lower = text.lower()
        content_scores = {ct: 0.0 for ct in ContentType}
        
        # Score based on keyword matches
        for keyword, content_type in self.keyword_mapping.items():
            if keyword in text_lower:
                content_scores[content_type] += 1.0
        
        # Normalize scores
        total_score = sum(content_scores.values())
        if total_score > 0:
            content_scores = {ct: score/total_score for ct, score in content_scores.items()}
        else:
            # Default if no keywords found
            content_scores[ContentType.REASONING] = 1.0
        
        return content_scores
    
    def route_to_zones(self, content_scores: Dict[ContentType, float]) -> Dict[str, float]:
        """Route content types to brain zones with weights"""
        zone_weights = {}
        
        for content_type, score in content_scores.items():
            if score > 0.0:
                zones = self.content_to_zones.get(content_type, ['prefrontal_cortex'])
                for zone in zones:
                    if zone not in zone_weights:
                        zone_weights[zone] = 0.0
                    zone_weights[zone] += score / len(zones)  # Distribute score among zones
        
        return zone_weights

    # Convenience routing mirroring simplified mapping
    def route_text_to_zones(self, text: str) -> List[str]:
        """Route raw text to zone names based on keyword mapping."""
        if not text:
            return ['prefrontal_cortex']
        mapping = {
            # Reasoning
            'analyze': 'prefrontal_cortex', 'reason': 'prefrontal_cortex', 'logic': 'prefrontal_cortex',
            'think': 'prefrontal_cortex', 'solve': 'prefrontal_cortex', 'plan': 'prefrontal_cortex', 'decide': 'prefrontal_cortex',
            # Memory
            'remember': 'hippocampus', 'recall': 'hippocampus', 'memory': 'hippocampus', 'history': 'hippocampus', 'past': 'hippocampus',
            'context': 'hippocampus', 'experience': 'hippocampus',
            # Language/creativity
            'create': 'temporal_cortex', 'art': 'temporal_cortex', 'language': 'temporal_cortex', 'word': 'temporal_cortex',
            'semantic': 'temporal_cortex', 'meaning': 'temporal_cortex', 'creative': 'temporal_cortex', 'imagine': 'temporal_cortex',
            # Precision/fine-tuning
            'calculate': 'cerebellum', 'precise': 'cerebellum', 'fine': 'cerebellum', 'tune': 'cerebellum', 'adjust': 'cerebellum',
            'correct': 'cerebellum', 'refine': 'cerebellum',
        }
        text_lower = text.lower()
        zones = {zone for kw, zone in mapping.items() if kw in text_lower}
        # Match external lexicon tokens
        if self.external_lexicon:
            for tok, zone in self.external_lexicon.items():
                if tok.lower() in text_lower:
                    zones.add(zone)
        if not zones:
            zones.add('prefrontal_cortex')
        return list(zones)

    def set_external_lexicon(self, lex: Dict[str, str]):
        """Install external token->zone mapping (e.g., from vocab_src datasets)."""
        self.external_lexicon = {str(k).lower(): str(v) for k, v in lex.items()}

    def build_external_lexicon_from_dir(self, path: str, default_zone: str = 'prefrontal_cortex') -> Dict[str, str]:
        """Best-effort loader for jsonl/csv/txt in a directory to token->zone mapping.
        Heuristics: for jsonl, look for keys like 'word','token','term','text'; for csv, first column.
        Zone default derived from filename hints.
        """
        import os, json, csv
        mapping: Dict[str, str] = {}
        if not os.path.isdir(path):
            return mapping
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if not os.path.isfile(fpath):
                continue
            zone = default_zone
            fn = fname.lower()
            if 'amygdala' in fn or 'emotion' in fn:
                zone = 'amygdala'
            elif 'hippocampus' in fn or 'historical' in fn or 'facts' in fn:
                zone = 'hippocampus'
            elif 'temporal' in fn or 'semantic' in fn or 'language' in fn:
                zone = 'temporal_cortex'
            elif 'cerebellum' in fn or 'compute' in fn or 'precision' in fn:
                zone = 'cerebellum'
            elif 'prefrontal' in fn or 'intent' in fn or 'principles' in fn:
                zone = 'prefrontal_cortex'
            try:
                if fn.endswith('.jsonl'):
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            for key in ('word','token','term','text'):
                                if key in obj and isinstance(obj[key], str):
                                    mapping[obj[key].lower()] = zone
                elif fn.endswith('.csv'):
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if not row:
                                continue
                            mapping[str(row[0]).lower()] = zone
                elif fn.endswith('.txt'):
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            w = line.strip()
                            if w:
                                mapping[w.lower()] = zone
            except Exception:
                continue
        self.set_external_lexicon(mapping)
        return mapping

class NeuromorphicProcessor(BaseProcessor):
    """Enhanced processor with neuromorphic brain zone routing"""
    
    def __init__(self, 
                 d_model: int = 512,
                 processing_mode: ProcessingMode = ProcessingMode.NEUROMORPHIC,
                 event_bus: Optional[EventBus] = None):
        self.d_model = d_model
        self.processing_mode = processing_mode
        self.event_bus = event_bus
        
        # Content analysis and routing
        self.content_router = ContentRouter()
        
        # Brain zone processors (will be set by brain initialization)
        self.zone_processors = {}
        # Optional capabilities registry for zones
        # e.g., {'prefrontal_cortex': {'reasoning','analytical'}}
        self.zone_capabilities: Dict[str, set] = {}
        # Router mode and optional objects for liquid/topk gating
        self._router_mode: str = 'keyword'
        self._liquid_gating = None
        self._router_zone_names: List[str] = []
        # Optional plasticity engine (disabled by default)
        self._plasticity_engine: Optional["NeuralPlasticityEngine"] = None
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'content_type_distribution': {ct.value: 0 for ct in ContentType},
            'zone_utilization': {},
            'average_processing_time': 0.0
        }
        
        # Basic fallback processor
        self.basic_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        # Optional lazy projector for mismatched embedding dims (e.g., 1024 -> d_model)
        self._input_projector: Optional[nn.Linear] = None

    def _maybe_project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to self.d_model if last-dim != d_model. Lazy-create projector."""
        if not isinstance(x, torch.Tensor):
            return x
        in_dim = int(x.size(-1))
        if in_dim == int(self.d_model):
            return x
        # (Re)create projector if needed
        if self._input_projector is None or getattr(self._input_projector, 'in_features', None) != in_dim:
            self._input_projector = nn.Linear(in_dim, int(self.d_model), bias=False)
        proj = self._input_projector.to(device=x.device, dtype=x.dtype)
        return proj(x)
    
    def set_zone_processors(self, zone_processors: Dict[str, Any]):
        """Set brain zone processors from brain initialization"""
        self.zone_processors = zone_processors
        # Keep plasticity engine in sync if enabled
        if self._plasticity_engine is not None:
            self._plasticity_engine.set_brain_zones(self._collect_neuromorphic_zones())
    
    def set_zone_capabilities(self, capabilities: Dict[str, List[str]]):
        """Register zone capabilities for planning (optional)."""
        self.zone_capabilities = {z: set(caps) for z, caps in capabilities.items()}

    def set_router_mode(self, mode: str = 'keyword'):
        assert mode in ('keyword', 'liquid', 'topk'), "Unsupported router mode"
        self._router_mode = mode

    def _get_embedding_from_context(self, input: torch.Tensor, context: Optional[Dict[str, Any]]) -> np.ndarray:
        if context and 'embedding' in context:
            emb = context['embedding']
            return np.asarray(emb, dtype=np.float32)
        # fallback: mean over sequence and batch
        if isinstance(input, torch.Tensor):
            if input.dim() == 3:
                return input.mean(dim=(0,1)).detach().cpu().numpy().astype(np.float32)
            return input.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        return np.zeros((self.d_model,), dtype=np.float32)

    def build_plan(self, text: str = "", intents: Optional[List[str]] = None, top_k: int = 3, embedding: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """Construct an ordered list of (zone_name, weight) to execute.
        Weights are influenced by keyword routing and optional intents matched against zone_capabilities.
        """
        # Router selection
        if self._router_mode in ('liquid','topk') and self.zone_processors:
            # Initialize mapping and gating if needed
            if not self._router_zone_names:
                self._router_zone_names = list(self.zone_processors.keys())
            if self._router_mode == 'liquid':
                if self._liquid_gating is None:
                    from liquidmoe.gating import LiquidGating
                    self._liquid_gating = LiquidGating(dim=self.d_model, n_experts=len(self._router_zone_names), top_k=min(top_k, len(self._router_zone_names)))
                # Prefer caller-provided embedding, else compute a fallback zero-mean vector
                if embedding is not None:
                    h = embedding.astype(np.float32)
                else:
                    h = np.zeros((self.d_model,), dtype=np.float32)
                idx, gates = self._liquid_gating.route(h)
                active = [self._router_zone_names[int(i)] for i in idx]
                base = np.asarray(gates, dtype=np.float64)
            else:
                # topk router: quick score on dummy token to get per-expert weights
                if not hasattr(self, '_topk_router') or self._topk_router is None:
                    try:
                        from liquidmoe.moes.router import TopKRouter
                        self._topk_router = TopKRouter(d_model=self.d_model, num_experts=len(self._router_zone_names), top_k=min(top_k, len(self._router_zone_names)))
                    except Exception:
                        class _StubTopKRouter:
                            def __init__(self, d_model:int, num_experts:int, top_k:int=1):
                                self.top_k = max(1, top_k)
                            def __call__(self, x):
                                B,S,_ = x.shape
                                idx = torch.zeros(B,S,self.top_k, dtype=torch.long, device=x.device)
                                gates = torch.ones(B,S,self.top_k, dtype=x.dtype, device=x.device)
                                return idx, gates, None
                        self._topk_router = _StubTopKRouter(self.d_model, len(self._router_zone_names), min(top_k, len(self._router_zone_names)))
                dummy = torch.zeros(1, 1, self.d_model)
                indices, gates, aux = self._topk_router(dummy)
                idx = indices[0,0].cpu().numpy()
                gat = gates[0,0].detach().cpu().numpy()
                active = [self._router_zone_names[int(i)] for i in idx]
                base = gat.astype(np.float64)
        else:
            active = self.content_router.route_text_to_zones(text) if hasattr(self.content_router, 'route_text_to_zones') else list(self.zone_processors.keys())
            base = np.ones(len(active), dtype=np.float64)
        if not active:
            return []
        # Boost weights for capability matches
        if intents:
            intents_set = set(intents)
            for i, z in enumerate(active):
                caps = self.zone_capabilities.get(z, set())
                matches = len(intents_set.intersection(caps))
                if matches > 0:
                    base[i] *= (1.0 + 0.75 * matches)
        weights = np_softmax(base)
        items = list(zip(active, weights))
        # Prefer prefrontal first, cerebellum last if present
        items.sort(key=lambda p: (p[0] == 'cerebellum', p[0] != 'prefrontal_cortex'))
        if top_k and len(items) > top_k:
            items = items[:top_k]
        return items

    def run_plan(self, input: torch.Tensor, text: str = "", intents: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None, top_k: int = 3) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute planned zone sequence and return output and plan info."""
        # Ensure input matches internal model dim
        if isinstance(input, torch.Tensor):
            input = self._maybe_project_input(input)
        emb = self._get_embedding_from_context(input, context)
        plan = self.build_plan(text, intents=intents, top_k=top_k, embedding=emb)
        zone_weights = {z: float(w) for z, w in plan if z in self.zone_processors}
        output = self._process_through_zones(input, zone_weights, context)
        info = {
            'mode': 'neuromorphic' if zone_weights else 'basic',
            'plan': plan,
            'selected_zones': list(zone_weights.keys())
        }
        # Always-on step for plasticity engine (if enabled)
        if self._plasticity_engine is not None:
            self._plasticity_engine.step()
        return output, info

    def update_router_rewards(self, selected_zones: List[str], reward: float):
        """Propagate a scalar reward to liquid router (if active)."""
        if self._router_mode != 'liquid' or self._liquid_gating is None or not selected_zones:
            return
        idx = [self._router_zone_names.index(z) for z in selected_zones if z in self._router_zone_names]
        if idx:
            try:
                self._liquid_gating.update_rewards(idx, float(reward))
            except Exception:
                pass
    
    def process(self, input: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Process input through neuromorphic brain zones"""
        
        # Store zone activities for this processing step
        self._current_zone_activities = {}
        
        if self.processing_mode == ProcessingMode.BASIC:
            return self.basic_processor(input)
        
        # Extract text context for routing
        text_context = ""
        if context and 'text' in context:
            text_context = context['text']
        elif isinstance(input, str):
            text_context = input
            # Convert to tensor if needed
            input = torch.randn(1, self.d_model)  # Placeholder - replace with proper encoding
        # Align tensor input to expected d_model
        if isinstance(input, torch.Tensor):
            input = self._maybe_project_input(input)
        
        # Analyze content and route to zones
        # Prefer direct routing to zone names if available
        content_scores = {}
        if hasattr(self.content_router, 'route_text_to_zones'):
            active_zone_names = self.content_router.route_text_to_zones(text_context)
            zone_weights = {z: 1.0 / max(1, len(active_zone_names)) for z in active_zone_names}
        else:
            content_scores = self.content_router.analyze_content(text_context)
            zone_weights = self.content_router.route_to_zones(content_scores)
        
        # Update statistics (guard if empty content_scores)
        if content_scores:
            self._update_stats(content_scores, zone_weights)
        
        # Process through selected zones
        out = self._process_through_zones(input, zone_weights, context)
        if self._plasticity_engine is not None:
            self._plasticity_engine.step()
        return out
    
    def _process_through_zones(self, input: torch.Tensor, 
                              zone_weights: Dict[str, float],
                              context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Process input through weighted brain zones"""
        
        if not self.zone_processors or not zone_weights:
            if isinstance(input, torch.Tensor):
                input = self._maybe_project_input(input)
            return self.basic_processor(input)
        
        zone_outputs = []
        zone_weights_list = []
        
        for zone_name, weight in zone_weights.items():
            if weight > 0.01 and zone_name in self.zone_processors:  # Minimum threshold
                zone_processor = self.zone_processors[zone_name]
                
                try:
                    if hasattr(zone_processor, 'process'):
                        zone_output, zone_activity = zone_processor.process(input, context)
                    else:
                        result = zone_processor(input, context=context)
                        if isinstance(result, tuple) and len(result) == 2:
                            zone_output, zone_activity = result
                        else:
                            zone_output = result
                            zone_activity = {}
                    
                    zone_outputs.append(zone_output)
                    zone_weights_list.append(weight)
                    
                    # Store zone activity
                    self._current_zone_activities[zone_name] = zone_activity
                    
                    # Broadcast processing event
                    if self.event_bus:
                        self.event_bus.broadcast_neuron_fired({
                            'processor': 'neuromorphic',
                            'zone': zone_name,
                            'weight': weight,
                            'activity': zone_activity
                        })
                    # Feed plasticity engine with fresh zone metrics
                    if self._plasticity_engine is not None:
                        self._plasticity_engine.update_zone_activity(zone_name, zone_activity)
                        
                except Exception as e:
                    # Fallback if zone processing fails
                    print(f"Zone {zone_name} processing failed: {e}, using fallback")
                    continue
        
        # Combine zone outputs with shape-awareness
        if zone_outputs:
            # Use stable softmax from maths module
            weights_np = np_softmax(np.array(zone_weights_list, dtype=np.float64))
            weights_tensor = torch.tensor(weights_np, dtype=zone_outputs[0].dtype, device=zone_outputs[0].device)

            # Determine rank and stack along a new leading axis
            if zone_outputs[0].dim() == 2:  # [B, D]
                stacked_outputs = torch.stack(zone_outputs, dim=0)  # [Z, B, D]
                combined_output = torch.einsum('z,zbd->bd', weights_tensor, stacked_outputs)
            elif zone_outputs[0].dim() == 3:  # [B, T, D]
                stacked_outputs = torch.stack(zone_outputs, dim=0)  # [Z, B, T, D]
                combined_output = torch.einsum('z,zbtd->btd', weights_tensor, stacked_outputs)
            else:
                # Fallback simple mean if unexpected shape
                stacked_outputs = torch.stack(zone_outputs, dim=0)
                combined_output = torch.mean(stacked_outputs, dim=0)

            return combined_output
        else:
            # Fallback to basic processing
            return self.basic_processor(input)

    def _update_stats(self, content_scores: Dict[ContentType, float], zone_weights: Dict[str, float]):
        """Update processing statistics"""
        self.processing_stats['total_processed'] += 1
        for content_type, score in content_scores.items():
            if score > 0:
                self.processing_stats['content_type_distribution'][content_type.value] += 1
        for zone_name, weight in zone_weights.items():
            if zone_name not in self.processing_stats['zone_utilization']:
                self.processing_stats['zone_utilization'][zone_name] = []
            self.processing_stats['zone_utilization'][zone_name].append(weight)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = self.processing_stats.copy()
        avg_utilization = {}
        for zone_name, weights in stats['zone_utilization'].items():
            if weights:
                avg_utilization[zone_name] = {
                    'avg_weight': sum(weights) / len(weights),
                    'max_weight': max(weights),
                    'usage_count': len(weights),
                    'usage_percentage': len(weights) / stats['total_processed'] * 100
                }
        stats['average_zone_utilization'] = avg_utilization
        # Include current zone activities
        if hasattr(self, '_current_zone_activities'):
            stats['zone_activities'] = self._current_zone_activities
        return stats

    def get_recommendations(self) -> List[str]:
        """Get processing optimization recommendations"""
        recommendations = []
        stats = self.get_processing_stats()
        if 'average_zone_utilization' in stats:
            utilization = stats['average_zone_utilization']
            underutilized = [zone for zone, data in utilization.items() if data['usage_percentage'] < 10]
            if underutilized:
                recommendations.append(f"Underutilized zones: {underutilized}. Consider adjusting content routing.")
            overutilized = [zone for zone, data in utilization.items() if data['usage_percentage'] > 80]
            if overutilized:
                recommendations.append(f"Overutilized zones: {overutilized}. Consider load balancing.")
        content_dist = stats['content_type_distribution']
        active_types = [ct for ct, count in content_dist.items() if count > 0]
        if len(active_types) < 3:
            recommendations.append("Low content type diversity. Consider expanding input variety.")
        return recommendations

    # -----------------------------
    # Plasticity engine integration
    # -----------------------------

    def enable_plasticity_engine(self) -> "NeuralPlasticityEngine":
        """Enable a lightweight, always-on plasticity engine that adapts neuron parameters over time.

        The engine listens to 'neuron_fired' events (if an EventBus is provided)
        and also ingests zone activity returned during processing.
        """
        if self._plasticity_engine is None:
            if self.event_bus is None:
                self.event_bus = EventBus()
            self._plasticity_engine = NeuralPlasticityEngine(self.event_bus, self._collect_neuromorphic_zones())
        return self._plasticity_engine

    def _collect_neuromorphic_zones(self) -> Dict[str, Any]:
        zones: Dict[str, Any] = {}
        for name, zone in (self.zone_processors or {}).items():
            neu = getattr(zone, 'neuromorphic_processor', None)
            if neu is not None:
                zones[name] = neu
        return zones


# -----------------------------------------------------------------------------
# Neural Plasticity Engine (lightweight, framework-integrated)
# -----------------------------------------------------------------------------

class PlasticityType(_EnumAlias):
    HOMEOSTATIC = "homeostatic"
    INTRINSIC = "intrinsic"


@dataclass
class PlasticityRule:
    plasticity_type: PlasticityType
    target_rate: float = 0.10
    learning_rate: float = 0.01
    enabled: bool = True


class NeuralPlasticityEngine:
    """Lightweight engine to maintain healthy activity via homeostatic/intrinsic plasticity.

    - Consumes 'neuron_fired' events and per-zone activity snapshots
    - Nudges neuron-level baselines and thresholds towards target firing rates
    - Designed to be safe-by-default and test-friendly (no background loops)
    """

    def __init__(self, event_bus: EventBus, brain_zones: Dict[str, Any]):
        self.event_bus = event_bus
        self._zones = brain_zones
        self._zone_rate: Dict[str, float] = {}
        self._last_step_ts: float = 0.0
        self.rules: List[PlasticityRule] = [
            PlasticityRule(PlasticityType.HOMEOSTATIC, target_rate=0.10, learning_rate=0.01, enabled=True),
            PlasticityRule(PlasticityType.INTRINSIC, target_rate=0.15, learning_rate=0.005, enabled=True),
        ]
        # Subscribe to neuron events
        self.event_bus.subscribe('neuron_fired', self._on_neuron_fired)

    def set_brain_zones(self, brain_zones: Dict[str, Any]) -> None:
        self._zones = brain_zones or {}

    def update_zone_activity(self, zone_name: str, zone_activity: Dict[str, Any]) -> None:
        try:
            rate = float(zone_activity.get('avg_firing_rate', 0.0))
            if rate >= 0.0:
                self._zone_rate[zone_name] = rate
        except Exception:
            pass

    def _on_neuron_fired(self, event) -> None:
        data = event.data
        zone = data.get('zone')
        fr = data.get('firing_rate')
        if zone is not None and isinstance(fr, (int, float)):
            self._zone_rate[zone] = float(fr)

    def step(self) -> None:
        """One adaptation step. Intended to be called frequently during processing."""
        if not self._zones:
            return
        for zone_name, neu_zone in self._zones.items():
            try:
                current_rate = float(self._zone_rate.get(zone_name, 0.0))
                # Iterate each neuron group inside the neuromorphic zone
                neuron_groups = getattr(neu_zone, 'neuron_groups', None)
                if neuron_groups is None:
                    continue
                for _, neuron_module in neuron_groups.items():
                    # Homeostatic drive for Izhikevich/AdEx via bias current 'homeo_i'
                    homeo_i = getattr(neuron_module, 'homeo_i', None)
                    if homeo_i is not None and hasattr(neuron_module, 'target_firing_rate'):
                        target = float(getattr(neuron_module, 'target_firing_rate', 0.10))
                        eta = float(getattr(neuron_module, 'homeostasis_lr', 0.01))
                        err = current_rate - target
                        with torch.no_grad():
                            homeo_i.add_(torch.tensor(-eta * err, device=homeo_i.device, dtype=homeo_i.dtype)).clamp_(-5.0, 5.0)
                    # Intrinsic plasticity for LIF: nudge threshold based on activity
                    threshold = getattr(neuron_module, 'threshold', None)
                    if threshold is not None and hasattr(neuron_module, 'surrogate_slope'):
                        target = 0.15
                        eta = 0.005
                        err = current_rate - target
                        with torch.no_grad():
                            threshold.add_(torch.tensor(eta * err * 1.5, device=threshold.device, dtype=threshold.dtype)).clamp_(-100.0, -10.0)
            except Exception:
                # Never allow adaptive step to disrupt main processing
                continue

class EventDrivenProcessor(BaseProcessor):
    """Event-driven processor that responds to brain events"""
    
    def __init__(self, event_bus: EventBus, neuromorphic_processor: NeuromorphicProcessor):
        self.event_bus = event_bus
        self.neuromorphic_processor = neuromorphic_processor
        
        # Subscribe to relevant events
        self.event_bus.subscribe('neuron_fired', self._handle_neuron_fired)
        self.event_bus.subscribe('brain_stats_updated', self._handle_stats_updated)
        
        # Processing queue for event-driven responses
        self.processing_queue = []
        self.event_responses = {}
    
    def _handle_neuron_fired(self, event):
        """Handle neuron firing events"""
        event_data = event.data
        print(event_data)
        
        # Adjust processing based on firing patterns
        if 'firing_rate' in event_data:
            firing_rate = event_data['firing_rate']
            layer = event_data.get('layer', 'unknown')
            print(f"FIRE: layer={layer} rate={firing_rate}")
            
            if firing_rate < 0.001:
                self.event_responses[layer] = 'increase_stimulation'
            elif firing_rate > 0.8:
                self.event_responses[layer] = 'decrease_stimulation'
    
    def _handle_stats_updated(self, event):
        """Handle brain statistics updates"""
        stats_data = event.data
        
        # Queue processing adjustments based on stats
        if 'training_stability' in stats_data:
            stability = stats_data['training_stability']
            if stability == 'exploding':
                self.processing_queue.append({'action': 'reduce_learning_rate'})
            elif stability == 'vanishing':
                self.processing_queue.append({'action': 'increase_learning_rate'})
    
    def process(self, input: Any) -> Any:
        """Process with event-driven adaptations"""
        
        # Apply any queued adaptations
        self._apply_queued_adaptations()
        
        # Process through neuromorphic processor
        output = self.neuromorphic_processor.process(input)
        
        return output
    
    def _apply_queued_adaptations(self):
        """Apply queued processing adaptations"""
        while self.processing_queue:
            adaptation = self.processing_queue.pop(0)
            action = adaptation.get('action')
            
            if action == 'reduce_learning_rate':
                # Could adjust processor parameters here
                pass
            elif action == 'increase_learning_rate':
                # Could adjust processor parameters here
                pass

class MultiModalProcessor(BaseProcessor):
    """Processor for handling multiple input modalities"""
    
    def __init__(self, neuromorphic_processor: NeuromorphicProcessor):
        self.neuromorphic_processor = neuromorphic_processor
        
        # Modality-specific preprocessors
        self.text_preprocessor = nn.Linear(768, neuromorphic_processor.d_model)  # Example for BERT embeddings
        self.image_preprocessor = nn.Linear(2048, neuromorphic_processor.d_model)  # Example for ResNet features
        self.audio_preprocessor = nn.Linear(128, neuromorphic_processor.d_model)  # Example for audio features
    
    def process(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process multi-modal input"""
        
        processed_inputs = []
        combined_context = {}
        
        # Process each modality
        if 'text' in input:
            text_processed = self.text_preprocessor(input['text'])
            processed_inputs.append(text_processed)
            combined_context['has_text'] = True
        
        if 'image' in input:
            image_processed = self.image_preprocessor(input['image'])
            processed_inputs.append(image_processed)
            combined_context['has_image'] = True
        
        if 'audio' in input:
            audio_processed = self.audio_preprocessor(input['audio'])
            processed_inputs.append(audio_processed)
            combined_context['has_audio'] = True
        
        # Combine modalities
        if processed_inputs:
            combined_input = torch.stack(processed_inputs).mean(dim=0)
        else:
            raise ValueError("No valid input modalities provided")
        
        # Process through neuromorphic zones
        return self.neuromorphic_processor.process(combined_input, {'multimodal': True, **combined_context})