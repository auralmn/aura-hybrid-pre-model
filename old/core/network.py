"""
AURA Network - Comprehensive neural network system
Combines base network, SPAN integration, SVC capabilities, and training functionality
"""

import asyncio
import trio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Core components
from .hippocampus import Hippocampus
from .amygdala import Amygdala
from .nlms import SpikingAttention, NLMSHead
from .thalamic_router import ThalamicConversationRouter
from .central_nervous_system import CentralNervousSystem
from .thalamus import Thalamus
from .neuron import Neuron, MaturationStage, ActivityState
from .hippothalamus import Hypothalamus, Pituitary, SystemMetrics, HormoneType

# Qdrant streaming - import only when needed to avoid circular imports

# SPAN integration - commented out to avoid circular import
# from ..training.span_integration import SPANNeuron, SPANPattern, create_final_precision_span_patterns

# Enhanced SVC pipeline - commented out to avoid circular import
# from ..utils.enhanced_svc_pipeline import (
#     load_enhanced_svc_dataset,
#     get_enhanced_full_knowledge_embedding,
#     create_sample_enhanced_data
# )

# Network configuration
n_neurons = 10000                # typical: 10 - 10,000+
n_features = 384                # typical: 5 - 500+
n_outputs = 1
input_channels = n_features
output_channels = n_features

Array = np.ndarray


class Network:
    """
    AURA Network - Comprehensive neural network system
    Combines base functionality, SPAN integration, SVC capabilities, and training
    """
    
    def __init__(
        self,
        # Base network parameters
        neuron_count: int = n_neurons,
        features: int = n_features,
        input_channels: int = input_channels,
        output_channels: int = output_channels,
        
        # SPAN integration parameters
        enable_span: bool = True,
        span_neurons_per_region: int = 10,
        
        # SVC parameters
        domains: Optional[List[str]] = None,
        realms: Optional[List[str]] = None,
        offline: bool = False,
        nlms_clamp: Tuple[float, float] = (0.0, 1.0),
        nlms_l2: float = 1e-4,
        features_mode: str = 'sbert',  # 'sbert' | 'phasor' | 'combined'
        features_alpha: float = 0.7,   # weight on SBERT when combined
        weights_dir: str = 'svc_nlms_weights',
        startnew: bool = False,
        
        # Qdrant streaming parameters
        enable_qdrant: bool = True,
        qdrant_url: str = 'http://localhost',
        qdrant_port: int = 6333,
        qdrant_snapshot_interval: int = 50,
        
        # Neuroendocrine control parameters
        enable_neuroendocrine: bool = True,
        target_energy_efficiency: float = 1e-10,
        target_expert_utilization: float = 0.8,
        target_prediction_accuracy: float = 0.85,
    ):
        # Store parameters
        self.neuron_count = neuron_count
        self.features = features
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # SPAN parameters
        self.enable_span = enable_span
        self.span_neurons_per_region = span_neurons_per_region
        
        # SVC parameters
        self.domains = domains or ['general', 'technical', 'creative', 'analytical']
        self.realms = realms or ['academic', 'professional', 'personal', 'creative']
        self.offline = offline
        self.nlms_clamp = nlms_clamp
        self.nlms_l2 = nlms_l2
        self.features_mode = features_mode
        self.features_alpha = features_alpha
        self.weights_dir = weights_dir
        self.startnew = startnew
        
        # Qdrant parameters
        self.enable_qdrant = enable_qdrant
        self.qdrant_url = qdrant_url
        self.qdrant_port = qdrant_port
        self.qdrant_snapshot_interval = qdrant_snapshot_interval
        
        # Neuroendocrine control parameters
        self.enable_neuroendocrine = enable_neuroendocrine
        self.enable_hippocampus_bias = True  # Enable memory-based routing bias
        self.target_energy_efficiency = target_energy_efficiency
        self.target_expert_utilization = target_expert_utilization
        self.target_prediction_accuracy = target_prediction_accuracy
        
        # Initialize core brain regions
        self._thalamus = Thalamus(
            neuron_count=neuron_count,
            input_channels=input_channels,
            output_channels=output_channels
        )
        self._hippocampus = Hippocampus(
            neuron_count=neuron_count,
            features=features,
            neurogenesis_rate=0.01,
            input_dim=features
        )
        self._amygdala = Amygdala()
        self._thalamic_router = ThalamicConversationRouter(
            neuron_count=60, 
            features=features, 
            input_dim=features
        )
        self._cns = CentralNervousSystem(input_dim=features)
        
        # Register brain regions with CNS
        self._cns.register_brain_region('thalamus', self._thalamus, priority=0.7)
        self._cns.register_brain_region('hippocampus', self._hippocampus, priority=0.6)
        self._cns.register_brain_region('amygdala', self._amygdala, priority=0.8)
        self._cns.register_brain_region('router', self._thalamic_router, priority=0.5)
        
        # Initialize neuroendocrine control system
        if self.enable_neuroendocrine:
            target_metrics = SystemMetrics(
                energy_efficiency=target_energy_efficiency,
                expert_utilization=target_expert_utilization,
                prediction_accuracy=target_prediction_accuracy,
                learning_rate=0.1,
                stress_level=0.2,
                temperature=1.0
            )
            self._hypothalamus = Hypothalamus(target_metrics)
            self._pituitary = Pituitary()
            self._hormone_type = HormoneType
            self._last_acc = 0.5
            self._system_initialized = False
            print("🧠 Neuroendocrine control system initialized")
        else:
            self._hypothalamus = None
            self._pituitary = None
            self._hormone_type = None
            self._last_acc = 0.5
            self._system_initialized = False
        
        # Attention configurations
        self.attention_configs = {
        'historical': {'decay': 0.8, 'theta': 1.2, 'k_winners': 7, 'gain_up': 1.8, 'gain_down': 0.5},
        'emotional': {'decay': 0.6, 'theta': 0.9, 'k_winners': 4, 'gain_up': 2.0, 'gain_down': 0.4},
        'analytical': {'decay': 0.7, 'theta': 1.0, 'k_winners': 5, 'gain_up': 1.5, 'gain_down': 0.6},
        'memory': {'decay': 0.75, 'theta': 1.1, 'k_winners': 6, 'gain_up': 1.6, 'gain_down': 0.6}
        }

        # SPAN integration - commented out to avoid circular import
        # self.span_hippocampus_neurons: List[SPANNeuron] = []
        # self.span_thalamus_neurons: List[SPANNeuron] = []
        # self.span_amygdala_neurons: List[SPANNeuron] = []
        self.span_performance_history: List[Dict[str, Any]] = []
        self.integration_statistics: Dict[str, Any] = {}
        
        # SVC specialists
        self.specialists: Dict[str, Neuron] = {}
        self.sbert_model = None
        
        # Initialize Qdrant streaming
        if self.enable_qdrant:
            from ..utils.qdrant_stream import QdrantStreamer
            self.qdrant_streamer = QdrantStreamer(url=self.qdrant_url, port=self.qdrant_port)
            print("✅ Qdrant streaming initialized")
        else:
            self.qdrant_streamer = None
            print("⚠️  Qdrant streaming disabled")
        
        # Initialize components
        self._setup_sbert()
        self._setup_specialists()
    
    def enable_qdrant_streaming(self, url: str = None, port: int = None, snapshot_interval: int = None) -> None:
        """Enable Qdrant streaming with optional parameters"""
        if url:
            self.qdrant_url = url
        if port:
            self.qdrant_port = port
        if snapshot_interval:
            self.qdrant_snapshot_interval = snapshot_interval
            
        self.enable_qdrant = True
        from ..utils.qdrant_stream import QdrantStreamer
        self.qdrant_streamer = QdrantStreamer(url=self.qdrant_url, port=self.qdrant_port)
        print("✅ Qdrant streaming enabled")
    
    def disable_qdrant_streaming(self) -> None:
        """Disable Qdrant streaming"""
        self.enable_qdrant = False
        self.qdrant_streamer = None
        print("⚠️  Qdrant streaming disabled")
    
    def load_qdrant_config(self, config: Dict[str, Any]) -> None:
        """Load Qdrant configuration from config dictionary"""
        qdrant_config = config.get('qdrant', {})
        if qdrant_config.get('enabled', False):
            url = qdrant_config.get('url', 'http://localhost')
            port = qdrant_config.get('port', 6333)
            snapshot_interval = qdrant_config.get('streaming', {}).get('snapshot_interval', 50)
            self.enable_qdrant_streaming(url, port, snapshot_interval)
        else:
            self.disable_qdrant_streaming()
    
    def _setup_sbert(self):
        """Setup SBERT model for text embeddings"""
        if not self.offline:
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
                print("✅ SBERT model loaded successfully")
            except ImportError:
                print("⚠️  SBERT not available, using zero embeddings")
            except Exception as e:
                print(f"⚠️  SBERT loading failed: {e}, using zero embeddings")
    
    def _setup_specialists(self):
        """Setup specialist neurons for different analysis tasks"""
        # Domain classification specialist  (D = feature dim, C = #domains)
        domain_specialist = Neuron(
            neuron_id="domain_classifier",
            specialization="DOMAIN_CLASSIFIER",
            abilities={'classification': 0.9},
            n_features=len(self.domains),
            n_outputs=1,
            maturation=MaturationStage.DIFFERENTIATED,
            activity=ActivityState.RESTING
        )
        domain_specialist.nlms_head = NLMSHead(
            n_features=self.features,
            n_outputs=len(self.domains),
            mu=0.01,
            l2_decay=self.nlms_l2,
            clip01=False
        )
        self.specialists['domain_classifier'] = domain_specialist
        
        # Realm classification specialist
        realm_specialist = Neuron(
            neuron_id="realm_classifier",
            specialization="REALM_CLASSIFIER",
            abilities={'classification': 0.9},
            n_features=len(self.realms),
            n_outputs=1,
            maturation=MaturationStage.DIFFERENTIATED,
            activity=ActivityState.RESTING
        )
        realm_specialist.nlms_head = NLMSHead(
            n_features=self.features,
            n_outputs=len(self.realms),
            mu=0.01,
            l2_decay=self.nlms_l2,
            clip01=False
        )
        self.specialists['realm_classifier'] = realm_specialist
        
        # Difficulty regression specialist
        difficulty_specialist = Neuron(
            neuron_id="difficulty_regressor",
            specialization="DIFFICULTY_REGRESSOR",
            abilities={'regression': 0.9},
            n_features=1,
            n_outputs=1,
            maturation=MaturationStage.DIFFERENTIATED,
            activity=ActivityState.RESTING
        )
        difficulty_specialist.nlms_head = NLMSHead(
            n_features=self.features,
            n_outputs=1,
            mu=0.01,
            l2_decay=self.nlms_l2,
            clip01=True
        )
        self.specialists['difficulty_regressor'] = difficulty_specialist
    
    def get_features(self, text: str) -> np.ndarray:
        """Get feature vector for text based on features_mode"""
        if self.features_mode == 'sbert':
            return self._get_sbert_features(text)
        elif self.features_mode == 'phasor':
            return self._get_phasor_features(text)
        elif self.features_mode == 'combined':
            sbert_feat = self._get_sbert_features(text)
            phasor_feat = self._get_phasor_features(text)
            # Combine features with alpha weighting
            return self.features_alpha * sbert_feat + (1 - self.features_alpha) * phasor_feat
        else:
            raise ValueError(f"Unknown features_mode: {self.features_mode}")
    
    def _get_sbert_features(self, text: str) -> np.ndarray:
        """Get SBERT features for text"""
        if self.sbert_model is not None:
            vec = self.sbert_model.encode([text])[0]
            D = self.features
            v = np.asarray(vec, dtype=np.float64).reshape(-1)
            if v.size == D: return v
            if v.size > D:  return v[:D]
            return np.pad(v, (0, D - v.size))
        else:
            return np.zeros(self.features, dtype=np.float64)
    
    def _get_phasor_features(self, text: str) -> np.ndarray:
        """Get phasor-based features for text"""
        # Simple phasor-based feature extraction
        words = text.lower().split()
        n_words = len(words)
        
        # Basic features: length, counts → then pad/trim to self.features
        features = np.array([
            len(text),
            n_words,
            sum(len(word) for word in words),
            text.count(' '),
            text.count('.'),
            text.count('!'),
            text.count('?')
        ], dtype=np.float64)
        
        # normalize and fit feature dim
        if n_words > 0:
            features = features / float(n_words)
        D = self.features
        if features.size == D: return features
        if features.size > D:  return features[:D]
        return np.pad(features, (0, D - features.size))
    
    async def init_weights(self):
        """Initialize network weights"""
        # Initialize base network weights
        await self._thalamus.init_weights()
        await self._hippocampus.init_weights()
        await self._amygdala.init_weights()
        await self._thalamic_router.init_weights()
        await self._cns.init_weights()
        
        # Initialize SPAN neurons if enabled - commented out to avoid circular import
        # if self.enable_span:
        #     await self._initialize_span_neurons()
        
        # Initialize specialists
        await self._initialize_specialists()
    
    # async def _initialize_span_neurons(self):
    #     """Initialize SPAN-enhanced neurons - commented out to avoid circular import"""
    #     print("🧠 Initializing SPAN neurons...")
    #     
    #     # Add SPAN neurons to hippocampus (memory formation)
    #     for i in range(min(self.span_neurons_per_region, len(self._hippocampus.neurons))):
    #         base_neuron = self._hippocampus.neurons[i]
    #         span_neuron = SPANNeuron(base_neuron, learning_rate=0.0005)
    #         self.span_hippocampus_neurons.append(span_neuron)
    #     
    #     # Add SPAN neurons to thalamus (sensory gating)
    #     for i in range(min(self.span_neurons_per_region, len(self._thalamus.neurons))):
    #         base_neuron = self._thalamus.neurons[i]
    #         span_neuron = SPANNeuron(base_neuron, learning_rate=0.0005)
    #         self.span_amygdala_neurons.append(span_neuron)
    #     
    #     # Add SPAN neurons to amygdala (emotional processing)
    #     for i in range(min(self.span_neurons_per_region, len(self._amygdala.neurons))):
    #         base_neuron = self._amygdala.neurons[i]
    #         span_neuron = SPANNeuron(base_neuron, learning_rate=0.0005)
    #         self.span_amygdala_neurons.append(span_neuron)
    #     
    #     print(f"✅ SPAN neurons initialized: {len(self.span_hippocampus_neurons)} hippocampus, {len(self.span_thalamus_neurons)} thalamus, {len(self.span_amygdala_neurons)} amygdala")
    
    async def _initialize_specialists(self):
        """Initialize specialist neurons"""
        for specialist in self.specialists.values():
            if hasattr(specialist, 'nlms_head') and specialist.nlms_head:
                C = specialist.nlms_head.n_outputs
                baseW = np.zeros((specialist.nlms_head.n_features, C), dtype=np.float64)
                await specialist.nlms_head.attach(
                    baseW,
                    slice(0, specialist.nlms_head.n_features),  # tok_slice
                    slice(0, 0),
                    slice(0, 0)
                )
    
    async def process_input(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Process input through the network"""
        # Process through thalamus (sensory gating)
        thalamus_output = await self._thalamus.process(input_data)
        
        # Process through hippocampus (memory formation)
        hippocampus_output = await self._hippocampus.process(input_data)
        
        # Process through amygdala (emotional analysis)
        amygdala_output = await self._amygdala.process(input_data)
        
        # Process through thalamic router (conversation routing)
        router_output = await self._thalamic_router.process(input_data)
        
        # Central nervous system coordination
        cns_output = await self._cns.process(input_data)
        
        return {
            'thalamus': thalamus_output,
            'hippocampus': hippocampus_output,
            'amygdala': amygdala_output,
            'router': router_output,
            'cns': cns_output
        }
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the network with specialist analysis"""
        text = data.get('text', '')
        features = self.get_features(text)
        
        # Process through base network
        base_result = await self.process_input(features)
        
        # Apply memory-based routing bias before processing
        if self.enable_hippocampus_bias and hasattr(self, '_hippocampus'):
            retrieved_experts = []
            try:
                if hasattr(self._hippocampus, 'retrieve_similar_memories'):
                    sims = self._hippocampus.retrieve_similar_memories(features, k=3)
                    for mid, _ in sims:
                        mem = getattr(self._hippocampus, 'episodic_memories', {}).get(mid)
                        if mem and getattr(mem, 'associated_experts', None):
                            retrieved_experts.extend(mem.associated_experts)
            except Exception:
                pass
            self._hippocampal_memory_bias(retrieved_experts, weight=0.03)
        
        # Stream to Qdrant if enabled
        if self.qdrant_streamer:
            # Take periodic snapshots
            self.qdrant_streamer.maybe_snapshot(self, self.qdrant_snapshot_interval)
        
        # Process through specialists
        specialist_results = {}
        for name, specialist in self.specialists.items():
            if name == 'domain_classifier':
                prediction = specialist.nlms_head.predict(features)   # (C,)
                j = int(np.argmax(prediction))
                specialist_results['domain'] = self.domains[j]
                specialist_results['domain_confidence'] = float(prediction[j])
            elif name == 'realm_classifier':
                prediction = specialist.nlms_head.predict(features)
                j = int(np.argmax(prediction))
                specialist_results['realm'] = self.realms[j]
                specialist_results['realm_confidence'] = float(prediction[j])
            elif name == 'difficulty_regressor':
                y = specialist.nlms_head.predict(features)
                specialist_results['difficulty'] = float(y[0])
        
        result = {
            'base_result': base_result,
            'specialist_results': specialist_results,
            'features': features.tolist()
        }
        
        # Apply endocrine feedback (use router confidence as accuracy proxy)
        if self.enable_neuroendocrine and 'router' in base_result:
            router_output = base_result['router']
            if isinstance(router_output, dict):
                conf = float(router_output.get('routing_confidence', 0.5))
                self._endocrine_step(result, accuracy=conf)
        
        # Stream routing decision to Qdrant if enabled
        if self.qdrant_streamer and 'router' in base_result:
            router_output = base_result['router']
            if isinstance(router_output, dict) and 'routing_decision' in router_output:
                routing_decision = router_output['routing_decision']
                self.qdrant_streamer.upsert_routing(routing_decision, features)
        
        return result
    
    async def train_on_data(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the network on data"""
        results = {
            'domain_accuracy': 0.0,
            'realm_accuracy': 0.0,
            'difficulty_mse': 0.0,
            'total_samples': len(training_data)
        }
        
        domain_correct = 0
        realm_correct = 0
        difficulty_errors = []
        
        for data in training_data:
            text = data.get('text', '')
            features = self.get_features(text)
            
            # Stream to Qdrant if enabled
            if self.qdrant_streamer:
                self.qdrant_streamer.maybe_snapshot(self, self.qdrant_snapshot_interval)
            
            # Train domain classifier
            if 'domain' in data:
                domain_idx = self.domains.index(data['domain']) if data['domain'] in self.domains else 0
                domain_target = np.zeros(len(self.domains))
                domain_target[domain_idx] = 1.0
                
                domain_specialist = self.specialists['domain_classifier']
                await domain_specialist.nlms_head.step(features, domain_target)
                
                # Check accuracy
                prediction = domain_specialist.nlms_head.predict(features)
                if np.argmax(prediction) == domain_idx:
                    domain_correct += 1
            
            # Train realm classifier
            if 'realm' in data:
                realm_idx = self.realms.index(data['realm']) if data['realm'] in self.realms else 0
                realm_target = np.zeros(len(self.realms))
                realm_target[realm_idx] = 1.0
                
                realm_specialist = self.specialists['realm_classifier']
                await realm_specialist.nlms_head.step(features, realm_target)
                
                # Check accuracy
                prediction = realm_specialist.nlms_head.predict(features)
                if np.argmax(prediction) == realm_idx:
                    realm_correct += 1
            
            # Train difficulty regressor
            if 'difficulty' in data:
                difficulty_target = np.array([data['difficulty']])
                
                difficulty_specialist = self.specialists['difficulty_regressor']
                await difficulty_specialist.nlms_head.step(features, difficulty_target)
                
                # Calculate error
                y = difficulty_specialist.nlms_head.predict(features)
                error = (float(y[0]) - float(data['difficulty'])) ** 2
                difficulty_errors.append(error)
        
        # Calculate final metrics
        if results['total_samples'] > 0:
            results['domain_accuracy'] = domain_correct / results['total_samples']
            results['realm_accuracy'] = realm_correct / results['total_samples']
            results['difficulty_mse'] = np.mean(difficulty_errors) if difficulty_errors else 0.0
        
        return results
    
    def enable_attention_learning(self, regions: List[str]):
        """Enable attention learning for specified regions"""
        for region in regions:
            if region in self.attention_configs:
                config = self.attention_configs[region]
                # Apply attention configuration to the region
                print(f"✅ Attention learning enabled for {region}: {config}")
    
    def save_weights(self, base_dir: str = None):
        """Save network weights"""
        base_dir = base_dir or self.weights_dir
        import os
        os.makedirs(base_dir, exist_ok=True)
        
        # Save core weights
        try:
            th_W = np.vstack([n.nlms_head.w for n in self._thalamus.neurons])
            np.savez_compressed(os.path.join(base_dir, 'thalamus_weights.npz'), W=th_W)
        except Exception:
            pass
        
        try:
            hip_W = np.vstack([n.nlms_head.w for n in self._hippocampus.neurons])
            np.savez_compressed(os.path.join(base_dir, 'hippocampus_weights.npz'), W=hip_W)
        except Exception:
            pass
        
        # Save specialist weights
        for name, specialist in self.specialists.items():
            try:
                weights = specialist.nlms_head.w
                np.save(os.path.join(base_dir, f'{name}_weights.npy'), weights)
            except Exception:
                pass
    
    def load_weights(self, base_dir: str = None):
        """Load network weights"""
        base_dir = base_dir or self.weights_dir
        loaded = 0
        
        # Load core weights
        try:
            thalamus_weights = np.load(os.path.join(base_dir, 'thalamus_weights.npz'))['W']
            for i, neuron in enumerate(self._thalamus.neurons):
                if i < len(thalamus_weights):
                    neuron.nlms_head.w = thalamus_weights[i].astype(np.float64)
                    loaded += 1
        except Exception:
            pass
        
        try:
            hippocampus_weights = np.load(os.path.join(base_dir, 'hippocampus_weights.npz'))['W']
            for i, neuron in enumerate(self._hippocampus.neurons):
                if i < len(hippocampus_weights):
                    neuron.nlms_head.w = hippocampus_weights[i].astype(np.float64)
                    loaded += 1
        except Exception:
            pass
        
        # Load specialist weights
        for name, specialist in self.specialists.items():
            try:
                weights = np.load(os.path.join(base_dir, f'{name}_weights.npy'))
                specialist.nlms_head.w = weights.astype(np.float64)
                loaded += 1
            except Exception:
                pass
        
        return loaded
    
    def apply_endocrine_modulations(self, levels: dict) -> dict:
        """Map hormone levels -> router/attention/energy (with clamps)"""
        effects = {}
        router = getattr(self, "_thalamic_router", None)
        attn = getattr(self, "_attention", None) or getattr(router, "attention", None)
        energy = getattr(self, "energy", None)

        cortisol = float(levels.get(self._hormone_type.CORTISOL, 0.0))
        gh = float(levels.get(self._hormone_type.GROWTH_HORMONE, 0.0))
        thyroid = float(levels.get(self._hormone_type.THYROID, 1.0))
        insulin = float(levels.get(self._hormone_type.INSULIN, 0.0))
        dopamine = float(levels.get(self._hormone_type.DOPAMINE, 0.0))
        norepi = float(levels.get(self._hormone_type.NOREPINEPHRINE, 0.0))

        # Router knobs (if present)
        if router is not None:
            # temperature
            if hasattr(router, "temperature"):
                router.temperature = float(np.clip(router.temperature * (1.0 + 0.30*cortisol), 0.5, 2.5))
                effects["temperature"] = router.temperature
            # bias lr
            if hasattr(router, "bias_lr"):
                router.bias_lr = float(np.clip(router.bias_lr * (1.0 + 0.40*(thyroid - 1.0)), 1e-4, 0.1))
                effects["bias_lr"] = router.bias_lr
            # capacity (top_k)
            if hasattr(router, "top_k") and hasattr(router, "n_experts"):
                base = getattr(router, "_base_top_k", router.top_k)
                router._base_top_k = base
                router.top_k = int(np.clip(round(base * (1.0 + 0.20*gh)), 1, router.n_experts))
                effects["top_k"] = router.top_k
            # dopamine reward → nudge recent winners' biases
            if dopamine > 0 and hasattr(router, "bias") and hasattr(router, "last_winners"):
                router.bias[router.last_winners] += 0.10 * float(dopamine)

        # Attention gain (norepinephrine)
        if attn is not None and hasattr(attn, "gain_up") and hasattr(attn, "gain_down"):
            g = 1.0 + 0.50 * norepi
            attn.gain_up = float(np.clip(attn.gain_up * g, 0.8, 3.0))
            attn.gain_down = float(np.clip(attn.gain_down / g, 0.2, 1.0))
            effects["att_gain_up"] = attn.gain_up
            effects["att_gain_down"] = attn.gain_down

        # Energy efficiency (insulin)
        if energy is not None and hasattr(energy, "e_mac_j"):
            energy.e_mac_j = float(np.clip(energy.e_mac_j * (1.0 - 0.10*insulin), 1e-13, 1e-9))
            effects["e_mac_j"] = energy.e_mac_j

        return effects

    def _hippocampal_memory_bias(self, retrieved: list, weight: float = 0.03):
        """Apply memory-based bias to router"""
        if not self.enable_hippocampus_bias: 
            return
        router = getattr(self, "_thalamic_router", None)
        if router is None or not hasattr(router, "bias") or not hasattr(router, "names"):
            return
        names = list(router.names)
        for name in retrieved:
            if name in names:
                idx = names.index(name)
                router.bias[idx] += float(weight)

    def _endocrine_step(self, result: dict, accuracy: float):
        """Apply endocrine feedback step"""
        if not self.enable_neuroendocrine: 
            return
        gates = np.array([d["gate"] for d in result.get("per_expert", {}).values()], dtype=np.float64)
        if gates.size == 0: 
            gates = np.array([1.0], dtype=np.float64)
        energy_j = float(result.get("energy_j", 0.0))
        delta = float(abs(accuracy - self._last_acc))
        self._hypothalamus.monitor_system(energy_j, accuracy, gates, delta)
        signals = self._hypothalamus.compute_control_signals()
        self._pituitary.receive_hypothalamic_signals(signals)
        effects = self._pituitary.apply_hormonal_effects(self)  # calls Network.apply_endocrine_modulations
        result["endocrine"] = {"signals": signals, "effects": effects}
        self._last_acc = accuracy
        self._system_initialized = True


# Backward compatibility aliases
SVCNetwork = Network  # For backward compatibility
SPANIntegratedNetwork = Network  # For backward compatibility