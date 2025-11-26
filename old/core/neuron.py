from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List
import numpy as np
from .neuronal_state import NeuronalState
from .nlms import NLMSHead
from .phasor import PhasorBank, PhasorState

class MaturationStage(Enum):
    PROGENITOR = auto()
    MIGRATING = auto()
    DIFFERENTIATED = auto()
    MYELINATED = auto()

class ActivityState(Enum):
    RESTING = auto()
    FIRING = auto()
    REFRACTORY = auto()

class Synapse:
    def __init__(self, target, weight, plasticity=0.0):
        self.target = target         # Target neuron reference
        self.weight = weight         # Synaptic strength
        self.plasticity = plasticity # modulation factor

class NeuronCoordinates():
    x: float
    y: float
    z: float
    t: datetime
    universe: str
    state: Any
    

class Neuron:
    def __init__(self, neuron_id, specialization, abilities,
             n_features, n_outputs, maturation=MaturationStage.PROGENITOR, 
             activity=ActivityState.RESTING, position: NeuronCoordinates | None = None,
             # NEW: Add attention parameters
             enable_attention: bool = False,
             attention_config: Dict | None = None):
        self.neuron_id = neuron_id
        self.specialization = specialization   # e.g., 'place_cell', 'interneuron'
        self.abilities = abilities            # e.g., {'memory': 0.9, 'speed': 0.7}
        self.maturation = maturation
        self.activity = activity
        self.phasor = PhasorState(delta0=7)
        self.phasor_bank = PhasorBank(delta0=7, H=384)
        self.neuronal_state = NeuronalState(
            kind=specialization,
            position=position,
            gene_expression={'Neurogenin': 0.4},
            connections=[],
            environment={},
        )

        self.membrane_potential = 0.0
        self.refractory_timer = 0
        self.weights = []       # Synaptic weights
        self.synapses = []      # List of Synapse
        self.spike_history = [] # [(timestamp, value)]
        self.gene_expression = {}  # Track lineage factors
        # NEW: Configure NLMS with attention
        self.nlms_head = NLMSHead(
        n_features=n_features,
        n_outputs=n_outputs,
        tok_slice=slice(0, n_features),  # Learn on all features by default
        realm_slice=slice(0, 0),
        phase_slice=slice(0, 0),
        seed=27,
        # NEW: Add attention parameters
        enable_attention=enable_attention,
        attention_config=attention_config
    )
    
        # NEW: Attention-related attributes
        self.enable_attention = enable_attention
        self.attention_stats = {
            'total_sequences_processed': 0,
            'attention_modulated_steps': 0,
            'average_attention_gain': 1.0
        }
        

    async def attach(self):
        await self.nlms_head.attach(
        np.zeros(self.nlms_head.n_features, dtype=np.float64),
        slice(0, self.nlms_head.n_features),   # tok_slice
        slice(0, 0),                           # realm_slice
        slice(0, 0)                            # phase_slice
    ) 

    def connect(self, target_neuron, weight, plasticity=0.1):
        syn = Synapse(target_neuron, weight, plasticity)
        self.synapses.append(syn)
        self.weights.append(weight)

    def advance_stage(self, signals, gene_expression):
        if self.maturation == MaturationStage.PROGENITOR and signals.get('Wnt', 0) > 0.7:
            self.maturation = MaturationStage.MIGRATING
        elif self.maturation == MaturationStage.MIGRATING and gene_expression.get('Neurogenin', 0) > 0.8:
            self.maturation = MaturationStage.DIFFERENTIATED
        elif self.maturation == MaturationStage.DIFFERENTIATED and signals.get('FGF2', 0) > 0.5:
            self.maturation = MaturationStage.MYELINATED

    def update_activity(self, input_current, time):
        # Accept scalar or vector input; reduce vector to scalar drive
        if isinstance(input_current, np.ndarray):
            try:
                input_current = float(np.mean(input_current))
            except Exception:
                input_current = float(0.0)
        if self.activity == ActivityState.REFRACTORY:
            self.refractory_timer -= 1
            if self.refractory_timer <= 0:
                self.activity = ActivityState.RESTING
        else:
            self.membrane_potential += input_current
            if self.membrane_potential > 1.0:
                self.activity = ActivityState.FIRING
                self.spike_history.append((time, 1))
                self.membrane_potential = 0.0
                self.activity = ActivityState.REFRACTORY
                self.refractory_timer = 3
            else:
                self.activity = ActivityState.RESTING
                self.spike_history.append((time, 0))

    def specialize(self, specialization, abilities, metabolic_eps=1e-3):
        self.specialization = specialization
        self.abilities = abilities
        self.nlms_head.mu = min(self.nlms_head.mu, metabolic_eps)


    def transmit_spike(self, time):
        if self.activity == ActivityState.FIRING:
            for syn in self.synapses:
                syn.target.update_activity(syn.weight, time)

    async def update_nlms(self, x: np.ndarray, y_true: float):
        # Update NLMSHead with current features and target label
        y_pred = await self.nlms_head.step(x, y_true) 
        return y_pred

    def get_readout(self, X: np.ndarray):
        # Get current prediction from the NLMSHead with safe vector handling
        x = np.asarray(X, dtype=np.float64).reshape(-1)
        n = self.nlms_head.n_features
        if x.size < n:
            x = np.pad(x, (0, n - x.size))
        elif x.size > n:
            x = x[:n]
        return self.nlms_head.predict(x)
    
    def get_features(self, input_signal):
        # Generate multi-harmonic leaky resonator features and map to 384 dims
        feats = self.phasor_bank.step(input_signal)
        n = self.nlms_head.n_features
        if feats.shape[0] >= n:
            return feats[:n]
        return np.pad(feats, (0, n - feats.shape[0]))
    
    def tokenize_content(self, text: str, max_length: int = 512) -> List[int]:
        """Simple tokenization for attention processing"""
        words = text.lower().split()[:max_length]
        return [hash(word) % 50000 for word in words]
    
    async def update_nlms_with_attention(self, x: np.ndarray, y_true: float, 
                                   content_text: str = "") -> float:
        """Update NLMS with attention if enabled"""
        
        if self.enable_attention and content_text:
            token_sequence = self.tokenize_content(content_text)
            self.attention_stats['total_sequences_processed'] += 1
            self.attention_stats['attention_modulated_steps'] += 1
            
            return await self.nlms_head.step_with_attention(x, y_true, token_sequence)
        else:
            return await self.nlms_head.step(x, y_true)

    async def update_nlms_with_multi_channel_attention(
        self, 
        x: np.ndarray, 
        y_true: float,
        content_text: str = "",
        multi_channel_attention = None
    ) -> float:
        """Update NLMS with multi-channel spiking attention if enabled"""
        
        if self.enable_attention and content_text and multi_channel_attention:
            # Tokenize content
            tokens = content_text.lower().split()
            token_ids = [hash(token) % 50000 for token in tokens]
            
            # Extract prosody channels
            from .multi_channel_attention import prosody_channels_from_text, build_token_to_feature_mapping
            amp, pitch, boundary = prosody_channels_from_text(tokens)
            
            # Build token-to-feature mapping
            feature_size = self.nlms_head.tok_slice.stop - self.nlms_head.tok_slice.start
            token_to_feature = build_token_to_feature_mapping(tokens, feature_size)
            
            # Update attention stats
            self.attention_stats['total_sequences_processed'] += 1
            self.attention_stats['attention_modulated_steps'] += 1
            
            return await self.nlms_head.step_with_multi_channel_attention(
                x, y_true, token_ids, amp, pitch, boundary, 
                multi_channel_attention, token_to_feature
            )
        else:
            return await self.nlms_head.step(x, y_true)

# Example usage:
