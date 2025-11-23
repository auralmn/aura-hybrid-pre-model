import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class STDPLearner:
    """
    Spike-Timing Dependent Plasticity (STDP) learner for text tokens.
    Learns temporal associations between tokens based on "spike" timing.
    """
    def __init__(self, 
                 learning_rate_plus: float = 0.01, 
                 learning_rate_minus: float = 0.012,
                 time_window: int = 5,
                 w_min: float = 0.0,
                 w_max: float = 1.0,
                 decay: float = 0.99):
        
        self.lr_plus = learning_rate_plus
        self.lr_minus = learning_rate_minus
        self.window = time_window
        self.w_min = w_min
        self.w_max = w_max
        self.decay = decay
        
        # Weight matrix: token_id -> weight (simplified scalar weight per token for salience)
        # In a full network, this would be [pre_neuron, post_neuron]
        self.token_weights: Dict[int, float] = {}
        
        # Spike timing traces: token_id -> last_spike_time
        self.spike_traces: Dict[int, float] = {}
        self.current_time = 0.0

    def process_sequence(self, token_ids: List[int], spikes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process a sequence of tokens and update weights using STDP.
        
        Args:
            token_ids: List of token IDs in sequence
            spikes: Optional binary array indicating if token "spiked" (was salient).
                    If None, assumes all tokens in sequence "fire" in order.
        """
        if not token_ids:
            return {}
            
        # If no specific spike pattern provided, assume sequential firing
        if spikes is None:
            spikes = np.ones(len(token_ids), dtype=bool)
            
        updates = 0
        
        for t, (token, is_spike) in enumerate(zip(token_ids, spikes)):
            if not is_spike:
                continue
                
            # Current spike time
            now = self.current_time + t * 0.1  # 100ms steps
            
            # 1. LTP (Long-Term Potentiation): Strengthen association with recent past spikes
            # Check recently fired tokens (pre-synaptic)
            for prev_token, prev_time in self.spike_traces.items():
                dt = now - prev_time
                if 0 < dt < self.window:
                    # Hebbian reinforcement: "Cells that fire together, wire together"
                    # Specifically: Pre before Post -> Strengthen
                    self._update_weight(token, self.lr_plus * np.exp(-dt))
                    updates += 1
            
            # 2. LTD (Long-Term Depression): Weaken association if reverse order (not implemented in simple scalar model)
            # In this simplified scalar model, we decay weights periodically instead
            
            # Update trace for this token
            self.spike_traces[token] = now
            
        self.current_time += len(token_ids) * 0.1
        
        # Apply passive decay
        if self.current_time > 100.0:  # Occasional cleanup
            self._decay_weights()
            self.current_time = 0.0
            self.spike_traces.clear()
            
        return {"updates": updates, "active_tokens": len(self.token_weights)}

    def _update_weight(self, token: int, delta: float):
        """Update weight with bounds checking"""
        w = self.token_weights.get(token, 0.5) # Start neutral
        w += delta
        self.token_weights[token] = max(self.w_min, min(self.w_max, w))

    def _decay_weights(self):
        """Apply exponential decay to all weights"""
        for tok in list(self.token_weights.keys()):
            self.token_weights[tok] *= self.decay
            if self.token_weights[tok] < 0.01:
                del self.token_weights[tok]

    def get_modulations(self, token_ids: List[int]) -> np.ndarray:
        """Get STDP modulation factors for a sequence of tokens"""
        mods = np.ones(len(token_ids), dtype=np.float32)
        for i, tok in enumerate(token_ids):
            w = self.token_weights.get(tok, 0.0)
            # Modulation factor: 1.0 + alpha * weight
            mods[i] = 1.0 + (0.2 * w)
        return mods

    def save_state(self) -> Dict:
        return {"token_weights": self.token_weights.copy()}

    def load_state(self, state: Dict):
        self.token_weights = state.get("token_weights", {}).copy()
