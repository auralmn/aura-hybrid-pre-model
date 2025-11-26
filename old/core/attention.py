"""
Multi-Channel Spiking Attention for ThalamicConversationRouter
Clean, drop-in attention upgrade using μ scaling
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

def prosody_channels_from_text(tokens: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crude per-token proxies:
      amplitude ~ ALLCAPS / exclamation
      pitch     ~ ? / emotive words / emojis
      boundary  ~ punctuation boundary marks
    """
    T = len(tokens)
    amp = np.zeros(T, dtype=np.float64)
    pitch = np.zeros(T, dtype=np.float64)
    boundary = np.zeros(T, dtype=np.float64)
    
    emotive = {"wow", "amazing", "terrible", "great", "awful", "love", "hate", "yay", "ugh", 
               "awesome", "horrible", "fantastic", "incredible", "disgusting", "beautiful", "ugly"}
    
    for i, w in enumerate(tokens):
        base = w.strip()
        amp[i] = 1.0 if (base.isupper() and len(base) > 2) or "!" in base else 0.0
        pitch[i] = 1.0 if ("?" in base or base.lower() in emotive or 
                          any(ch in base for ch in "😊😂😭😡🤯🎉🔥💯")) else 0.0
        boundary[i] = 1.0 if any(ch in base for ch in ".!?;,:") else 0.0
    
    return amp, pitch, boundary

@dataclass
class MultiChannelSpikingAttention:
    """Fuse amplitude/pitch/boundary spike trains into k-WTA attention gains."""
    k_winners: int = 5

    # LIF params per channel
    decay_amp: float = 0.7
    theta_amp: float = 1.0
    decay_pitch: float = 0.7
    theta_pitch: float = 1.0
    decay_bound: float = 0.7
    theta_bound: float = 1.0

    # fusion weights
    w_amp: float = 1.0
    w_pitch: float = 1.2
    w_bound: float = 0.8

    # LR multipliers
    gain_up: float = 1.8
    gain_down: float = 0.6
    min_gain: float = 0.2
    max_gain: float = 3.0

    smoothing: int = 0                 # 0/1 off
    normalize_salience: bool = True

    def _lif(self, x: np.ndarray, decay: float, theta: float) -> np.ndarray:
        """Leaky integrate-and-fire"""
        v = 0.0
        out = np.zeros_like(x, dtype=np.float64)
        for i, xi in enumerate(np.asarray(x, dtype=np.float64)):
            v = decay * v + xi
            if v >= theta:
                out[i] = 1.0
                v -= theta
        return out

    def compute(
        self,
        token_ids: List[int],
        amp: np.ndarray,
        pitch: np.ndarray,
        boundary: np.ndarray,
        *,
        feature_size: Optional[int] = None,
        token_to_feature: Optional[Dict[int, int]] = None,
    ) -> Dict[str, object]:
        """Compute multi-channel spiking attention gains"""
        T = len(token_ids)
        assert amp.shape[0] == T and pitch.shape[0] == T and boundary.shape[0] == T, "Channel length mismatch."

        # Generate spikes per channel
        s_amp = self._lif(amp, self.decay_amp, self.theta_amp)
        s_pitch = self._lif(pitch, self.decay_pitch, self.theta_pitch)
        s_bound = self._lif(boundary, self.decay_bound, self.theta_bound)

        # Fuse to salience
        sal = self.w_amp * s_amp + self.w_pitch * s_pitch + self.w_bound * s_bound

        # Optional smoothing
        if self.smoothing and self.smoothing > 1:
            k = int(self.smoothing)
            kernel = np.ones(k, dtype=np.float64) / k
            sal = np.convolve(sal, kernel, mode="same")

        # Normalize salience
        if self.normalize_salience and len(sal) > 0:
            m = float(np.max(sal))
            if m > 0:
                sal = sal / m

        # k-WTA selection
        order = np.argsort(-sal)
        winners_idx = [int(i) for i in order[:max(1, self.k_winners)] if sal[int(i)] > 0]
        seen_idx = np.nonzero(sal > 0)[0].tolist()

        # Scalar μ multiplier with gentle clamping
        mu_scalar = 1.0
        if winners_idx:
            raw_gain = np.mean(sal[winners_idx]) * self.gain_up
            mu_scalar = float(np.clip(raw_gain, 0.95, 1.12))  # Gentle clamping as suggested

        # Optional per-feature gains
        per_feat = None
        if feature_size is not None and token_to_feature is not None:
            per_feat = np.ones(feature_size, dtype=np.float64)
            # Depress non-winners
            for pos in seen_idx:
                tok = token_ids[pos]
                j = token_to_feature.get(tok)
                if j is not None and 0 <= j < feature_size:
                    per_feat[j] = self.gain_down
            # Boost winners
            for pos in winners_idx:
                tok = token_ids[pos]
                j = token_to_feature.get(tok)
                if j is not None and 0 <= j < feature_size:
                    per_feat[j] = self.gain_up
            per_feat = np.clip(per_feat, self.min_gain, self.max_gain)

        return {
            "mu_scalar": mu_scalar,
            "per_feature_gains": per_feat,
            "salience": sal,
            "spikes": {"amp": s_amp, "pitch": s_pitch, "boundary": s_bound},
            "winners_idx": winners_idx,
        }

    def reset_state(self):
        """Reset LIF state for fresh processing"""
        pass  # Stateless version

# Preset configurations for different use cases
class RouterAttentionPresets:
    """Pre-configured attention settings for router applications"""
    
    @staticmethod
    def conversational() -> MultiChannelSpikingAttention:
        """For general conversational routing"""
        return MultiChannelSpikingAttention(
            k_winners=5,
            w_amp=1.0, w_pitch=1.2, w_bound=0.6,  # Reduced boundary weight
            decay_amp=0.7, theta_amp=1.0,
            decay_pitch=0.7, theta_pitch=1.0,
            decay_bound=0.7, theta_bound=1.3,  # Higher threshold for boundary
            gain_up=1.8, gain_down=0.6,
            smoothing=1
        )
    
    @staticmethod
    def technical() -> MultiChannelSpikingAttention:
        """For technical/analytical routing"""
        return MultiChannelSpikingAttention(
            k_winners=3,
            w_amp=0.8, w_pitch=1.0, w_bound=1.2,
            decay_amp=0.8, theta_amp=1.2,
            decay_pitch=0.6, theta_pitch=0.8,
            decay_bound=0.8, theta_bound=1.2,
            gain_up=1.5, gain_down=0.7,
            smoothing=2
        )
    
    @staticmethod
    def emotional() -> MultiChannelSpikingAttention:
        """For emotional/sentiment routing"""
        return MultiChannelSpikingAttention(
            k_winners=6,
            w_amp=1.2, w_pitch=1.5, w_bound=0.6,
            decay_amp=0.6, theta_amp=0.8,
            decay_pitch=0.6, theta_pitch=0.8,
            decay_bound=0.7, theta_bound=1.0,
            gain_up=2.0, gain_down=0.4,
            smoothing=0
        )
    
    @staticmethod
    def streaming() -> MultiChannelSpikingAttention:
        """For real-time streaming routing"""
        return MultiChannelSpikingAttention(
            k_winners=4,
            w_amp=1.0, w_pitch=1.0, w_bound=1.0,
            decay_amp=0.7, theta_amp=1.0,
            decay_pitch=0.7, theta_pitch=1.0,
            decay_bound=0.7, theta_bound=1.0,
            gain_up=1.6, gain_down=0.5,
            smoothing=0,
            normalize_salience=False
        )
