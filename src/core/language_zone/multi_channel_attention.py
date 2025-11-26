"""
Multi-Channel Spiking Attention System
Fuses amplitude/pitch/boundary spike streams into k-WTA attention gains
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class MultiChannelSpikingAttention:
    """Fuse amplitude/pitch/boundary spike trains into k-WTA attention gains."""
    k_winners: int = 5

    # Per-channel leaky integrate-and-fire (LIF) params
    decay_amp: float = 0.7
    theta_amp: float = 1.0
    decay_pitch: float = 0.7
    theta_pitch: float = 1.0
    decay_bound: float = 0.7
    theta_bound: float = 1.0

    # Channel fusion weights
    w_amp: float = 1.0
    w_pitch: float = 1.0
    w_bound: float = 1.0

    # Learning-rate multipliers
    gain_up: float = 1.8     # winners
    gain_down: float = 0.6   # non-winners that spiked
    min_gain: float = 0.5    # Raised floor to prevent over-suppression
    max_gain: float = 2.5    # Widened ceiling for high prosody range

    # Post-fusion tweaks
    smoothing: int = 0                 # moving average window on salience (0/1 = off)
    normalize_salience: bool = True    # scale salience to [0,1] by max

    # Stateful LIF for streaming (optional)
    v_amp: float = 0.0
    v_pitch: float = 0.0
    v_bound: float = 0.0

    def _lif(self, x: np.ndarray, decay: float, theta: float) -> np.ndarray:
        """Leaky integrate-and-fire (stateless version)"""
        v = 0.0
        out = np.zeros_like(x, dtype=np.float64)
        for i, xi in enumerate(np.asarray(x, dtype=np.float64)):
            v = decay * v + xi
            if v >= theta:
                out[i] = 1.0
                v -= theta  # soft reset
        return out

    def _lif_stateful(self, x: np.ndarray, v0: float, decay: float, theta: float) -> Tuple[np.ndarray, float]:
        """Leaky integrate-and-fire (stateful version for streaming)"""
        v = v0
        out = np.zeros_like(x, dtype=np.float64)
        for i, xi in enumerate(np.asarray(x, dtype=np.float64)):
            v = decay * v + xi
            if v >= theta:
                out[i] = 1.0
                v -= theta  # soft reset
        return out, v

    def compute(
        self,
        token_ids: List[int],
        amp: np.ndarray,
        pitch: np.ndarray,
        boundary: np.ndarray,
        *,
        feature_size: Optional[int] = None,
        token_to_feature: Optional[Dict[int, int]] = None,
        use_stateful: bool = False,
    ) -> Dict[str, object]:
        """
        Compute multi-channel spiking attention gains.
        
        Args:
            token_ids: length T list of token ids (aligned with channels)
            amp/pitch/boundary: R^T channel intensities (already extracted)
            feature_size: if provided with token_to_feature, returns per-feature gains
            token_to_feature: maps a token id -> feature index in your vectorizer space
            use_stateful: use stateful LIF for streaming applications
        
        Returns:
            Dict containing:
            - mu_scalar: scalar multiplier for token LR
            - per_feature_gains: optional vector gains for tok slice
            - salience: per-token salience
            - spikes: per-channel spike trains
            - winners_idx: token positions that won WTA
        """
        T = len(token_ids)
        assert amp.shape[0] == T and pitch.shape[0] == T and boundary.shape[0] == T, "Channel length mismatch."

        # 1) spikes per channel (LIF)
        if use_stateful:
            s_amp, self.v_amp = self._lif_stateful(amp, self.v_amp, self.decay_amp, self.theta_amp)
            s_pitch, self.v_pitch = self._lif_stateful(pitch, self.v_pitch, self.decay_pitch, self.theta_pitch)
            s_bound, self.v_bound = self._lif_stateful(boundary, self.v_bound, self.decay_bound, self.theta_bound)
        else:
            s_amp = self._lif(amp, self.decay_amp, self.theta_amp)
            s_pitch = self._lif(pitch, self.decay_pitch, self.theta_pitch)
            s_bound = self._lif(boundary, self.decay_bound, self.theta_bound)

        # 2) fuse to salience
        sal = self.w_amp * s_amp + self.w_pitch * s_pitch + self.w_bound * s_bound

        # optional smoothing
        if self.smoothing and self.smoothing > 1:
            k = int(self.smoothing)
            kernel = np.ones(k, dtype=np.float64) / k
            sal = np.convolve(sal, kernel, mode="same")

        # normalize to [0,1] for robust winner scaling
        if self.normalize_salience and len(sal) > 0:
            m = float(np.max(sal))
            if m > 0:
                sal = sal / m

        # 3) k-WTA over token positions
        order = np.argsort(-sal)
        winners_idx = [int(i) for i in order[:max(1, self.k_winners)] if sal[int(i)] > 0]
        seen_idx = np.nonzero(sal > 0)[0].tolist()

        # 4) μ scalar (fallback when no per-feature mapping)
        # TANH-BASED GAIN FORMULA for better separation:
        # - Low prosody (sal≈0): μ ≈ 0.5
        # - Medium prosody (sal≈0.5): μ ≈ 1.5
        # - High prosody (sal≈1.0): μ ≈ 2.3 (asymptotic to max_gain)
        mu_scalar = 1.0
        if winners_idx:
            avg_winner_salience = float(np.mean(sal[winners_idx]))
            
            # Nonlinear mapping using tanh for smooth saturation
            # tanh compresses high values gracefully without hard clipping
            gain_range = self.max_gain - self.min_gain
            mu_scalar = float(
                self.min_gain + gain_range * np.tanh(self.gain_up * avg_winner_salience)
            )
            
            # Ensure within bounds (tanh ensures this naturally, but be explicit)
            mu_scalar = float(np.clip(mu_scalar, self.min_gain, self.max_gain))

        # 5) optional per-feature gains (sparse emphasis)
        per_feat = None
        if feature_size is not None and token_to_feature is not None:
            per_feat = np.ones(feature_size, dtype=np.float64)
            # depress all 'seen' non-winners
            for pos in seen_idx:
                tok = token_ids[pos]
                j = token_to_feature.get(tok)
                if j is not None and 0 <= j < feature_size:
                    per_feat[j] = self.gain_down
            # boost winners
            for pos in winners_idx:
                tok = token_ids[pos]
                j = token_to_feature.get(tok)
                if j is not None and 0 <= j < feature_size:
                    per_feat[j] = self.gain_up
            per_feat = np.clip(per_feat, self.min_gain, self.max_gain)

        return {
            "mu_scalar": mu_scalar,                 # scalar multiplier for token LR
            "per_feature_gains": per_feat,          # optional vector gains for tok slice
            "salience": sal,                        # per-token salience
            "spikes": {"amp": s_amp, "pitch": s_pitch, "boundary": s_bound},
            "winners_idx": winners_idx,             # token positions that won WTA
        }

    def reset_state(self):
        """Reset LIF state for fresh streaming session"""
        self.v_amp = 0.0
        self.v_pitch = 0.0
        self.v_bound = 0.0


def prosody_channels_from_text(tokens: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract crude prosody channels from text tokens.
    
    Args:
        tokens: List of token strings
        
    Returns:
        Tuple of (amplitude, pitch, boundary) arrays
    """
    T = len(tokens)
    amp = np.zeros(T, dtype=np.float64)
    pitch = np.zeros(T, dtype=np.float64)
    bound = np.zeros(T, dtype=np.float64)
    
    emotive = {"wow", "amazing", "terrible", "great", "awful", "love", "hate", "yay", "ugh", 
               "fantastic", "horrible", "incredible", "disgusting", "beautiful", "ugly"}
    
    for i, w in enumerate(tokens):
        base = w.strip()
        
        # Amplitude: ALLCAPS/exclamation density
        amp[i] = 1.0 if (base.isupper() and len(base) > 2) or "!" in base else 0.0
        
        # Pitch: question marks / emojis / emotive words
        pitch[i] = 1.0 if ("?" in base or base.lower() in emotive or 
                          any(ch in base for ch in "😊😂😭😡🎉🔥💯")) else 0.0
        
        # Boundary: punctuation boundary markers
        bound[i] = 1.0 if any(ch in base for ch in ".!?;,:") else 0.0
    
    return amp, pitch, bound


def build_token_to_feature_mapping(
    tokens: List[str], 
    feature_size: int,
    hash_mod: int = 50000
) -> Dict[int, int]:
    """
    Build a mapping from token hashes to feature indices.
    
    Args:
        tokens: List of token strings
        feature_size: Size of feature vector
        hash_mod: Modulo for token hashing
        
    Returns:
        Dict mapping token_hash -> feature_index
    """
    token_to_feature = {}
    for i, token in enumerate(tokens):
        token_hash = hash(token) % hash_mod
        feature_idx = i % feature_size  # Simple round-robin mapping
        token_to_feature[token_hash] = feature_idx
    
    return token_to_feature


# Example usage and configuration presets
class AttentionPresets:
    """Pre-configured attention settings for different use cases"""
    
    @staticmethod
    def analytical() -> MultiChannelSpikingAttention:
        """For analytical/linguistic processing"""
        return MultiChannelSpikingAttention(
            k_winners=3,
            w_amp=0.8, w_pitch=1.2, w_bound=1.0,
            gain_up=1.5, gain_down=0.7,
            smoothing=2
        )
    
    @staticmethod
    def emotional() -> MultiChannelSpikingAttention:
        """For emotional/sentiment processing"""
        return MultiChannelSpikingAttention(
            k_winners=5,
            w_amp=1.2, w_pitch=1.5, w_bound=0.6,
            gain_up=2.0, gain_down=0.4,
            smoothing=1
        )
    
    @staticmethod
    def historical() -> MultiChannelSpikingAttention:
        """For historical/temporal processing"""
        return MultiChannelSpikingAttention(
            k_winners=4,
            w_amp=1.0, w_pitch=1.0, w_bound=1.3,
            gain_up=1.8, gain_down=0.6,
            smoothing=3
        )
    
    @staticmethod
    def streaming() -> MultiChannelSpikingAttention:
        """For streaming/real-time processing"""
        return MultiChannelSpikingAttention(
            k_winners=6,
            w_amp=1.0, w_pitch=1.0, w_bound=1.0,
            gain_up=1.6, gain_down=0.5,
            smoothing=0,  # No smoothing for real-time
            normalize_salience=False  # Keep raw salience for streaming
        )
