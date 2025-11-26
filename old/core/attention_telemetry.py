"""
Attention Telemetry System
Provides detailed logging and monitoring of attention events
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable
import time
import numpy as np


@dataclass
class AttentionEvent:
    """Single attention event with all relevant metrics"""
    t: float
    text_len: int
    mu_scalar: float
    mu_applied_ratio: float
    winners_idx: List[int]
    salience_mean: float
    salience_max: float
    salience_std: float
    spike_rate_amp: float
    spike_rate_pitch: float
    spike_rate_boundary: float
    primary: Optional[str] = None
    routing_confidence: Optional[float] = None
    routing_success: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable string representation"""
        return (f"AttentionEvent(t={self.t:.3f}, μ={self.mu_scalar:.2f}, "
                f"winners={len(self.winners_idx)}, sal_μ={self.salience_mean:.2f}, "
                f"spikes=({self.spike_rate_amp:.2f},{self.spike_rate_pitch:.2f},{self.spike_rate_boundary:.2f}))")


class AttentionTelemetryBuffer:
    """Circular buffer for attention events with summary statistics"""
    
    def __init__(self, maxlen: int = 1000):
        self.maxlen = int(maxlen)
        self._events: List[AttentionEvent] = []

    def push(self, ev: AttentionEvent) -> None:
        """Add a new attention event"""
        self._events.append(ev)
        if len(self._events) > self.maxlen:
            del self._events[: len(self._events) - self.maxlen]

    def recent(self, n: int = 50) -> List[AttentionEvent]:
        """Get the most recent n events"""
        return self._events[-n:]

    def recent_dicts(self, n: int = 50) -> List[Dict]:
        """Get the most recent n events as dictionaries"""
        return [e.to_dict() for e in self.recent(n)]

    def summary(self) -> Dict[str, float]:
        """Compute summary statistics for all events"""
        if not self._events:
            return {}
        
        # Extract arrays for statistical computation
        mu = np.array([e.mu_scalar for e in self._events], dtype=np.float64)
        mratio = np.array([e.mu_applied_ratio for e in self._events], dtype=np.float64)
        smax = np.array([e.salience_max for e in self._events], dtype=np.float64)
        smean = np.array([e.salience_mean for e in self._events], dtype=np.float64)
        sstd = np.array([e.salience_std for e in self._events], dtype=np.float64)
        ra = np.array([e.spike_rate_amp for e in self._events], dtype=np.float64)
        rp = np.array([e.spike_rate_pitch for e in self._events], dtype=np.float64)
        rb = np.array([e.spike_rate_boundary for e in self._events], dtype=np.float64)
        
        return {
            "mu_scalar_mean": float(mu.mean()),
            "mu_scalar_std": float(mu.std()),
            "mu_scalar_p95": float(np.percentile(mu, 95)),
            "mu_scalar_p99": float(np.percentile(mu, 99)),
            "mu_applied_ratio_mean": float(mratio.mean()),
            "mu_applied_ratio_std": float(mratio.std()),
            "salience_max_mean": float(smax.mean()),
            "salience_max_std": float(smax.std()),
            "salience_mean_mean": float(smean.mean()),
            "salience_mean_std": float(smean.std()),
            "salience_std_mean": float(sstd.mean()),
            "salience_std_std": float(sstd.std()),
            "spike_rate_amp_mean": float(ra.mean()),
            "spike_rate_amp_std": float(ra.std()),
            "spike_rate_pitch_mean": float(rp.mean()),
            "spike_rate_pitch_std": float(rp.std()),
            "spike_rate_boundary_mean": float(rb.mean()),
            "spike_rate_boundary_std": float(rb.std()),
            "n_events": len(self._events),
            "avg_text_len": float(np.mean([e.text_len for e in self._events])),
            "avg_winners": float(np.mean([len(e.winners_idx) for e in self._events])),
        }

    def clear(self) -> None:
        """Clear all events"""
        self._events.clear()

    def get_events_by_primary(self, primary: str) -> List[AttentionEvent]:
        """Get events filtered by primary specialist"""
        return [e for e in self._events if e.primary == primary]

    def get_events_by_note(self, note: str) -> List[AttentionEvent]:
        """Get events filtered by note"""
        return [e for e in self._events if e.note == note]

    def get_high_attention_events(self, threshold: float = 2.0) -> List[AttentionEvent]:
        """Get events with high attention (mu_scalar > threshold)"""
        return [e for e in self._events if e.mu_scalar > threshold]

    def get_spike_analysis(self) -> Dict[str, float]:
        """Analyze spike patterns across all events"""
        if not self._events:
            return {}
        
        ra = np.array([e.spike_rate_amp for e in self._events], dtype=np.float64)
        rp = np.array([e.spike_rate_pitch for e in self._events], dtype=np.float64)
        rb = np.array([e.spike_rate_boundary for e in self._events], dtype=np.float64)
        
        # Calculate correlations
        corr_amp_pitch = float(np.corrcoef(ra, rp)[0, 1]) if len(ra) > 1 else 0.0
        corr_amp_boundary = float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 1 else 0.0
        corr_pitch_boundary = float(np.corrcoef(rp, rb)[0, 1]) if len(ra) > 1 else 0.0
        
        return {
            "correlation_amp_pitch": corr_amp_pitch,
            "correlation_amp_boundary": corr_amp_boundary,
            "correlation_pitch_boundary": corr_pitch_boundary,
            "dominant_channel": "amplitude" if ra.mean() > max(rp.mean(), rb.mean()) 
                              else "pitch" if rp.mean() > rb.mean() else "boundary",
            "spike_diversity": float(np.mean([ra.std(), rp.std(), rb.std()])),
        }


class AttentionTelemetryLogger:
    """Simple logger for attention events"""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.log_count = 0
    
    def log_event(self, event: AttentionEvent) -> None:
        """Log a single attention event"""
        self.log_count += 1
        print(f"[ATTN-{self.log_count:04d}] {event}")
    
    def log_summary(self, summary: Dict[str, float]) -> None:
        """Log attention summary statistics"""
        print(f"[ATTN-SUMMARY] Events: {summary.get('n_events', 0)}")
        print(f"  μ scalar: {summary.get('mu_scalar_mean', 0):.3f}±{summary.get('mu_scalar_std', 0):.3f} "
              f"(p95: {summary.get('mu_scalar_p95', 0):.3f})")
        print(f"  Salience: μ={summary.get('salience_mean_mean', 0):.3f}±{summary.get('salience_mean_std', 0):.3f}, "
              f"max={summary.get('salience_max_mean', 0):.3f}±{summary.get('salience_max_std', 0):.3f}")
        print(f"  Spike rates: amp={summary.get('spike_rate_amp_mean', 0):.3f}±{summary.get('spike_rate_amp_std', 0):.3f}, "
              f"pitch={summary.get('spike_rate_pitch_mean', 0):.3f}±{summary.get('spike_rate_pitch_std', 0):.3f}, "
              f"boundary={summary.get('spike_rate_boundary_mean', 0):.3f}±{summary.get('spike_rate_boundary_std', 0):.3f}")
        print(f"  Text length: {summary.get('avg_text_len', 0):.1f}, Winners: {summary.get('avg_winners', 0):.1f}")


# Example usage and utility functions
def create_attention_hook(telemetry_buffer: AttentionTelemetryBuffer, 
                         logger: Optional[AttentionTelemetryLogger] = None) -> Callable[[AttentionEvent], None]:
    """Create an attention hook that logs to both buffer and logger"""
    def hook(event: AttentionEvent) -> None:
        if logger:
            logger.log_event(event)
    return hook

def print_attention_summary(telemetry_buffer: AttentionTelemetryBuffer) -> None:
    """Print a formatted attention summary"""
    summary = telemetry_buffer.summary()
    if not summary:
        print("No attention events recorded")
        return
    
    print("🧠 Attention Telemetry Summary")
    print("=" * 50)
    print(f"Total events: {summary['n_events']}")
    print(f"Average text length: {summary['avg_text_len']:.1f} tokens")
    print(f"Average winners: {summary['avg_winners']:.1f}")
    print()
    print("μ Scalar Statistics:")
    print(f"  Mean: {summary['mu_scalar_mean']:.3f} ± {summary['mu_scalar_std']:.3f}")
    print(f"  95th percentile: {summary['mu_scalar_p95']:.3f}")
    print(f"  99th percentile: {summary['mu_scalar_p99']:.3f}")
    print()
    print("Salience Statistics:")
    print(f"  Mean: {summary['salience_mean_mean']:.3f} ± {summary['salience_mean_std']:.3f}")
    print(f"  Max: {summary['salience_max_mean']:.3f} ± {summary['salience_max_std']:.3f}")
    print()
    print("Spike Rates:")
    print(f"  Amplitude: {summary['spike_rate_amp_mean']:.3f} ± {summary['spike_rate_amp_std']:.3f}")
    print(f"  Pitch: {summary['spike_rate_pitch_mean']:.3f} ± {summary['spike_rate_pitch_std']:.3f}")
    print(f"  Boundary: {summary['spike_rate_boundary_mean']:.3f} ± {summary['spike_rate_boundary_std']:.3f}")
    
    # Spike analysis
    spike_analysis = telemetry_buffer.get_spike_analysis()
    if spike_analysis:
        print()
        print("Spike Analysis:")
        print(f"  Dominant channel: {spike_analysis['dominant_channel']}")
        print(f"  Spike diversity: {spike_analysis['spike_diversity']:.3f}")
        print(f"  Correlations: amp-pitch={spike_analysis['correlation_amp_pitch']:.3f}, "
              f"amp-boundary={spike_analysis['correlation_amp_boundary']:.3f}, "
              f"pitch-boundary={spike_analysis['correlation_pitch_boundary']:.3f}")
