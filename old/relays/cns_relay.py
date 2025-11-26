from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class CentralNervousSystemModule:
    input_dim: int = 384

    def suggest_attention_mode(self, context: Dict[str, Any]) -> str:
        if context.get('urgency', 0.0) > 0.7:
            return 'FOCUSED'
        if context.get('multi_topic', False):
            return 'DIVIDED'
        if context.get('filter_needed', False):
            return 'SELECTIVE'
        return 'DIFFUSE'

    def adjust_arousal(self, base: float, threat: float | None = None) -> float:
        if threat is None:
            return float(np.clip(base, 0.0, 1.0))
        # Nudge arousal toward threat level
        return float(np.clip(0.7 * base + 0.3 * threat, 0.0, 1.0))

    def allocate_resources(self, regions: List[str], priorities: Dict[str, float]) -> Dict[str, float]:
        # Softmax allocation by priority
        xs = np.array([priorities.get(r, 0.5) for r in regions], dtype=np.float64)
        if xs.size == 0:
            return {}
        xs = xs - xs.max()
        e = np.exp(xs)
        e = e / (e.sum() + 1e-8)
        return {r: float(e[i]) for i, r in enumerate(regions)}

