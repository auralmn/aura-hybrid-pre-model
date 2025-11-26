from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ThalamicRouterModule:
    input_dim: int
    routing_threshold: float = 0.6

    def score(self, meta: Dict[str, Any]) -> float:
        """Lightweight heuristic score using metadata flags from analysis.
        Expects booleans like is_greeting, is_historical, is_emotional, etc.
        """
        w = {
            'is_greeting': 0.3,
            'is_historical': 0.8,
            'is_emotional': 0.7,
            'is_memory_query': 0.6,
            'is_analytical': 0.7,
            'is_casual': 0.4,
        }
        score = 0.0
        for k, v in w.items():
            score += (1.0 if meta.get(k) else 0.0) * v
        # Normalize to [0,1]
        return max(0.0, min(1.0, score / 3.0))

