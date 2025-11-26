import numpy as np
from typing import Any, Dict, Optional


class AmygdalaModule:
    def __init__(self, input_dim: int = 384, threat_threshold: float = 0.7, emotional_sensitivity: float = 0.8):
        self.input_dim = input_dim
        self.threat_threshold = float(threat_threshold)
        self.emotional_sensitivity = float(emotional_sensitivity)

    def assess_threat(self, x: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Simple heuristic threat assessment: magnitude + context keywords
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        mag = float(np.linalg.norm(x) / max(1.0, len(x)))
        threat = mag

        reason = []
        if context:
            cats = [c.lower() for c in context.get('categories', [])]
            high = {'war', 'revolution', 'collapse', 'invasion', 'persecution', 'famine', 'plague'}
            if any(c in high for c in cats):
                threat = min(1.0, threat + 0.25)
                reason.append('high_risk_category')
            tone = str(context.get('tone', '')).lower()
            if any(k in tone for k in ['hostile', 'fear', 'panic', 'riot']):
                threat = min(1.0, threat + 0.15)
                reason.append('hostile_tone')
            impact = str(context.get('impact', '')).lower()
            if any(k in impact for k in ['catastrophic','disaster','crisis']):
                threat = min(1.0, threat + 0.2)
                reason.append('impact_keyword')

        return {
            'threat_score': float(threat),
            'above_threshold': bool(threat >= self.threat_threshold),
            'reasons': reason,
        }

