from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from ..core.network import Network
from .historical_features import get_enhanced_historical_embedding, get_historical_period


def guess_context_type(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ['history','ancient','medieval','rome','empire','dynasty','war']):
        return 'historical_analysis'
    if any(k in q for k in ['feel','scared','worried','afraid','anxious','angry']):
        return 'emotion'
    if any(k in q for k in ['remember','recall','yesterday','previous']):
        return 'memory'
    return 'chat'


@dataclass
class ChatOrchestrator:
    net: Network
    offline: bool = True

    def _encode_query(self, query: str) -> np.ndarray:
        # Try SBERT if available and not offline
        if not self.offline:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                sbert = SentenceTransformer('all-MiniLM-L6-v2')
                return np.asarray(sbert.encode(query, convert_to_tensor=False), dtype=np.float32)
            except Exception:
                pass
        return np.zeros(384, dtype=np.float32)

    async def respond(self, user_query: str) -> Dict[str, Any]:
        # 1) CNS assessment
        state0 = self.net._cns.assess_global_state()

        # 2) Features
        qfeat = self._encode_query(user_query)
        context_type = guess_context_type(user_query)
        enh = get_enhanced_historical_embedding({'text': user_query, 'year_start': 0}, None)

        # 3) Thalamic router
        decision = self.net._thalamic_router.analyze_conversation_intent(user_query, qfeat)
        plan = self.net._thalamic_router.route_conversation(decision, user_query, qfeat)

        # 4) CNS coordinate for this context
        coord = await self.net._cns.coordinate({'type': context_type, 'urgency': 0.6, 'threat': 0.0})

        # 5) Execute plan (sequential for simplicity)
        contributions: List[Dict[str, Any]] = []
        response_text_parts: List[str] = []

        for spec in plan['routing_sequence']:
            if spec == 'general_chat':
                txt = f"Let’s explore your question: ‘{user_query}’."
                contributions.append({'specialist': spec, 'text': txt, 'quality': 0.6})
                response_text_parts.append(txt)
            elif spec == 'historical_specialist':
                period = get_historical_period(0)
                txt = f"Historically, relevant context points to {period} themes and parallels."
                contributions.append({'specialist': spec, 'text': txt, 'quality': 0.7})
                response_text_parts.append(txt)
            elif spec == 'amygdala_specialist':
                th = self.net._amygdala.process_threat(qfeat, context={'categories': [], 'tone': '', 'impact': ''})
                sal = self.net._amygdala.process_emotional_salience(qfeat, event_data={'title': user_query})
                txt = f"Emotionally, threat={th.get('threat_score',0):.2f}, intensity={sal.get('emotional_intensity',0):.2f}."
                contributions.append({'specialist': spec, 'text': txt, 'quality': 0.6})
                response_text_parts.append(txt)
            elif spec == 'hippocampus_specialist':
                mem = self.net._hippocampus.encode_memory(qfeat)
                strength = float(np.mean(mem)) if mem else 0.0
                txt = f"Memory recall patterns activated (strength {strength:.2f})."
                contributions.append({'specialist': spec, 'text': txt, 'quality': 0.6})
                response_text_parts.append(txt)
            elif spec == 'analytical_specialist':
                txt = "Analytically, we can decompose causes/effects and psychological mechanisms involved."
                contributions.append({'specialist': spec, 'text': txt, 'quality': 0.7})
                response_text_parts.append(txt)
            else:
                txt = f"Routed to {spec}."
                contributions.append({'specialist': spec, 'text': txt, 'quality': 0.5})
                response_text_parts.append(txt)

        # 6) Adaptive routing update (dummy)
        await self.net._thalamic_router.adaptive_routing_update(plan, {'user_satisfaction': 0.8, 'response_quality': 0.7}, qfeat)

        return {
            'response_text': ' '.join(response_text_parts),
            'routing_plan': plan,
            'routing_explanation': self.net._thalamic_router.explain_routing_decision(decision),
            'cns_state_before': state0,
            'cns_coordination': coord,
            'contributions': contributions,
        }

