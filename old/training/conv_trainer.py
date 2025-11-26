from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from ..core.network import Network


TOPIC_TO_SPECIALISTS: Dict[str, List[str]] = {
    'database_types': ['analytical_specialist', 'general_chat'],
    'quantum_physics': ['analytical_specialist', 'scientific_specialist'],
    'history': ['historical_specialist', 'hippocampus_specialist'],
    'emotions': ['amygdala_specialist', 'general_chat'],
    'empathy': ['amygdala_specialist', 'general_chat'],
    'programming': ['analytical_specialist', 'technical_specialist'],
    'philosophy': ['analytical_specialist', 'general_chat'],
    'science': ['analytical_specialist', 'scientific_specialist'],
    'general': ['general_chat'],
}


def expected_specialists_for_topic(topic: str) -> List[str]:
    return TOPIC_TO_SPECIALISTS.get(topic, TOPIC_TO_SPECIALISTS['general'])


@dataclass
class AuraConversationTrainer:
    net: Network
    offline: bool = True
    stats: Dict[str, Any] = field(default_factory=lambda: {
        'total_processed': 0,
        'routing_accuracy': [],
        'topic_relevance': [],
    })

    def _encode_text(self, text: str) -> np.ndarray:
        # Use zeros to avoid model downloads by default
        return np.zeros(384, dtype=np.float32)

    def _topic_relevance_router(self, query: str, topic: str) -> float:
        # Use our thalamic router as a classifier for routing relevance
        feat = self._encode_text(query)
        decision = self.net._thalamic_router.analyze_conversation_intent(query, feat)
        scores = decision.get('routing_scores', {})
        expected = expected_specialists_for_topic(topic)
        # Average confidence over expected specialists, fallback to 0
        vals = [scores[k]['confidence'] for k in expected if k in scores]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _routing_overlap(self, decision: Dict[str, Any], topic: str) -> float:
        expected = set(expected_specialists_for_topic(topic))
        primary = decision.get('primary_target', '')
        second = set(decision.get('secondary_targets', []) or [])
        chosen = {primary} | second
        return len(chosen & expected) / max(1, len(expected))

    def assess_response_quality(self, user_query: str, assistant_response: str, topic: str) -> float:
        # Use router-based topic relevance rather than keyword matching
        rel = self._topic_relevance_router(assistant_response, topic)
        # Length heuristic as tie-breaker
        n = len(assistant_response.split())
        len_score = 0.8 if 10 <= n <= 200 else (0.4 if n < 10 else 0.6)
        return 0.7 * rel + 0.3 * len_score

    async def process_conversation(self, conv: Dict[str, Any]) -> Dict[str, Any]:
        """Process one conversation record and update stats."""
        topic = conv.get('topic', 'general')
        turns = conv.get('turns', [])
        last_overlap = 0.0
        last_rel = 0.0
        any_user = False
        # Iterate turns in pairs user→assistant
        for i in range(0, len(turns), 2):
            if i >= len(turns) or turns[i].get('role') != 'user':
                continue
            any_user = True
            user_q = turns[i].get('content', '')
            # Router decision on user query
            feat = self._encode_text(user_q)
            decision = self.net._thalamic_router.analyze_conversation_intent(user_q, feat)
            last_overlap = self._routing_overlap(decision, topic)
            last_rel = self._topic_relevance_router(user_q, topic)
            self.stats['routing_accuracy'].append(last_overlap)
            self.stats['topic_relevance'].append(last_rel)
            # If next turn is assistant, assess response quality
            if i + 1 < len(turns) and turns[i + 1].get('role') == 'assistant':
                resp = turns[i + 1].get('content', '')
                q = self.assess_response_quality(user_q, resp, topic)
                # Could be used to update learning; here we just aggregate
        self.stats['total_processed'] += 1
        return {'routing_accuracy': last_overlap, 'topic_relevance': last_rel, 'had_user_turn': any_user}

    async def process_dataset(self, path: str, limit: Optional[int] = None, default_topic: Optional[str] = None) -> Dict[str, Any]:
        cnt = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # Apply default topic if missing
                if default_topic and not rec.get('topic'):
                    rec['topic'] = default_topic
                await self.process_conversation(rec)
                cnt += 1
                if limit and cnt >= limit:
                    break
        # Summaries
        ra = np.array(self.stats['routing_accuracy']) if self.stats['routing_accuracy'] else np.array([0.0])
        tr = np.array(self.stats['topic_relevance']) if self.stats['topic_relevance'] else np.array([0.0])
        return {
            'processed': self.stats['total_processed'],
            'routing_accuracy_mean': float(ra.mean()),
            'topic_relevance_mean': float(tr.mean()),
        }
