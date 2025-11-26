from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from ..relays.amygdala_relay import AmygdalaRelay
from ..relays.amygdala_module import AmygdalaModule
from .neuron import Neuron, ActivityState, MaturationStage
import numpy as np


@dataclass
class Amygdala:
    relay: AmygdalaRelay

    def __init__(self,
                 labels_json: str = 'emotion_classifier_labels.json',
                 W_path: str = 'emotion_classifier_W.npy',
                 b_path: str = 'emotion_classifier_b.npy',
                 neuron_count: int = 30,
                 features: int = 384,
                 input_dim: int = 384,
                 threat_threshold: float = 0.7,
                 emotional_sensitivity: float = 0.8):
        # Emotion classifier-backed relay
        self.relay = AmygdalaRelay(labels_json=labels_json, W_path=W_path, b_path=b_path)
        # Threat/emotion relay module
        self.module = AmygdalaModule(input_dim=input_dim, threat_threshold=threat_threshold, emotional_sensitivity=emotional_sensitivity)

        # Populations
        third = max(1, neuron_count // 3)
        self.threat_detectors: List[Neuron] = [
            Neuron(neuron_id=f'threat_{i}', specialization='threat_detector', abilities={'fear_processing': 0.95, 'threat_assessment': 0.9},
                   maturation=MaturationStage.DIFFERENTIATED, activity=ActivityState.RESTING,
                   n_features=features, n_outputs=1)
            for i in range(third)
        ]
        self.emotional_processors: List[Neuron] = [
            Neuron(neuron_id=f'emotion_{i}', specialization='emotional_processor', abilities={'emotional_salience': 0.9, 'valence_detection': 0.85},
                   maturation=MaturationStage.DIFFERENTIATED, activity=ActivityState.RESTING,
                   n_features=features, n_outputs=1)
            for i in range(third)
        ]
        self.social_evaluators: List[Neuron] = [
            Neuron(neuron_id=f'social_{i}', specialization='social_evaluator', abilities={'social_threat': 0.8, 'group_dynamics': 0.85},
                   maturation=MaturationStage.DIFFERENTIATED, activity=ActivityState.RESTING,
                   n_features=features, n_outputs=1)
            for i in range(neuron_count - 2 * third)
        ]
        self.neurons = self.threat_detectors + self.emotional_processors + self.social_evaluators
        for n in self.neurons:
            n.nlms_head.clamp = (0.0, 1.0)
            n.nlms_head.l2 = 1e-4

        self.fear_memories: List[Dict[str, Any]] = []
        self.emotional_associations: Dict[str, Any] = {}
        self.threat_patterns: Dict[str, Any] = {}
        self.stress_level: float = 0.0
        self.alert_state: bool = False

    async def init_population(self):
        """Initialize all neurons with proper attach calls"""
        import trio
        async with trio.open_nursery() as n:
            for neuron in self.neurons:
                n.start_soon(neuron.attach)

    def classify(self, record: Dict[str, Any]) -> Tuple[str, float]:
        return self.relay.predict_record(record)

    def embed(self, record: Dict[str, Any]):
        return self.relay.embed_record(record)

    def process_threat(self, input_pattern: np.ndarray, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # Threat assessment via module
        assess = self.module.assess_threat(input_pattern, context=context or {})
        # Activate threat detector population
        threat_scores = [float(n.get_readout(input_pattern)) for n in self.threat_detectors]
        threat_level = float(np.mean(threat_scores)) if threat_scores else 0.0
        if threat_level > self.module.threat_threshold:
            self.stress_level = min(1.0, self.stress_level + 0.2)
            self.alert_state = True
            self.fear_memories.append({
                'pattern': input_pattern,
                'threat_level': threat_level,
                'context': context,
                'timestamp': len(self.fear_memories)
            })
        else:
            self.stress_level = max(0.0, self.stress_level - 0.1)
            self.alert_state = False
        assess.update({
            'population_threat': threat_level,
            'stress_response': self.stress_level,
            'alert_state': self.alert_state,
        })
        return assess

    def process_emotional_salience(self, input_pattern: np.ndarray, event_data: Dict[str, Any] | None = None) -> Dict[str, float]:
        scores = [float(n.get_readout(input_pattern)) for n in self.emotional_processors]
        valence = float(np.mean(scores)) if scores else 0.0
        intensity = float(np.std(scores)) if scores else 0.0
        if event_data and intensity > 0.5:
            key = event_data.get('title', 'unknown_event')
            self.emotional_associations[key] = {
                'valence': valence,
                'intensity': intensity,
                'categories': event_data.get('categories', []),
                'year': event_data.get('year_start'),
            }
        return {'emotional_valence': valence, 'emotional_intensity': intensity, 'significance': valence * intensity}

    def assess_social_threat(self, input_pattern: np.ndarray, event_data: Dict[str, Any] | None = None) -> Dict[str, float]:
        scores = [float(n.get_readout(input_pattern)) for n in self.social_evaluators]
        social_threat_level = float(np.mean(scores)) if scores else 0.0
        if event_data:
            cats = [str(c).capitalize() for c in event_data.get('categories', [])]
            high = {'War', 'Revolution', 'Collapse', 'Invasion', 'Persecution'}
            boost = sum(1 for c in cats if c in high) * 0.2
            social_threat_level = min(1.0, social_threat_level + boost)
        return {'social_threat_level': social_threat_level, 'group_stability': 1.0 - social_threat_level}

    async def process(self, input_data: np.ndarray) -> dict:
        """Process input through the amygdala"""
        # Process threat assessment
        threat_assessment = self.process_threat(input_data)
        
        # Process emotional salience
        emotional_salience = self.process_emotional_salience(input_data)
        
        # Assess social threat
        social_threat = self.assess_social_threat(input_data)
        
        return {
            'threat_assessment': threat_assessment,
            'emotional_salience': emotional_salience,
            'social_threat': social_threat,
            'stress_level': self.stress_level,
            'alert_state': self.alert_state
        }

    async def fear_conditioning(self, input_pattern: np.ndarray, outcome: str, event_data: Dict[str, Any] | None = None) -> None:
        t_target = 1.0 if outcome == 'threatening' else 0.0
        for n in self.threat_detectors:
            await n.update_nlms(input_pattern, t_target)
        e_target = self._get_emotional_target(outcome, event_data)
        for n in self.emotional_processors:
            await n.update_nlms(input_pattern, e_target)
        s_target = self._get_social_target(outcome, event_data)
        for n in self.social_evaluators:
            await n.update_nlms(input_pattern, s_target)

    def _get_emotional_target(self, outcome: str, event_data: Dict[str, Any] | None) -> float:
        if outcome == 'threatening':
            return -0.8
        elif outcome == 'positive':
            return 0.8
        else:
            return 0.0

    def _get_social_target(self, outcome: str, event_data: Dict[str, Any] | None) -> float:
        if not event_data:
            return 0.5
        cats = [str(c).capitalize() for c in event_data.get('categories', [])]
        if 'War' in cats or 'Revolution' in cats:
            return 0.9
        elif 'Culture' in cats or 'Innovation' in cats:
            return 0.2
        else:
            return 0.5

    def get_fear_memory_patterns(self) -> List[np.ndarray]:
        return [mem['pattern'] for mem in self.fear_memories]

    def get_emotional_associations(self) -> Dict[str, Any]:
        return self.emotional_associations

    def reset_stress_response(self) -> None:
        self.stress_level = 0.0
        self.alert_state = False

    async def init_weights(self):
        """Initialize weights for all neurons in the amygdala"""
        for neuron in self.neurons:
            if hasattr(neuron, 'init_weights'):
                await neuron.init_weights()
            elif hasattr(neuron, 'nlms_head') and hasattr(neuron.nlms_head, 'init_weights'):
                await neuron.nlms_head.init_weights()
