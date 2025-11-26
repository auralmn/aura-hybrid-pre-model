from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, List, Tuple

from ..relays.cns_relay import CentralNervousSystemModule


class ConsciousnessLevel(Enum):
    DEEP_SLEEP = auto()
    LIGHT_SLEEP = auto()
    DROWSY = auto()
    ALERT = auto()
    FOCUSED = auto()
    HYPERVIGILANT = auto()


class AttentionMode(Enum):
    DIFFUSE = auto()
    FOCUSED = auto()
    DIVIDED = auto()
    SELECTIVE = auto()


class SystemPriority(Enum):
    SURVIVAL = auto()
    LEARNING = auto()
    SOCIAL = auto()
    EXPLORATION = auto()
    MAINTENANCE = auto()


@dataclass
class CentralNervousSystem:
    input_dim: int = 384

    def __post_init__(self) -> None:
        self.cns_relay = CentralNervousSystemModule(input_dim=self.input_dim)
        self.brain_regions: Dict[str, Any] = {}
        self.region_states: Dict[str, Dict[str, Any]] = {}
        self.consciousness_level = ConsciousnessLevel.ALERT
        self.attention_mode = AttentionMode.DIFFUSE
        self.current_priority = SystemPriority.LEARNING
        self.global_arousal: float = 0.5
        self.cognitive_load: float = 0.3
        self.energy_level: float = 1.0
        self.emergency_mode: bool = False
        self.threat_level: float = 0.0

    def register_brain_region(self, name: str, obj: Any, priority: float = 0.5) -> None:
        self.brain_regions[name] = obj
        self.region_states[name] = {
            'priority': float(priority),
            'activity_level': 0.0,
            'resource_allocation': 0.2,
            'last_active': None,
            'performance_score': 0.5,
        }

    def assess_global_state(self) -> Dict[str, Any]:
        # Approximate cognitive load from region activity levels
        if self.region_states:
            act = [st['activity_level'] for st in self.region_states.values()]
            self.cognitive_load = float(sum(act) / max(1, len(act)))
        # Update consciousness level from arousal and load
        a, c = self.global_arousal, self.cognitive_load
        if a > 0.8 and c > 0.6: self.consciousness_level = ConsciousnessLevel.HYPERVIGILANT
        elif a > 0.6:           self.consciousness_level = ConsciousnessLevel.FOCUSED
        elif a > 0.4:           self.consciousness_level = ConsciousnessLevel.ALERT
        elif a > 0.2:           self.consciousness_level = ConsciousnessLevel.DROWSY
        else:                   self.consciousness_level = ConsciousnessLevel.LIGHT_SLEEP
        return {
            'consciousness_level': self.consciousness_level.name,
            'attention_mode': self.attention_mode.name,
            'current_priority': self.current_priority.name,
            'global_arousal': self.global_arousal,
            'cognitive_load': self.cognitive_load,
            'energy_level': self.energy_level,
            'emergency_mode': self.emergency_mode,
            'threat_level': self.threat_level,
        }

    async def process(self, input_data: np.ndarray) -> dict:
        """Process input through the central nervous system"""
        # Assess global state
        global_state = self.assess_global_state()
        
        # Coordinate with context
        context = {'type': 'general', 'threat': 0.0}
        coordination = await self.coordinate(context)
        
        return {
            'global_state': global_state,
            'coordination': coordination,
            'region_states': self.region_states
        }

    async def coordinate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Update attention mode
        self.attention_mode = AttentionMode[self.cns_relay.suggest_attention_mode(context)]
        # Adjust arousal with optional threat from context
        self.global_arousal = self.cns_relay.adjust_arousal(self.global_arousal, context.get('threat'))
        # Determine active regions (simple rule-based)
        ty = str(context.get('type', 'general')).lower()
        active: List[str] = []
        if 'chat' in ty: active.append('router')
        if 'svc' in ty: active.extend(['thalamus','hippocampus'])
        if 'emotion' in ty or context.get('threat',0) > 0.6: active.append('amygdala')
        active = [r for r in active if r in self.brain_regions]
        # Allocate resources by priority
        pr = {name: self.region_states.get(name, {}).get('priority', 0.5) for name in active}
        alloc = self.cns_relay.allocate_resources(active, pr)
        for name, w in alloc.items():
            self.region_states[name]['resource_allocation'] = w
            self.region_states[name]['activity_level'] = min(1.0, self.region_states[name]['activity_level'] + 0.1)
        return {
            'active_regions': active,
            'resource_allocation': alloc,
            'attention_mode': self.attention_mode.name,
            'global_arousal': self.global_arousal,
        }

    async def init_weights(self):
        """Initialize weights for the central nervous system"""
        # CNS doesn't have neurons to initialize, but we can initialize the relay
        if hasattr(self, 'cns_relay') and hasattr(self.cns_relay, 'init_weights'):
            await self.cns_relay.init_weights()

