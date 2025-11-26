from typing import Dict
import numpy as np
from .neuron import Neuron, MaturationStage, ActivityState


def create_specialist(neuron_id: str, n_features: int = 384) -> Neuron:
    n = Neuron(
        neuron_id=neuron_id,
        specialization='specialist',
        abilities={'classification': 0.9},
        n_features=n_features,
        n_outputs=1,
        maturation=MaturationStage.PROGENITOR,
        activity=ActivityState.RESTING,
    )
    n.nlms_head.clamp = (0.0, 1.0)
    n.nlms_head.l2 = 1e-4
    return n


class SpecialistRegistry:
    def __init__(self, n_features: int = 384):
        self.n_features = n_features
        self._specs: Dict[str, Neuron] = {}

    def get(self, name: str) -> Neuron | None:
        return self._specs.get(name)

    def ensure(self, name: str) -> Neuron:
        n = self._specs.get(name)
        if n is None:
            n = create_specialist(name, n_features=self.n_features)
            self._specs[name] = n
        return n

    @property
    def all(self) -> Dict[str, Neuron]:
        return self._specs

