
from .neuron import Neuron, MaturationStage, ActivityState
from .nlms import NLMSHead


class NeuronPopulation:
    def __init__(self, n_neurons, n_features, n_outputs):
        self.neurons = [Neuron(i, 'generic', {}, MaturationStage.PROGENITOR, ActivityState.RESTING, n_features, n_outputs)
                        for i in range(n_neurons)]
        self.nlms_head = NLMSHead(n_features=n_features, n_outputs=n_outputs, seed=42)

    async def update_population_nlms(self, features, targets):
        await self.nlms_head.step(features, targets)
