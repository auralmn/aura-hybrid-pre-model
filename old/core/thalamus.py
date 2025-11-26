import numpy as np
from .neuron import ActivityState, MaturationStage, Neuron
from ..relays.thalamus_relay import ThalamusRelay
from .specialists import SpecialistRegistry




class Thalamus:
    def __init__(self, neuron_count, input_channels=384, output_channels=384,
                 input_dims=384, output_dims=384, gating_strength=0.7):
        self.thalamus_relay = self.setup_thalamus_relay(
            input_dim=input_dims,
            output_dim=output_dims,
            gating_strength=gating_strength
        )
        self.feature_dim = output_dims
        self.neurons = [Neuron(
            neuron_id=i, specialization='relay',
            abilities={'gating':0.9},
            maturation=MaturationStage.DIFFERENTIATED,
            activity=ActivityState.RESTING,
            n_features=input_channels,
            n_outputs=output_channels)
            for i in range(neuron_count)
        ]
        # Bound outputs and add mild regularization for stability
        for n in self.neurons:
            n.nlms_head.clamp = (0.0, 1.0)
            n.nlms_head.l2 = 1e-4
        self.projected_targets = []
        # Dynamic specialists managed by the thalamus
        self.specialists = SpecialistRegistry(n_features=self.feature_dim)

    def relay(self, sensory_input):
        vec = np.asarray(sensory_input, dtype=np.float64).reshape(-1)
        gated_input = self.thalamus_relay.relay(vec)
        gout = np.asarray(gated_input, dtype=np.float64).reshape(-1)
        # pad/trim to neuron feature dim
        n = self.neurons[0].nlms_head.n_features if self.neurons else gout.size
        if gout.size < n: 
            gout = np.pad(gout, (0, n - gout.size))
        elif gout.size > n: 
            gout = gout[:n]
        return [neuron.get_readout(gout) for neuron in self.neurons]

    async def process(self, input_data: np.ndarray) -> dict:
        """Process input through the thalamus"""
        # Use the relay method to process input
        outputs = self.relay(input_data)
        
        return {
            'gated_output': outputs,
            'gating_strength': self.thalamus_relay.gating_strength,
            'neuron_count': len(self.neurons)
        }

    async def process_specialists(self, task_type: str, target_key: str, features) -> dict:
        """Route features to specialists of a given task family.
        - task_type: e.g., 'domain', 'realm', 'era', 'svc_subject'
        - target_key: e.g., 'math', 'theoretical', 'Industrial'
        Creates the positive specialist on demand and updates negatives.
        Returns a dict of specialist_name -> score (prediction before update clamp).
        """
        fam_prefix = f"{task_type}_"
        # Ensure positive specialist exists
        pos_name = f"{fam_prefix}{target_key}"
        pos_spec = self.specialists.ensure(pos_name)

        # Collect all family specialists (including the new one)
        family = {name: n for name, n in self.specialists.all.items() if name.startswith(fam_prefix)}
        scores: dict = {}
        # Update: positive for target, negative for others
        for name, n in family.items():
            y_true = 1.0 if name == pos_name else 0.0
            pred = await n.update_nlms(features, y_true)
            scores[name] = float(pred)
        return scores

    def connect_targets(self, targets):
        self.projected_targets = targets

    def synchronize(self, phase_value):
        for neuron in self.neurons:
            neuron.abilities['phase'] = phase_value

    def setup_thalamus_relay(self, input_dim, output_dim, gating_strength):
        return ThalamusRelay(input_dim=input_dim, output_dim=output_dim, gating_strength=gating_strength)

    async def init_weights(self):
        """Initialize weights for all neurons in the thalamus"""
        for neuron in self.neurons:
            if hasattr(neuron, 'init_weights'):
                await neuron.init_weights()
            elif hasattr(neuron, 'nlms_head') and hasattr(neuron.nlms_head, 'init_weights'):
                await neuron.nlms_head.init_weights()
