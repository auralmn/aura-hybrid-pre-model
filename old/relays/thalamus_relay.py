import numpy as np

class ThalamusRelay:
    def __init__(self, input_dim, output_dim, gating_strength=0.7):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gating_strength = gating_strength
        self.gating_weights = np.ones(output_dim) * gating_strength

    def relay(self, x: np.ndarray) -> np.ndarray:
        # Apply gating/synchronization to relay features
        return x[:self.output_dim] * self.gating_weights