import numpy as np


class HippocampusModule:
    def __init__(self, input_dim=384, memory_strength=0.85):
        self.input_dim = input_dim
        self.memory_matrix = np.zeros((input_dim,input_dim), dtype=np.float32)
        self.memory_strength = memory_strength

    def encode(self, x: np.ndarray) -> np.ndarray:
        # Pattern completion/memory-like transformation
        memory_out = x + self.memory_strength * (self.memory_matrix @ x)
        return memory_out

    def update_memory(self, x: np.ndarray):
        # Hebbian-like update for context encoding
        self.memory_matrix += np.outer(x, x) * 0.01