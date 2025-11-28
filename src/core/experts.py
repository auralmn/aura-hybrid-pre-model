import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field



class ExpertHead:
    n_features: int
    vocab_size: int
    attention_config: Dict[str, Any]
    mu: float = 0.5
    mu_decay: float = 0.99995
    mu_min: float = 0.1
    initial_bias: float = 0.0
    
    # State
    w: np.ndarray = field(init=False)
    bias: float = field(init=False)
    update_count: int = field(init=False)
    total_error_sq: float = field(init=False)
    last_error: float = field(init=False)
    mu_initial: float = field(init=False)

    def __post_init__(self):
        self.w = np.zeros(self.n_features, dtype=np.float64)
        self.bias = float(self.initial_bias)
        self.update_count = 0
        self.total_error_sq = 0.0
        self.last_error = 0.0
        self.mu_initial = self.mu

    def predict(self, x: np.ndarray) -> float:
        # Linear prediction: y = w*x + b
        return float(np.dot(self.w, x) + self.bias)

    async def update(self, x: np.ndarray, y_true: float, token_ids: List[int],
                     attention_bundle: Optional[Dict[str, Any]] = None) -> float:
        # 1. Prediction
        y_hat = self.predict(x)
        
        # 2. Error
        error = y_true - y_hat
        self.last_error = error
        self.total_error_sq += error**2
        self.update_count += 1
        
        # 3. NLMS Update
        # w = w + mu * error * x / (||x||^2 + eps)
        norm_sq = np.dot(x, x) + 1e-6
        step = (self.mu * error) / norm_sq
        self.w += step * x
        
        # 4. Bias Update
        self.bias += self.mu * error * 0.1 # Slower bias learning
        
        # 5. Decay learning rate
        if self.mu > self.mu_min:
            self.mu *= self.mu_decay
            
        return y_hat

    def get_rmse(self) -> float:
        if self.update_count == 0: return float('inf')
        return float(np.sqrt(self.total_error_sq / self.update_count))

    def state_dict(self) -> Dict:
        return {"w": self.w, "bias": self.bias, "mu": self.mu, "update_count": self.update_count}

    def load_state_dict(self, state: Dict):
        self.w = state["w"]
        self.bias = state.get("bias", 0.0)
        self.mu = state.get("mu", self.mu_initial)
        self.update_count = state.get("update_count", 0)

class NLMSExpertAdapter:
    def __init__(self, neuron: ExpertHead):
        self.neuron = neuron
        
    def predict(self, x: np.ndarray) -> float:
        return self.neuron.predict(x)
        
    async def update(self, x: np.ndarray, y_true: float, token_ids: List[int],
                     attention_bundle: Optional[Dict[str, Any]] = None) -> float:
        return await self.neuron.update(x, y_true, token_ids, attention_bundle)
        
    def state_dict(self) -> Dict:
        return self.neuron.state_dict()
        
    def load_state_dict(self, state: Dict):
        self.neuron.load_state_dict(state)


ExpertNLMSHead = ExpertHead
