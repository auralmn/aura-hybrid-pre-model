import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
from scipy.special import softmax

def softplus(x): return np.logaddexp(0, x)
def tanh(x): return np.tanh(x)

@dataclass
class EnergyMeter:
    """Tracks energy consumption in Joules (J)"""
    total_j: float = 0.0
    def add_macs(self, n_macs: int):
        # Approx 4.6pJ per MAC (32-bit float) on modern hardware
        self.total_j += n_macs * 4.6e-12
    def reset(self):
        self.total_j = 0.0

# Bandit-based gating for expert selection
class BanditGating:
    """UCB-based bandit gating to track and select best-performing experts"""
    def __init__(self, n_experts: int, exploration_factor: float = 2.0):
        self.n_experts = n_experts
        self.exploration_factor = exploration_factor
        # Track rewards (negative errors) and counts for each expert
        self.total_rewards = np.zeros(n_experts, dtype=np.float64)
        self.selection_counts = np.ones(n_experts, dtype=np.float64)  # Start at 1 to avoid division by zero
        self.total_selections = n_experts  # Total times any expert was selected

    def update(self, expert_idx: int, error: float):
        """Update bandit statistics with expert performance"""
        # Convert error to reward (lower error = higher reward)
        reward = 1.0 / (1.0 + abs(error))  # Reward in [0, 1]
        self.total_rewards[expert_idx] += reward
        self.selection_counts[expert_idx] += 1.0
        self.total_selections += 1

    def get_ucb_scores(self) -> np.ndarray:
        """Compute UCB scores for all experts"""
        # Average reward
        avg_rewards = self.total_rewards / self.selection_counts

        # Confidence bound
        confidence = self.exploration_factor * np.sqrt(
            np.log(self.total_selections + 1) / self.selection_counts
        )

        # UCB = average reward + confidence bound
        ucb_scores = avg_rewards + confidence
        return ucb_scores

    def select_top_k(self, k: int, base_gates: np.ndarray) -> tuple:
        """Select top-k experts using UCB scores, modulated by base gates"""
        ucb_scores = self.get_ucb_scores()

        # Combine UCB scores with base gates (weighted combination)
        # UCB provides exploration/exploitation, gates provide input-specific routing
        combined_scores = 0.7 * base_gates + 0.3 * ucb_scores

        # Select top-k
        topk_idx = np.argsort(combined_scores)[-k:][::-1]

        # Normalize gates
        topk_gates = combined_scores[topk_idx]
        topk_gates = np.maximum(topk_gates, 0.0)  # Ensure non-negative
        gate_sum = np.sum(topk_gates)
        if gate_sum > 0:
            topk_gates = topk_gates / gate_sum
        else:
            topk_gates = np.ones(k) / k  # Uniform if all zero

        return topk_idx, topk_gates

    def reset(self):
        """Reset bandit statistics"""
        self.total_rewards.fill(0.0)
        self.selection_counts.fill(1.0)
        self.total_selections = self.n_experts

@dataclass
class LiquidCell:
    in_dim: int; hidden_dim: int; dt: float = 0.02
    tau_min: float = 0.02; tau_max: float = 2.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1337))
    W: np.ndarray = field(init=False); U: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False); V: np.ndarray = field(init=False)
    c: np.ndarray = field(init=False); h: np.ndarray = field(init=False)
    def __post_init__(self):
        self.W = self.rng.normal(0, np.sqrt(2.0/(self.hidden_dim+self.hidden_dim)), (self.hidden_dim, self.hidden_dim))
        self.U = self.rng.normal(0, np.sqrt(2.0/(self.in_dim+self.hidden_dim)), (self.hidden_dim, self.in_dim))
        self.b = np.zeros((self.hidden_dim,), dtype=np.float64)
        self.V = self.rng.normal(0, np.sqrt(2.0/(self.in_dim+self.hidden_dim)), (self.hidden_dim, self.in_dim))
        self.c = self.rng.normal(0, 0.1, (self.hidden_dim,))
        self.h = np.zeros((self.hidden_dim,), dtype=np.float64)
    def reset(self): self.h[:] = 0.0
    def step(self, x: np.ndarray, energy: Optional[EnergyMeter] = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        vx = self.V @ x + self.c; tau = self.tau_min + softplus(vx)
        tau = np.minimum(tau, self.tau_max)
        Wh = self.W @ self.h; Ux = self.U @ x
        a = tanh(Wh + Ux + self.b); dh = - self.h / np.maximum(tau, 1e-6) + a
        self.h = self.h + self.dt * dh
        if energy is not None: energy.add_macs((self.hidden_dim*self.hidden_dim) + (self.hidden_dim*self.in_dim))
        return self.h.copy()
    def state_dict(self) -> Dict: return {"W": self.W, "U": self.U, "b": self.b, "V": self.V, "c": self.c, "h": self.h}
    def load_state_dict(self, state: Dict):
        self.W = state["W"]; self.U = state["U"]; self.b = state["b"]
        self.V = state["V"]; self.c = state["c"]; self.h = state["h"]

@dataclass
class LiquidGatingNetwork:
    in_dim: int; hidden_dim: int; n_experts: int
    top_k: int = 2; temperature: float = 1.0
    usage_smoothing: float = 0.99; bias_lr: float = 0.01
    usage_beta: float = 0.5
    cell: LiquidCell = field(init=False); Wg: np.ndarray = field(init=False)
    bg: np.ndarray = field(init=False); usage_ma: np.ndarray = field(init=False)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(2025))
    energy: EnergyMeter = field(default_factory=EnergyMeter)
    def __post_init__(self):
        self.cell = LiquidCell(self.in_dim, self.hidden_dim, rng=self.rng)
        self.Wg = self.rng.normal(0, np.sqrt(2.0/(self.hidden_dim+self.n_experts)), (self.n_experts, self.hidden_dim))
        self.bg = np.zeros((self.n_experts,), dtype=np.float64)
        self.usage_ma = np.zeros((self.n_experts,), dtype=np.float64)
        self._lock = asyncio.Lock()
    async def forward(self, x: np.ndarray, attn_gain: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        async with self._lock:
            h = self.cell.step(x, self.energy)
            logits = (self.Wg @ h) + self.bg
            
            # Apply usage bias (if enabled)
            if self.usage_beta > 0:
                logits = self._apply_usage_bias(logits)
            
            # CRITICAL FIX: Apply temperature scaling with attn_gain
            # Design choice: high prosody (high attn_gain) -> lower temp -> sharper distribution (focused)
            #                low prosody (low attn_gain) -> higher temp -> flatter distribution (exploratory)
            temp = max(0.2, self.temperature / max(1e-6, attn_gain))
            logits_scaled = logits / temp
            
            # Softmax with temperature applied
            probs = np.exp(logits_scaled - np.max(logits_scaled))
            probs = probs / np.sum(probs)
            
            k = max(1, min(self.top_k, self.n_experts))
            topk_idx = np.argpartition(probs, -k)[-k:]
            topk_probs = probs[topk_idx]
            if topk_probs.sum() <= 1e-12: gates = np.ones_like(topk_probs) / len(topk_probs)
            else: gates = topk_probs / topk_probs.sum()
            out = np.zeros_like(probs); out[topk_idx] = gates
            eps = 0.01
            if self.n_experts > 0 and eps > 0:
                j = int(np.argmin(self.usage_ma)); out = (1.0 - eps) * out; out[j] += eps
            self.usage_ma = self.usage_smoothing * self.usage_ma + (1.0 - self.usage_smoothing) * out
            # Track last winners for dopamine reward
            self.last_winners = topk_idx
            return out, topk_idx, probs
    def _apply_usage_bias(self, logits: np.ndarray) -> np.ndarray:
        eps = 1e-6; target = 1.0 / self.n_experts
        inv_usage = target / (self.usage_ma + eps)
        return logits + self.usage_beta * np.log(inv_usage)
    async def apply_endocrine(self, *, cortisol: float = 0.0, gh: float = 0.0,
                             thyroid: float = 1.0, dopamine: float = 0.0, eps: Optional[float] = None) -> None:
        """Apply endocrine modulation to gating network"""
        async with self._lock:
            # Temperature ↑ with cortisol (stress) — clamp for stability
            self.temperature = float(np.clip(self.temperature * (1.0 + 0.30 * cortisol), 0.5, 2.5))

            # Bias LR scales with thyroid (metabolic rate around 1.0 baseline)
            self.bias_lr = float(np.clip(self.bias_lr * (1.0 + 0.40 * (thyroid - 1.0)), 1e-4, 0.1))

            # Top-K capacity expands with GH (growth hormone), but never beyond n_experts
            base_top_k = getattr(self, 'base_top_k', self.top_k)
            self.base_top_k = base_top_k
            self.top_k = int(np.clip(round(base_top_k * (1.0 + 0.20 * gh)), 1, self.n_experts))

            # Dopamine: nudge most recent winners' biases (reward)
            if dopamine > 0 and hasattr(self, 'last_winners') and self.last_winners is not None:
                self.bg[self.last_winners] += 0.10 * float(dopamine)

            # Optional: exploration epsilon override
            if eps is not None:
                self.eps = float(np.clip(eps, 0.0, 0.05))
    async def nudge_for_load_balance(self) -> None:
        async with self._lock:
            if self.n_experts <= 0: return
            target = 1.0 / float(self.n_experts); delta = target - self.usage_ma
            self.bg += self.bias_lr * delta
    def reset(self): self.cell.reset(); self.usage_ma[:] = 0.0; self.energy.reset()
    def state_dict(self) -> Dict:
        return {"cell": self.cell.state_dict(), "Wg": self.Wg, "bg": self.bg, "usage_ma": self.usage_ma}
    def load_state_dict(self, state: Dict):
        self.cell.load_state_dict(state["cell"]); self.Wg = state["Wg"]; self.bg = state["bg"]; self.usage_ma = state["usage_ma"]

@dataclass
class LiquidMoERouter:
    experts: Dict[str, Any] # Type Any to avoid circular import with ExpertNLMSAdapter
    in_dim: int; hidden_dim: int; top_k: int = 2
    temperature: float = 1.0
    gating: LiquidGatingNetwork = field(init=False)
    bandit: BanditGating = field(init=False)
    names: List[str] = field(init=False)
    energy: EnergyMeter = field(default_factory=EnergyMeter)
    use_bandit: bool = True  # Enable bandit gating
    def __post_init__(self):
        self.names = list(self.experts.keys())
        self.gating = LiquidGatingNetwork(
            in_dim=self.in_dim, hidden_dim=self.hidden_dim,
            n_experts=len(self.names), top_k=self.top_k,
            temperature=self.temperature,
        )
        self.bandit = BanditGating(n_experts=len(self.names), exploration_factor=2.0)
    async def route(self, x: np.ndarray, attn_gain: float = 1.0) -> Dict[str, any]:
        gates_sparse, topk_idx_base, probs = await self.gating.forward(x, attn_gain=attn_gain)

        # Use bandit gating if enabled
        if self.use_bandit:
            # Convert sparse gates to dense for bandit
            gates_dense = np.zeros(len(self.names), dtype=np.float64)
            for i, idx in enumerate(topk_idx_base):
                gates_dense[int(idx)] = float(gates_sparse[i])

            # Get bandit-selected experts
            topk_idx, topk_gates = self.bandit.select_top_k(self.top_k, gates_dense)

            # Update gates_sparse and topk_idx
            gates_sparse = topk_gates
            topk_idx = topk_idx
        else:
            topk_idx = topk_idx_base

        chosen = [(self.names[i], float(gates_sparse[j])) for j, i in enumerate(topk_idx)]
        y = 0.0; per_expert: Dict[str, Dict[str, float]] = {}
        for j, i in enumerate(topk_idx):
            name = self.names[int(i)]; gate = float(gates_sparse[j])
            pred = float(self.experts[name].predict(x)); y += gate * pred
            self.energy.add_macs(self.in_dim); per_expert[name] = {"gate": gate, "pred": pred}
        return {'y_hat': float(y), 'topk': chosen, 'probs': probs, 'per_expert': per_expert,
            'energy_j': self.energy.total_j + self.gating.energy.total_j}
    async def learn(self, x: np.ndarray, token_ids: List[int], y_true: float,
                    attn_gain: float = 1.0, attention_bundle: Optional[Dict[str, Any]] = None) -> Dict[str, any]:
        out = await self.route(x, attn_gain=attn_gain); tasks = []
        expert_errors = {}  # Track errors for bandit updates

        for name, info in out['per_expert'].items():
            gate = float(info['gate']);
            if gate <= 0.0: continue
            target = float(y_true)
            pred = float(info['pred'])
            error = abs(target - pred)
            expert_errors[name] = error
            tasks.append(self.experts[name].update(x, target, token_ids, attention_bundle))

        await asyncio.gather(*tasks)

        # Update bandit with expert performance
        if self.use_bandit:
            for name, error in expert_errors.items():
                if name in self.names:
                    expert_idx = self.names.index(name)
                    self.bandit.update(expert_idx, error)

        await self.gating.nudge_for_load_balance(); return out
    def reset(self):
        self.gating.reset()
        self.energy.reset()
        if self.use_bandit:
            self.bandit.reset()
    def state_dict(self) -> Dict:
        expert_states = {name: expert.state_dict() for name, expert in self.experts.items()}
        bandit_state = {
            "total_rewards": self.bandit.total_rewards.tolist(),
            "selection_counts": self.bandit.selection_counts.tolist(),
            "total_selections": self.bandit.total_selections
        } if self.use_bandit else None
        return {"gating": self.gating.state_dict(), "experts": expert_states, "bandit": bandit_state}
    def load_state_dict(self, state: Dict):
        self.gating.load_state_dict(state["gating"])
        for name, expert_state in state["experts"].items():
            if name in self.experts: self.experts[name].load_state_dict(expert_state)
        if self.use_bandit and "bandit" in state and state["bandit"]:
            bandit_state = state["bandit"]
            self.bandit.total_rewards = np.array(bandit_state["total_rewards"])
            self.bandit.selection_counts = np.array(bandit_state["selection_counts"])
            self.bandit.total_selections = bandit_state["total_selections"]

