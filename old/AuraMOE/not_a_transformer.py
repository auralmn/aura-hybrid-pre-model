"""
AURA‑MOE: Liquid‑MoE Routing API (single‑file)
==============================================

A drop‑in, **streaming** Mixture‑of‑Experts router with **Liquid (continuous‑time)** gating,
**Top‑K sparse routing**, optional **spiking attention** gain, and **energy metering**.
No backprop across the router—experts learn locally (e.g., your NLMS heads).

Highlights
- **Liquid gating** (ODE‑style dynamics) for robust, adaptive expert selection
- **Top‑K sparse** routing with tiny **load‑balance nudge** (no global backprop)
- **Attention gain** hook for spike/K‑WTA modulation
- **EnergyMeter** to track MAC‑level energy (device‑tunable)
- **HF‑style forward(...)** + simple expert protocol, plus adapters for AURA Neuron

Quickstart
----------
```python
import numpy as np
from aura_moe_api import AURAMOE, NLMSExpertAdapter, SpikingAttention

# 1) Wrap your AURA NLMS specialists as experts
experts = {
    "general_chat": NLMSExpertAdapter(neuron_general),
    "historical":   NLMSExpertAdapter(neuron_hist),
    "amygdala":     NLMSExpertAdapter(neuron_amyg),
}

router = AURAMOE(
    experts=experts,
    in_dim=384,
    hidden_dim=64,
    top_k=2,
)

# Optional: attach spiking attention (k‑WTA)
router.attach_spiking_attention(SpikingAttention(decay=0.7, theta=1.0, k_winners=5, gain_up=1.6, gain_down=0.6))

x = np.random.randn(384)
text = "Compare Rome and Han dynasty military logistics."

# Inference
out = router.forward(x, text=text)  # dict with y, gates, per_expert, energy
print(router.pretty_route(out))

# Online learning (streaming)—teach the routed experts
import trio
async def teach():
    await router.learn(x, y_true=1.0, text=text)  # 1.0 = good routing outcome
trio.run(teach)
```

Torch wrapper (optional)
------------------------
If PyTorch is installed, a small `HFMoEAdapter(nn.Module)` is exposed so you
can slot this in a HF pipeline while keeping local learning in the experts.

"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Protocol, runtime_checkable
import numpy as np

# ---------------------- utils ----------------------

def _softplus(z: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)

def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def _softmax(z: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = z / max(1e-8, temp)
    z = z - np.max(z)
    ez = np.exp(z)
    s = ez.sum()
    return ez / (s + 1e-12)

# ---------------------- energy meter ----------------------

@dataclass
class EnergyMeter:
    """Crude MAC‑level energy accounting (J). Tune e_mac_j per device."""
    e_mac_j: float = 3e-12   # ~3 pJ / MAC (CPU/GPU/NPU dependent)
    total_j: float = 0.0

    def add_macs(self, nmacs: int) -> None:
        self.total_j += self.e_mac_j * float(nmacs)

    def reset(self) -> None:
        self.total_j = 0.0

# ---------------------- spiking attention (k‑WTA) ----------------------

@dataclass
class SpikingAttention:
    """k‑WTA spiking attention over token ids; returns per‑vocab gains if needed.
    In this API we only derive an aggregate **gain** ∈ [0.8, 2.0] to modulate gating temperature.
    """
    decay: float = 0.7
    theta: float = 1.0
    k_winners: int = 5
    gain_up: float = 1.5
    gain_down: float = 0.6
    vocab_size: int = 50000

    def compute_gain(self, token_seq: List[int]) -> float:
        if not token_seq:
            return 1.0
        v: Dict[int, float] = {}
        spikes: Dict[int, int] = {}
        for j in token_seq:
            vj = self.decay * v.get(j, 0.0) + 1.0
            if vj >= self.theta:
                spikes[j] = spikes.get(j, 0) + 1
                vj -= self.theta
            v[j] = vj
        if not spikes:
            return 1.0
        # k‑WTA winners
        ranked = sorted(spikes.items(), key=lambda kv: (-kv[1], kv[0]))
        winners = {j for j, _ in ranked[:max(1, self.k_winners)]}
        seen = set(spikes.keys()) | set(v.keys())
        gains = []
        for j in seen:
            gains.append(self.gain_up if j in winners else self.gain_down)
        g = float(np.mean(gains)) if gains else 1.0
        return float(np.clip(g, 0.8, 2.0))

# ---------------------- liquid cell (continuous‑time) ----------------------

@dataclass
class LiquidCell:
    in_dim: int
    hidden_dim: int
    dt: float = 0.02
    tau_min: float = 0.02
    tau_max: float = 2.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1337))

    W: np.ndarray = field(init=False)  # hidden->hidden
    U: np.ndarray = field(init=False)  # input->hidden
    b: np.ndarray = field(init=False)
    V: np.ndarray = field(init=False)  # input->tau
    c: np.ndarray = field(init=False)
    h: np.ndarray = field(init=False)

    def __post_init__(self):
        self.W = self.rng.normal(0, np.sqrt(2.0/(self.hidden_dim+self.hidden_dim)), (self.hidden_dim, self.hidden_dim))
        self.U = self.rng.normal(0, np.sqrt(2.0/(self.in_dim+self.hidden_dim)),     (self.hidden_dim, self.in_dim))
        self.b = np.zeros((self.hidden_dim,), dtype=np.float64)
        self.V = self.rng.normal(0, np.sqrt(2.0/(self.in_dim+self.hidden_dim)),     (self.hidden_dim, self.in_dim))
        self.c = self.rng.normal(0, 0.1, (self.hidden_dim,))
        self.h = np.zeros((self.hidden_dim,), dtype=np.float64)

    def reset(self) -> None:
        self.h[:] = 0.0

    def step(self, x: np.ndarray, energy: Optional[EnergyMeter] = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        vx = self.V @ x + self.c
        tau = self.tau_min + _softplus(vx)
        tau = np.minimum(tau, self.tau_max)
        Wh = self.W @ self.h
        Ux = self.U @ x
        a = _tanh(Wh + Ux + self.b)
        dh = - self.h / np.maximum(tau, 1e-6) + a
        self.h = self.h + self.dt * dh
        if energy is not None:
            nmacs = (self.hidden_dim*self.hidden_dim) + (self.hidden_dim*self.in_dim)
            energy.add_macs(nmacs)
        return self.h.copy()

# ---------------------- liquid gating over experts ----------------------

@dataclass
class LiquidGatingNetwork:
    in_dim: int
    hidden_dim: int
    n_experts: int
    top_k: int = 2
    temperature: float = 1.0
    usage_smoothing: float = 0.99
    bias_lr: float = 0.01

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(2025))
    energy: EnergyMeter = field(default_factory=EnergyMeter)

    cell: LiquidCell = field(init=False)
    Wg: np.ndarray = field(init=False)  # hidden->experts logits
    bg: np.ndarray = field(init=False)
    usage_ma: np.ndarray = field(init=False)

    def __post_init__(self):
        self.cell = LiquidCell(self.in_dim, self.hidden_dim, rng=self.rng)
        self.Wg = self.rng.normal(0, np.sqrt(2.0/(self.hidden_dim+self.n_experts)), (self.n_experts, self.hidden_dim))
        self.bg = np.zeros((self.n_experts,), dtype=np.float64)
        self.usage_ma = np.zeros((self.n_experts,), dtype=np.float64)

    def forward(self, x: np.ndarray, attn_gain: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self.cell.step(x, self.energy)
        logits = (self.Wg @ h) + self.bg
        temp = max(0.2, self.temperature / max(1e-6, attn_gain))
        probs = _softmax(logits, temp=temp)
        k = max(1, min(self.top_k, self.n_experts))
        topk_idx = np.argpartition(probs, -k)[-k:]
        topk_probs = probs[topk_idx]
        gates = topk_probs / (topk_probs.sum() + 1e-12)
        out = np.zeros_like(probs)
        out[topk_idx] = gates
        # moving‑avg usage for load balancing
        self.usage_ma = self.usage_smoothing*self.usage_ma + (1.0-self.usage_smoothing)*out
        return out, topk_idx, probs

    def nudge_for_load_balance(self) -> None:
        if self.n_experts <= 0:
            return
        target = 1.0 / float(self.n_experts)
        delta = target - self.usage_ma
        self.bg += self.bias_lr * delta

# ---------------------- expert protocol & adapters ----------------------

@runtime_checkable
class ExpertAdapter(Protocol):
    def predict(self, x: np.ndarray) -> float: ...
    async def update(self, x: np.ndarray, y_true: float) -> float: ...

class NLMSExpertAdapter:
    """Wraps an AURA Neuron (with NLMSHead) as an expert."""
    def __init__(self, neuron: Any):
        self.neuron = neuron

    def predict(self, x: np.ndarray) -> float:
        return float(self.neuron.get_readout(x))

    async def update(self, x: np.ndarray, y_true: float) -> float:
        return float(await self.neuron.update_nlms(x, y_true))

# ---------------------- config ----------------------

@dataclass
class LiquidMoEConfig:
    in_dim: int
    hidden_dim: int = 64
    top_k: int = 2
    temperature: float = 1.0
    e_mac_j: float = 3e-12

# ---------------------- main API ----------------------

class AURAMOE:
    """Liquid‑MoE router with HF‑style forward and streaming learn.

    Parameters
    ----------
    experts : Mapping[str, ExpertAdapter]
        Name -> expert implementing predict(x)->float and async update(x,y)->float.
    in_dim : int
        Feature dimension of input vector x.
    hidden_dim : int
        Liquid state size (64..256 works well).
    top_k : int
        Number of experts to route per call.
    temperature : float
        Softmax temperature (attention gain will modulate this per call).
    """

    def __init__(self,
                 experts: Mapping[str, ExpertAdapter],
                 in_dim: int,
                 hidden_dim: int = 64,
                 top_k: int = 2,
                 temperature: float = 1.0,
                 e_mac_j: float = 3e-12):
        if not experts:
            raise ValueError("AURAMOE requires at least one expert")
        self.names: List[str] = list(experts.keys())
        self.experts: Dict[str, ExpertAdapter] = dict(experts)
        self.in_dim = int(in_dim)
        self.energy = EnergyMeter(e_mac_j=e_mac_j)
        self.attn: Optional[SpikingAttention] = None
        self.gating = LiquidGatingNetwork(
            in_dim=self.in_dim,
            hidden_dim=int(hidden_dim),
            n_experts=len(self.names),
            top_k=int(top_k),
            temperature=float(temperature),
        )

    # ---------- attention utilities ----------
    def attach_spiking_attention(self, attn: SpikingAttention) -> None:
        self.attn = attn

    @staticmethod
    def _tokenize(text: str, vocab: int = 50000) -> List[int]:
        return [hash(w) % vocab for w in text.lower().split()] if text else []

    def _attn_gain(self, text: Optional[str]) -> float:
        if self.attn is None or not text:
            return 1.0
        toks = self._tokenize(text, self.attn.vocab_size)
        return self.attn.compute_gain(toks)

    # ---------- core ops ----------
    def route(self, x: np.ndarray, *, text: Optional[str] = None) -> Dict[str, Any]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.in_dim:
            raise ValueError(f"x has dim {x.shape[0]}, expected {self.in_dim}")
        attn_gain = self._attn_gain(text)
        gates_sparse, topk_idx, probs = self.gating.forward(x, attn_gain=attn_gain)
        chosen = [(self.names[i], float(gates_sparse[i])) for i in topk_idx]
        # energy for K expert readouts
        self.energy.add_macs(len(topk_idx) * self.in_dim)
        y = 0.0
        per_expert: Dict[str, Dict[str, float]] = {}
        for i in topk_idx:
            name = self.names[int(i)]
            gate = float(gates_sparse[i])
            pred = float(self.experts[name].predict(x))
            per_expert[name] = {"gate": gate, "pred": pred}
            y += gate * pred
        return {
            "y": float(y),
            "topk": chosen,
            "probs": probs,
            "per_expert": per_expert,
            "energy_j": self.energy.total_j + self.gating.energy.total_j,
            "attn_gain": attn_gain,
        }

    async def learn(self, x: np.ndarray, y_true: float, *, text: Optional[str] = None) -> Dict[str, Any]:
        out = self.route(x, text=text)
        # gate‑weighted local updates (simple; you can customize the target shaping)
        for name, info in out["per_expert"].items():
            await self.experts[name].update(x, float(y_true))
        self.gating.nudge_for_load_balance()
        return out

    # ---------- HF‑style forward ----------
    def forward(self, inputs: Any, attention_mask: Any = None, return_dict: bool = True, *, text: Optional[str] = None) -> Dict[str, Any]:
        """Accepts numpy array or torch tensor for convenience."""
        x = inputs
        # Lazy torch interop without hard dependency
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
        out = self.route(x, text=text)
        if return_dict:
            return out
        # else return y only
        return {"y": out["y"]}

    # ---------- pretty print ----------
    @staticmethod
    def pretty_route(out: Dict[str, Any], k: int = 4) -> str:
        pairs = sorted([(n, d["gate"], d["pred"]) for n, d in out["per_expert"].items()], key=lambda t: -t[1])
        lines = [f"AURA‑MOE → y={out['y']:.4f}  (attn_gain={out.get('attn_gain',1.0):.2f}, energy={out['energy_j']:.3e} J)"]
        for name, gate, pred in pairs[:k]:
            lines.append(f"  • {name:<24} gate={gate:.3f}  pred={pred:.3f}")
        return "\n".join(lines)

# ---------------------- optional torch wrapper ----------------------

try:
    import torch
    import torch.nn as nn

    class HFMoEAdapter(nn.Module):
        """Tiny nn.Module wrapper around AURAMOE (no backprop through router)."""
        def __init__(self, router: AURAMOE):
            super().__init__()
            self.router = router

        def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, *, text: Optional[str] = None):
            out = self.router.forward(x, attention_mask=attention_mask, return_dict=True, text=text)
            y = torch.tensor(out["y"]).to(x.device)
            return {"logits": y, "gates": torch.tensor(out["probs"]).to(x.device)}
except Exception:
    HFMoEAdapter = None  # type: ignore
