from dataclasses import dataclass
import numpy as np
from numpy import pi
Array = np.ndarray

@dataclass
class PhasorState:
    """Single-harmonic leaky resonator (kept for reference)."""
    delta0: float
    rho: float = 0.985
    kappa: float = 1.0

    def __post_init__(self):
        self.omega = 2 * pi / max(self.delta0, 1e-4)
        self.cx = 0.0
        self.cy = 0.0

    def step(self, u_t: float) -> Array:
        cosw, sinw = np.cos(self.omega), np.sin(self.omega)
        x, y = self.cx, self.cy
        xr = cosw * x - sinw * y
        yr = sinw * x + cosw * y
        x_new = self.rho * xr + self.kappa * u_t
        y_new = self.rho * yr
        self.cx, self.cy = x_new, y_new
        return np.array([1.0, self.cx, self.cy], dtype=np.float64)

@dataclass
class PhasorBank:
    """H-harmonic leaky resonator bank to approximate a time-selective kernel around Δ0.
    For k=1..H: v_k ← ρ R(kω) v_k + κ u_t e1. Features = [1, Re(v_1), Im(v_1), ..., Re(v_H), Im(v_H)].
    """
    delta0: float
    H: int = 384
    rho: float = 0.985
    kappa: float = 1.0

    def __post_init__(self):
        self.omega = 2 * pi / max(self.delta0, 1e-4)
        self.cx = np.zeros(self.H, dtype=np.float64)
        self.cy = np.zeros(self.H, dtype=np.float64)

    def step(self, u_t: float) -> Array:
        feats = [1.0]
        for k in range(1, self.H + 1):
            cosw, sinw = np.cos(k * self.omega), np.sin(k * self.omega)
            x, y = self.cx[k-1], self.cy[k-1]
            xr = cosw * x - sinw * y
            yr = sinw * x + cosw * y
            x_new = self.rho * xr + self.kappa * u_t
            y_new = self.rho * yr
            self.cx[k-1], self.cy[k-1] = x_new, y_new
            feats.extend([x_new, y_new])
        return np.array(feats, dtype=np.float64)

@dataclass
class TapFeatures:
    delta0: int
    J: int

    def __post_init__(self):
        L = 2 * self.J + 1
        self.buffer = [0.0] * (self.delta0 + self.J + 1)
        self.L = L

    def step(self, u_t: float) -> Array:
        self.buffer.insert(0, u_t)
        self.buffer.pop()
        feats = [1.0]
        for d in range(self.delta0 - self.J, self.delta0 + self.J + 1):
            feats.append(self.buffer[d] if 0 <= d < len(self.buffer) else 0.0)
        return np.array(feats, dtype=np.float64)