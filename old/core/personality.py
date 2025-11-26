from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class PersonalityProfile:
    embedding: np.ndarray              # Core personality embedding vector
    traits: Dict[str, float]           # e.g. {"openness": 0.7, "agreeableness": 0.5}
    description: str                   # Human-readable profile ("curious, cooperative, analytical")