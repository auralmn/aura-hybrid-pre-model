import numpy as np

def db10(x: float) -> float:
    x = max(float(x), 1e-20)
    return 10.0 * np.log10(x)