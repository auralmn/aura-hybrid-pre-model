import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np


def _generate_default_params(labels: List[str]) -> Dict[str, Dict[str, float]]:
    base_freq = 1.5
    base_phase = 0.5
    params: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(labels):
        params[label] = {
            "freq": base_freq + 0.3 * idx,
            "amp": 0.7,
            "phase": base_phase + 0.4 * idx,
        }
    return params


def _sinewave_embedding(primary: str, intensity: float, params: Dict[str, Dict[str, float]], length: int) -> np.ndarray:
    p = params.get(primary)
    if not p:
        return np.zeros(length, dtype=np.float32)
    amp = float(p["amp"]) * float(intensity)
    freq = float(p["freq"]) 
    phase = float(p["phase"]) 
    t = np.linspace(0, 2 * np.pi, length)
    return (amp * np.sin(freq * t + phase)).astype(np.float32)


def _multi_sine_embedding(record: Dict[str, Any], params: Dict[str, Dict[str, float]], length: int) -> np.ndarray:
    pl = record.get('plutchik', {})
    primary = pl.get('primary')
    if isinstance(primary, list):
        primary = primary[0] if primary else ''
    intensity = float(pl.get('intensity', 1.0))
    emb = _sinewave_embedding(primary, intensity, params, length)
    secondary = pl.get('secondary')
    if secondary:
        if isinstance(secondary, list):
            secondary = secondary[0] if secondary else ''
        if secondary in params:
            emb = emb + 0.5 * _sinewave_embedding(secondary, intensity, params, length)
    return emb.astype(np.float32)


@dataclass
class EmotionClassifier:
    W: np.ndarray
    b: np.ndarray

    @classmethod
    def from_files(cls, W_path: str, b_path: str) -> "EmotionClassifier":
        W = np.load(W_path)
        b = np.load(b_path)
        return cls(W=W, b=b)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        logits = X @ self.W + self.b
        logits = logits - np.max(logits, axis=1, keepdims=True)
        expz = np.exp(logits)
        return expz / np.sum(expz, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class AmygdalaRelay:
    def __init__(self,
                 labels_json: str = 'emotion_classifier_labels.json',
                 W_path: str = 'emotion_classifier_W.npy',
                 b_path: str = 'emotion_classifier_b.npy'):
        # Load labels
        try:
            with open(labels_json, 'r', encoding='utf-8') as f:
                ld = json.load(f)
            self.LABEL_TO_IDX: Dict[str, int] = {str(k): int(v) for k, v in ld['LABEL_TO_IDX'].items()}
            self.IDX_TO_LABEL: Dict[int, str] = {int(k): str(v) for k, v in ld['IDX_TO_LABEL'].items()}
        except Exception:
            # Fallback to empty mapping
            self.LABEL_TO_IDX = {}
            self.IDX_TO_LABEL = {}

        # Load classifier weights
        self.clf = EmotionClassifier.from_files(W_path, b_path)
        # Ensure params consistent with label order
        labels_sorted = [self.IDX_TO_LABEL[i] for i in range(len(self.IDX_TO_LABEL))] if self.IDX_TO_LABEL else []
        self.params = _generate_default_params(labels_sorted)
        self.input_dim = int(self.clf.W.shape[0])

    def embed_record(self, record: Dict[str, Any]) -> np.ndarray:
        return _multi_sine_embedding(record, self.params, self.input_dim)

    def predict_record(self, record: Dict[str, Any]) -> Tuple[str, float]:
        x = self.embed_record(record)
        proba = self.clf.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = self.IDX_TO_LABEL.get(idx, str(idx))
        return label, float(proba[idx])

    def predict_embedding(self, embedding: np.ndarray) -> Tuple[int, float]:
        proba = self.clf.predict_proba(embedding)[0]
        idx = int(np.argmax(proba))
        return idx, float(proba[idx])

