from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import os, json as _json

from ..core.neuron import Neuron, MaturationStage, ActivityState


def extract_numeric_answer(text: str) -> Optional[float]:
    if not text:
        return None
    # Look for final '#### value' first
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # Fallback: last number in the string
    m2 = list(re.finditer(r"([-+]?\d+(?:\.\d+)?)", text))
    if m2:
        try:
            return float(m2[-1].group(1))
        except Exception:
            return None
    return None


@dataclass
class SocraticTrainer:
    offline: bool = False
    device: Optional[str] = None
    verbose: bool = True
    epochs: int = 1
    weights_dir: str = 'svc_nlms_weights'

    def _load_encoder(self):
        if self.offline:
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            if self.device:
                return SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            return None

    async def train_file(self, path: str, limit: Optional[int] = None) -> Dict[str, Any]:
        sbert = self._load_encoder()
        EMBED = 384
        # Single NLMS neuron to regress numeric answer
        neuron = Neuron(
            neuron_id='socratic_regressor',
            specialization='math_word_problem',
            abilities={'regression': 0.9},
            maturation=MaturationStage.DIFFERENTIATED,
            activity=ActivityState.RESTING,
            n_features=EMBED,
            n_outputs=1,
        )
        # Configure NLMS learning for dense token features
        neuron.nlms_head.clamp = None  # allow real values
        neuron.nlms_head.mu_bias = 0.02
        neuron.nlms_head.mu_tok = 0.08
        neuron.nlms_head.l2 = 1e-3
        # Read dataset
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                q = str(rec.get('question', ''))
                a = str(rec.get('answer', ''))
                y = extract_numeric_answer(a)
                if q and y is not None:
                    records.append((q, float(y)))
                if limit and len(records) >= limit:
                    break
        if not records:
            return {'processed': 0, 'mae': None}
        # Try to resume weights
        try:
            w_path = os.path.join(self.weights_dir,'socratic_w.npy')
            if os.path.isfile(w_path):
                w_loaded = np.load(w_path)
                if w_loaded.shape[0] == EMBED:
                    neuron.nlms_head.w = w_loaded.astype(np.float64)
                    if self.verbose:
                        print("↺ Resumed socratic weights.")
        except Exception:
            pass

        # Attach head to ensure feature slice covers all dims
        try:
            await neuron.nlms_head.attach(
                np.zeros(EMBED, dtype=np.float64),
                slice(0, EMBED),  # tok_slice covers all features
                slice(0, 0),
                slice(0, 0),
            )
        except Exception:
            pass

        # Prepare target transform (log1p if non-negative)
        ys = [y for _, y in records]
        # Try to load scaling meta
        meta_path = os.path.join(self.weights_dir,'socratic_meta.json')
        use_log = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = _json.load(f)
                use_log = bool(meta.get('use_log', False))
                if not use_log:
                    mu_y = float(meta.get('mu_y', np.mean(ys)))
                    std_y = float(meta.get('std_y', np.std(ys) + 1e-8))
            except Exception:
                use_log = None
        if use_log is None:
            use_log = (min(ys) >= 0.0)
        if use_log:
            y_enc = lambda v: float(np.log1p(v))
            y_dec = lambda z: float(np.expm1(z))
        else:
            mu_y = float(locals().get('mu_y', np.mean(ys)))
            std_y = float(locals().get('std_y', np.std(ys) + 1e-8))
            y_enc = lambda v, m=mu_y, s=std_y: float((v - m) / s)
            y_dec = lambda z, m=mu_y, s=std_y: float(z * s + m)

        # Training loop
        n = len(records)
        for ep in range(1, max(1, int(self.epochs)) + 1):
            # Shuffle
            rng = np.random.default_rng(42 + ep)
            idxs = rng.permutation(n)
            mae_sum = 0.0
            count = 0
            for i, ix in enumerate(idxs, 1):
                q, y = records[ix]
                if sbert is not None:
                    x = np.asarray(sbert.encode(q, convert_to_tensor=False), dtype=np.float32)
                else:
                    x = np.zeros(EMBED, dtype=np.float32)
                # Normalize features to unit norm for stability
                norm = float(np.linalg.norm(x) + 1e-8)
                x = x / norm
                # Online NLMS update
                y_t = y_enc(float(y))
                y_pred_t = await neuron.update_nlms(x, y_t)
                # Decode prediction to raw scale for MAE
                y_hat = y_dec(float(y_pred_t))
                mae_sum += abs(float(y_hat) - float(y))
                count += 1
                if self.verbose and (i % 200 == 0 or i == n):
                    bar_w = 30
                    p = i / n
                    filled = int(p * bar_w)
                    mae = mae_sum / max(1, count)
                    print(f"[ep {ep}] [{('#'*filled).ljust(bar_w)}] {i}/{n}  mae={mae:.3f}")
        final_mae = mae_sum / max(1, count)
        # Save weights and scaler params
        try:
            out_dir = self.weights_dir
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, 'socratic_w.npy'), neuron.nlms_head.w)
            meta = {'use_log': bool(use_log)}
            if not use_log:
                meta.update({'mu_y': float(mu_y), 'std_y': float(std_y)})
            with open(os.path.join(out_dir, 'socratic_meta.json'), 'w', encoding='utf-8') as f:
                _json.dump(meta, f)
        except Exception:
            pass
        return {'processed': n, 'epochs': self.epochs, 'final_mae': final_mae, 'saved': {'socratic': True}}
