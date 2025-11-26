from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import os
import numpy as np

from ..system.historical_runner import HistoricalNetwork
from ..utils.historical_features import get_enhanced_historical_embedding
from .historical_teacher import HistoricalTeacher


@dataclass
class HistoricalTrainer:
    offline: bool = True
    teacher_path: str = 'historical_teacher.md'
    device: str | None = None
    verbose: bool = False
    weights_dir: str = 'svc_nlms_weights'
    epochs: int = 1
    mu_tok: float = 0.08
    mu_bias: float = 0.02
    l2: float = 1e-3
    no_hints: bool = False
    hint_threshold: float = 0.8
    balance: bool = False
    freeze_router: bool = False

    async def train_file(self, path: str, limit: Optional[int] = None) -> Dict[str, Any]:
        net = HistoricalNetwork(offline=self.offline, device=self.device)
        await net.init_weights()
        # Try to resume from previous weights
        try:
            from tools.weights_io import load_network_weights
            counts_loaded = load_network_weights(net, self.weights_dir)
            if self.verbose and counts_loaded:
                print(f"↺ Resumed weights: {counts_loaded}")
        except Exception:
            pass
        teacher = HistoricalTeacher(self.teacher_path)
        # Count total lines for progress (honor limit)
        total_lines = 0
        with open(path, 'r', encoding='utf-8') as fcnt:
            for _ in fcnt:
                total_lines += 1
        if limit:
            total_lines = min(total_lines, int(limit))

        # Configure NLMS params for era specialists
        try:
            for n in getattr(net, 'era_specialists', {}).values():
                n.nlms_head.mu_tok = float(self.mu_tok)
                n.nlms_head.mu_bias = float(self.mu_bias)
                n.nlms_head.l2 = float(self.l2)
        except Exception:
            pass

        # Optionally freeze router learning (mu -> 0)
        if self.freeze_router:
            try:
                router = getattr(net, '_thalamic_router', None)
                if router is not None:
                    for group in getattr(router, 'routing_neurons', {}).values():
                        for rn in group:
                            rn.nlms_head.mu_tok = 0.0
                            rn.nlms_head.mu_bias = 0.0
            except Exception:
                pass

        total = 0  # number of events with a usable ground-truth period
        correct = 0
        processed = 0

        def _canon_period(x: Optional[str]) -> Optional[str]:
            if not x:
                return None
            t = str(x).strip().lower().replace('-', '_').replace(' ', '_')
            mapping = {
                'neolithic': 'Neolithic',
                'ancient': 'Ancient',
                'classical': 'Ancient',
                'medieval': 'Medieval',
                'middle_ages': 'Medieval',
                'early_modern': 'Early_Modern',
                'earlymodern': 'Early_Modern',
                'industrial': 'Industrial',
                'modern': 'Modern',
            }
            return mapping.get(t, None)
        # Read all events into memory to allow multi-epoch passes
        events: List[Dict[str, Any]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                events.append(ev)
                if limit and len(events) >= limit:
                    break
        n_events = len(events)

        # Balanced sampling: interleave by ground-truth period if requested
        def _canon_period(x: Optional[str]) -> Optional[str]:
            if not x:
                return None
            t = str(x).strip().lower().replace('-', '_').replace(' ', '_')
            mapping = {
                'neolithic': 'Neolithic',
                'ancient': 'Ancient',
                'classical': 'Ancient',
                'medieval': 'Medieval',
                'middle_ages': 'Medieval',
                'early_modern': 'Early_Modern',
                'earlymodern': 'Early_Modern',
                'industrial': 'Industrial',
                'modern': 'Modern',
            }
            return mapping.get(t, None)

        if self.balance and n_events > 0:
            buckets: Dict[str, List[int]] = {}
            for idx, ev in enumerate(events):
                true_raw = ev.get('period') or ev.get('era') or ev.get('time_period')
                true = _canon_period(true_raw)
                if true is None:
                    true = 'Unknown'
                buckets.setdefault(true, []).append(idx)
            # round-robin order
            round_robin: List[int] = []
            while any(buckets.values()):
                for k in list(buckets.keys()):
                    if buckets[k]:
                        round_robin.append(buckets[k].pop(0))
            base_order = np.array(round_robin, dtype=int)
        else:
            base_order = np.arange(n_events)

        for ep in range(1, max(1, int(self.epochs)) + 1):
            # Deterministic shuffle per epoch
            rng = np.random.default_rng(1234 + ep)
            if self.balance:
                # Shuffle within balanced order chunks slightly
                idxs = base_order.copy()
                # small block shuffle
                block = 64
                for start in range(0, len(idxs), block):
                    end = min(len(idxs), start + block)
                    idxs[start:end] = rng.permutation(idxs[start:end])
            else:
                idxs = rng.permutation(n_events)
            for i, ix in enumerate(idxs, 1):
                ev = events[ix]
                processed += 1
                # Teacher hint (period guess) with confidence to guide learning
                hint, conf = teacher.hint_with_confidence(ev.get('text', '')) if not self.no_hints else (None, 0.0)
                # Process event with network
                scores = await net.process_event(ev)
                # If teacher provided a hint, positively reinforce the hinted era
                # Only apply hints when no GT period and confidence high enough
                true_raw = ev.get('period') or ev.get('era') or ev.get('time_period')
                has_gt = _canon_period(true_raw) is not None
                if hint and (not has_gt) and (conf >= float(self.hint_threshold)):
                    try:
                        emb = get_enhanced_historical_embedding(ev, net.sbert)
                        x = emb[:384]
                        name = f'era_{hint}'
                        if name in net.era_specialists:
                            # Positive reinforcement proportional to confidence
                            await net.era_specialists[name].update_nlms(x, float(min(1.0, 0.5 + 0.5*conf)))
                    except Exception:
                        pass
                # Argmax era from scores of form {'era_<name>': value}
                if scores:
                    best = max(scores.items(), key=lambda kv: kv[1])[0]
                    pred = best[len('era_'):] if best.startswith('era_') else best
                    true = _canon_period(true_raw)
                    if true is not None:
                        total += 1
                        if str(pred) == str(true):
                            correct += 1
                # Verbose progress
                if self.verbose and (i % 500 == 0 or i == n_events):
                    acc = (correct / total) if total > 0 else 0.0
                    bar_w = 30
                    p = i / max(1, n_events)
                    filled = int(p * bar_w)
                    print(f"[ep {ep}] [{('#'*filled).ljust(bar_w)}] {i}/{n_events} acc={acc:.3f}")
        # Save weights
        try:
            from tools.weights_io import save_network_weights
            counts = save_network_weights(net, self.weights_dir)
        except Exception:
            counts = {}
        acc = (correct / total) if total > 0 else 0.0
        coverage = (total / max(1, processed)) if processed > 0 else 0.0
        return {'events_processed': processed, 'events_scored': total, 'label_coverage': coverage, 'era_accuracy': acc, 'saved': counts}
