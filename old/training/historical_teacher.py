from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import re


@dataclass
class HistoricalTeacher:
    path: str = 'historical_teacher.md'

    def load(self) -> str:
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ''

    def rules(self) -> Dict[str, set]:
        """Return simple keyword rules per period, extended from the teacher file if available."""
        base: Dict[str, set] = {
            'Neolithic': {'neolithic', 'agriculture', 'farming', 'pottery', 'settlement'},
            'Ancient': {'ancient', 'empire', 'pharaoh', 'dynasty', 'rome', 'greece', 'mesopotamia'},
            'Medieval': {'medieval', 'feudal', 'knight', 'crusade', 'castle', 'monastery'},
            'Early_Modern': {'renaissance', 'reformation', 'exploration', 'colonial', 'printing'},
            'Industrial': {'industrial', 'steam', 'factory', 'railway', 'engine'},
            'Modern': {'modern', 'world war', 'nuclear', 'internet', 'globalization', 'digital'},
        }
        txt = self.load().lower()
        if not txt:
            return base
        # Structured parse: collect bullets under headings per period
        headings = {
            'Neolithic': r"neolithic",
            'Ancient': r"ancient",
            'Medieval': r"medieval",
            'Early_Modern': r"early\s*modern",
            'Industrial': r"industrial",
            'Modern': r"modern",
        }
        # Build sections by splitting on '##'
        sections = re.split(r"^##\s+", txt, flags=re.M)
        for sec in sections:
            # Determine which period this section refers to
            header_match = re.match(r"([a-zA-Z_\s]+)\b", sec)
            if not header_match:
                continue
            header = header_match.group(1).strip()
            for period, pat in headings.items():
                if re.search(rf"\b{pat}\b", header):
                    # Extract bullet lines after Keywords or plain bullets
                    # Prefer lines under a 'keywords' subheading if present
                    # Collect up to 50 tokens
                    kw_set = base[period]
                    # Find bullet lines (- or *) within this sec
                    for line in sec.splitlines():
                        line = line.strip()
                        if line.startswith('-') or line.startswith('*'):
                            # take words from bullet
                            toks = re.findall(r"[a-zA-Z][a-zA-Z_-]+", line)
                            for t in toks:
                                if len(t) > 2:
                                    kw_set.add(t.lower())
                    break
        return base

    def hint_period(self, text: str) -> Optional[str]:
        """Guess period by simple keyword matching using rules(). Returns best period or None."""
        if not text:
            return None
        t = text.lower()
        best = None
        best_score = 0
        for period, kws in self.rules().items():
            score = sum(1 for kw in kws if kw in t)
            if score > best_score:
                best_score = score
                best = period
        return best

    def hint_with_confidence(self, text: str) -> Tuple[Optional[str], float]:
        """Return (period, confidence in [0,1]) based on keyword coverage."""
        if not text:
            return None, 0.0
        t = text.lower()
        best = None
        best_cov = 0.0
        for period, kws in self.rules().items():
            if not kws:
                continue
            hits = sum(1 for kw in kws if kw in t)
            cov = hits / max(1, len(kws))
            if cov > best_cov:
                best_cov = cov
                best = period
        # Scale confidence to be more generous for sparse rule sets
        conf = min(1.0, best_cov * 20.0)
        return best, conf
