import asyncio
import trio
import numpy as np
from typing import List, Dict, Any

from ..core.network import Network
from ..core.neuron import Neuron, MaturationStage
from ..utils.historical_features import get_enhanced_historical_embedding


class HistoricalNetwork(Network):
    def __init__(self, offline: bool = False, device: str | None = None):
        super().__init__()
        self.offline = offline
        try:
            if not offline:
                from sentence_transformers import SentenceTransformer  # type: ignore
                # Respect device if provided (e.g., 'cuda', 'cuda', 'cpu')
                if device:
                    self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                else:
                    self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
            else:
                self.sbert = None
        except Exception:
            self.sbert = None

        # Era specialists
        self.era_specialists: Dict[str, Neuron] = {}
        eras = ['Neolithic','Ancient','Medieval','Early_Modern','Industrial','Modern']
        for era in eras:
            n = Neuron(
                neuron_id=f'era_{era}',
                specialization='era_classifier',
                abilities={'historical': 0.9},
                n_features=384, n_outputs=1,
                maturation=MaturationStage.PROGENITOR
            )
            n.nlms_head.clamp = (0.0, 1.0)
            self.era_specialists[f'era_{era}'] = n

    async def process_event(self, event: Dict[str, Any]) -> Dict[str, float]:
        emb = get_enhanced_historical_embedding(event, self.sbert)
        # Use SBERT slice for NLMS
        x = emb[:384]
        scores: Dict[str, float] = {}
        # Weak supervision by comparing period string
        period = event.get('period')
        for name, n in self.era_specialists.items():
            is_target = 1.0 if (period and name.endswith(str(period))) else 0.0
            y = await n.update_nlms(x, is_target)
            scores[name] = float(y)
        return scores


async def demo_historical():
    print("🗺️  Historical Network Demo")
    net = HistoricalNetwork(offline=True)
    await net.init_weights()

    # Synthetic mini dataset
    events = [
        {"title":"First farming settlements","text":"early agriculture and pottery","impact":"agriculture", "year_start": -8000, "region":"Middle_East","period":"Neolithic","categories":["technology","culture"],"actors":["settlers"]},
        {"title":"Roman Empire","text":"imperial expansion and law","impact":"statecraft", "year_start": -27, "region":"Europe","period":"Ancient","categories":["politics","war"],"actors":["Romans"]},
        {"title":"Industrial Revolution","text":"steam power and factories","impact":"technology", "year_start": 1800, "region":"Europe","period":"Industrial","categories":["technology","trade"],"actors":["engineers"]},
    ]
    # Process
    for ev in events:
        scores = await net.process_event(ev)
        best = max(scores.items(), key=lambda kv: kv[1])
        print(f"{ev['title']}: predicted={best[0]} score={best[1]:.3f} target={ev['period']}")


if __name__ == '__main__':
    trio.run(demo_historical)
