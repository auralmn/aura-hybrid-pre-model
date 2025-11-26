import os
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from .qdrant_mapper import QdrantMapper


@dataclass
class QdrantStreamer:
    url: str = os.getenv('QDRANT_URL', 'http://localhost')
    port: int = int(os.getenv('QDRANT_PORT', '6333'))

    def __post_init__(self) -> None:
        self.mapper = QdrantMapper(url=self.url, port=self.port)
        self.mapper.ensure_collections()
        self._counter = 0

    def maybe_snapshot(self, network: Any, every: int = 50) -> None:
        self._counter += 1
        if self._counter % every == 0:
            self.mapper.snapshot_network(network)

    def upsert_routing(self, decision: Dict[str, Any], query_vec: np.ndarray) -> None:
        pt = self.mapper.map_routing(decision, query_vec)
        self.mapper.upsert_points(self.mapper.collections['routing'], [pt])

