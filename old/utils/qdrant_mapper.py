from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import numpy as np


@dataclass
class QdrantMapper:
    url: str = "http://localhost"
    port: int = 6333
    vector_dim: int = 384

    def __post_init__(self) -> None:
        self.client = None
        self.models = None
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http.models import Distance, VectorParams, PointStruct  # type: ignore
            self._PointStruct = PointStruct
            self._Distance = Distance
            self._VectorParams = VectorParams
            self.client = QdrantClient(url=self.url, port=self.port)
        except Exception:
            self.client = None
        # Collection names
        self.collections = {
            'neurons': 'bio_neurons',
            'regions': 'bio_regions',
            'routing': 'bio_routing',
        }

    def ensure_collections(self) -> None:
        if self.client is None:
            print("Qdrant client not available; skip ensure_collections().")
            return
        for name in self.collections.values():
            try:
                self.client.recreate_collection(
                    collection_name=name,
                    vectors_config=self._VectorParams(size=self.vector_dim, distance=self._Distance.COSINE),
                )
                print(f"✓ Qdrant collection ready: {name}")
            except Exception as e:
                print(f"! Qdrant collection setup issue for {name}: {e}")

    def _pad(self, xs: List[float]) -> List[float]:
        if len(xs) < self.vector_dim:
            xs = xs + [0.0] * (self.vector_dim - len(xs))
        return xs[: self.vector_dim]

    def _get_region_type(self, region: str) -> str:
        """Map region names to their functional types"""
        region_types = {
            'thalamus': 'sensory_relay',
            'hippocampus': 'memory_formation',
            'amygdala': 'emotional_processing',
            'router': 'routing_decision',
            'cns': 'central_coordination'
        }
        return region_types.get(region, 'unknown')

    def map_neuron(self, neuron: Any, region: str) -> Dict[str, Any]:
        feats: List[float] = []
        # Basic neuron status features
        try:
            feats.extend([
                float(getattr(neuron.maturation, 'value', 0)),
                float(getattr(neuron.activity, 'value', 0)),
                float(getattr(neuron, 'membrane_potential', 0.0)),
                float(len(getattr(neuron, 'synapses', []) or [])),
                float(len(getattr(neuron, 'spike_history', []) or [])),
            ])
        except Exception:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        # Abilities (sparse)
        abilities = getattr(neuron, 'abilities', {}) or {}
        keys = ['memory','gating','threat_assessment','emotional_salience','classification','routing']
        feats.extend([float(abilities.get(k, 0.0)) for k in keys])
        # Append weight summary (mean/std) if available
        try:
            w = getattr(getattr(neuron, 'nlms_head', None), 'w', None)
            if w is not None and hasattr(w, 'shape'):
                feats.extend([float(np.mean(w)), float(np.std(w))])
        except Exception:
            pass
        feats = self._pad([float(x) for x in feats])
        # Get neuron type and status information
        maturation = getattr(neuron, 'maturation', None)
        activity = getattr(neuron, 'activity', None)
        
        payload = {
            'neuron_id': getattr(neuron, 'neuron_id', 'unknown'),
            'region': region,
            'region_type': self._get_region_type(region),
            'specialization': getattr(neuron, 'specialization', ''),
            'neuron_type': 'neuron',
            'maturation_stage': maturation.name if maturation else 'unknown',
            'activity_state': activity.name if activity else 'unknown',
            'membrane_potential': float(getattr(neuron, 'membrane_potential', 0.0)),
            'synapse_count': len(getattr(neuron, 'synapses', []) or []),
            'spike_count': len(getattr(neuron, 'spike_history', []) or []),
            'abilities': abilities,
            'primary_ability': max(abilities.items(), key=lambda x: x[1])[0] if abilities else 'none',
            'timestamp': datetime.now().isoformat(),
        }
        return {'id': str(uuid.uuid4()), 'vector': feats, 'payload': payload}

    def map_region(self, name: str, obj: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        feats: List[float] = []
        neurons = getattr(obj, 'neurons', []) or []
        feats.append(float(len(neurons)))
        try:
            feats.append(float(np.mean([getattr(n, 'membrane_potential', 0.0) for n in neurons])) if neurons else 0.0)
        except Exception:
            feats.append(0.0)
        # region-specific knobs
        if name == 'thalamus':
            feats.append(float(getattr(getattr(obj, 'thalamus_relay', None), 'gating_strength', 0.7)))
        elif name == 'hippocampus':
            feats.append(float(getattr(obj, 'neurogenesis_rate', 0.01)))
        elif name == 'amygdala':
            feats.append(float(getattr(obj, 'stress_level', 0.0)))
        else:
            feats.append(0.0)
        feats = self._pad(feats)
        
        # Calculate region statistics
        active_neurons = sum(1 for n in neurons if getattr(n, 'activity', None) and getattr(n.activity, 'name', '') == 'ACTIVE')
        avg_membrane_potential = np.mean([getattr(n, 'membrane_potential', 0.0) for n in neurons]) if neurons else 0.0
        
        payload = {
            'region_name': name,
            'region_type': self._get_region_type(name),
            'object_type': type(obj).__name__,
            'neuron_count': len(neurons),
            'active_neuron_count': active_neurons,
            'avg_membrane_potential': float(avg_membrane_potential),
            'activity_ratio': float(active_neurons / len(neurons)) if neurons else 0.0,
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
        }
        return {'id': str(uuid.uuid4()), 'vector': feats, 'payload': payload}

    def map_routing(self, decision: Dict[str, Any], query_vec: np.ndarray) -> Dict[str, Any]:
        base = query_vec.astype(np.float32).reshape(-1)
        vec = base.tolist()
        tail = [
            float(decision.get('routing_confidence', 0.0)),
            float(bool(decision.get('needs_multiple_specialists', False))),
            float(len(decision.get('secondary_targets', []))),
            float(decision.get('query_characteristics', {}).get('complexity_score', 0.0)),
        ]
        # place tail at the end
        vec[-len(tail):] = tail
        # Extract routing information
        primary_target = decision.get('primary_target', 'unknown')
        routing_strategy = decision.get('routing_strategy', 'unknown')
        query_characteristics = decision.get('query_characteristics', {})
        
        payload = {
            'primary_target': primary_target,
            'routing_strategy': routing_strategy,
            'routing_type': 'routing_decision',
            'confidence': float(decision.get('routing_confidence', 0.0)),
            'needs_multiple_specialists': bool(decision.get('needs_multiple_specialists', False)),
            'secondary_targets_count': len(decision.get('secondary_targets', [])),
            'complexity_score': float(query_characteristics.get('complexity_score', 0.0)),
            'query_type': query_characteristics.get('query_type', 'unknown'),
            'timestamp': datetime.now().isoformat(),
        }
        return {'id': str(uuid.uuid4()), 'vector': self._pad(vec), 'payload': payload}

    def upsert_points(self, collection: str, points: List[Dict[str, Any]]) -> None:
        if self.client is None:
            print(f"Qdrant client not available; would upsert {len(points)} points to {collection}.")
            return
        PointStruct = self._PointStruct
        ps = [PointStruct(id=p['id'], vector=p['vector'], payload=p['payload']) for p in points]
        self.client.upsert(collection_name=collection, points=ps)

    def snapshot_network(self, network: Any) -> None:
        # Regions
        regions = {
            'thalamus': getattr(network, '_thalamus', None),
            'hippocampus': getattr(network, '_hippocampus', None),
            'amygdala': getattr(network, '_amygdala', None),
            'router': getattr(network, '_thalamic_router', None),
        }
        region_points: List[Dict[str, Any]] = []
        neuron_points: List[Dict[str, Any]] = []
        for name, obj in regions.items():
            if obj is None:
                continue
            region_points.append(self.map_region(name, obj))
            for n in getattr(obj, 'neurons', []) or []:
                neuron_points.append(self.map_neuron(n, name))
        if region_points:
            self.upsert_points(self.collections['regions'], region_points)
        if neuron_points:
            # Upsert in chunks
            for i in range(0, len(neuron_points), 256):
                self.upsert_points(self.collections['neurons'], neuron_points[i:i+256])

