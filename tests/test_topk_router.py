import torch
import pytest

from base.snn_processor import NeuromorphicProcessor
from base.brain_zone_factory import (
    create_prefrontal_cortex,
    create_temporal_cortex,
)
class _StubTopKRouter:
    """Minimal stub to avoid external deps. Returns fixed top-1 selection and gate.
    Interface: forward(x) -> (indices, gates, aux)
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 1, *args, **kwargs):
        self.num_experts = num_experts
        self.top_k = max(1, top_k)

    def __call__(self, x):
        import torch
        B,S,_ = x.shape
        # pick expert 0 with gate 1.0
        idx = torch.zeros(B,S,self.top_k, dtype=torch.long)
        gates = torch.ones(B,S,self.top_k, dtype=x.dtype)
        aux = None
        return idx, gates, aux


def test_topk_router_plan_and_output_shape():
    d_model = 16
    proc = NeuromorphicProcessor(d_model=d_model)
    zones = {
        'prefrontal_cortex': create_prefrontal_cortex(d_model=d_model, max_neurons=d_model),
        'temporal_cortex': create_temporal_cortex(d_model=d_model, max_neurons=d_model),
    }
    proc.set_zone_processors(zones)

    # Inject topk router
    proc.set_router_mode('topk')
    # Inject stubbed router to avoid external deps
    proc._topk_router = _StubTopKRouter(d_model=d_model, num_experts=len(zones), top_k=1)
    proc._router_zone_names = list(zones.keys())

    x = torch.randn(2, 5, d_model)
    out, info = proc.run_plan(x, text="irrelevant", intents=None, context=None, top_k=1)
    assert out.shape == (2, 5, d_model)
    assert info.get('mode') == 'neuromorphic'
    assert len(info.get('selected_zones', [])) >= 1

