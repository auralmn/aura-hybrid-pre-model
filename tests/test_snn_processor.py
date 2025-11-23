import torch
import pytest

from base.snn_processor import NeuromorphicProcessor
from base.brain_zone_factory import (
    create_prefrontal_cortex,
    create_temporal_cortex,
)


def test_processor_routing_and_combination_keyword():
    torch.manual_seed(0)
    proc = NeuromorphicProcessor(d_model=64)
    zones = {
        'prefrontal_cortex': create_prefrontal_cortex(d_model=64, max_neurons=64),
        'temporal_cortex': create_temporal_cortex(d_model=64, max_neurons=64),
    }
    proc.set_zone_processors(zones)

    x = torch.randn(2, 5, 64)
    out = proc.process(x, {'text': 'analyze and create language'})
    assert out.shape == (2, 5, 64)
    # Zones were set; output shape suffices here


def test_processor_fallback_basic_when_no_zones():
    proc = NeuromorphicProcessor(d_model=32)
    x = torch.randn(3, 32)
    out = proc.process(x, {'text': 'unknown'})
    # With no zones set, should get a basic-processed tensor of same last dim
    assert out.shape[-1] == 32

