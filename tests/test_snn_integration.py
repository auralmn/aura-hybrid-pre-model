import torch
import pytest

from core.brain import create_aura_brain


def test_create_aura_brain_neuromorphic_flow_shapes():
    brain = create_aura_brain(d_model=128, use_neuromorphic=True)
    x = torch.randn(2, 7, 128)
    out, info = brain.process_input(x, text_context='analyze memory language')
    assert out.shape == (2, 7, 128)
    assert info.get('mode') == 'neuromorphic'


def test_brain_statistics_updates_with_zone_activity():
    brain = create_aura_brain(d_model=64, use_neuromorphic=True)
    x = torch.ones(1, 4, 64) * 0.4
    out, info = brain.process_input(x, text_context='analyze')
    stats = brain.get_brain_statistics()
    assert stats.num_zones >= 1
    # avg_firing_rate should be a float within [0,1]
    assert isinstance(stats.avg_firing_rate, float)
    assert 0.0 <= stats.avg_firing_rate <= 1.0

