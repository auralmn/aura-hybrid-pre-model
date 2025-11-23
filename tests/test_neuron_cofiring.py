import torch
import pytest

from base.brain_zone_factory import create_prefrontal_cortex


def test_neuron_cofiring_within_zone_timealigned():
    torch.manual_seed(0)
    zone = create_prefrontal_cortex(d_model=64, max_neurons=64)

    B, T, D = 2, 20, 64
    # Positive, constant drive to encourage synchronized activity across groups
    x = torch.ones(B, T, D) * 0.8

    out, activity = zone.process(x)
    assert out.shape == (B, T, D)
    assert isinstance(activity, dict)

    # Extract per-step activity histories from two neuron groups
    ng = zone.neuromorphic_processor.neuron_groups
    keys = list(ng.keys())
    assert len(keys) >= 2
    g1 = ng[keys[0]]
    g2 = ng[keys[1]]

    # First T entries correspond to this fresh run
    h1 = g1.activity_history[:T].clone()
    h2 = g2.activity_history[:T].clone()

    # Co-firing count: steps where both groups emitted spikes
    both = (h1 > 0) & (h2 > 0)
    cofiring_steps = int(both.sum().item())

    # Require some simultaneous spikes (at least 20% of steps)
    assert cofiring_steps >= max(1, T // 5)


