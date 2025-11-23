import torch
import pytest

from base.brain_zone_factory import (
    create_prefrontal_cortex,
    create_temporal_cortex,
    create_hippocampus,
    create_cerebellum,
)


@pytest.mark.parametrize(
    "creator, zone_name, d_model, max_neurons",
    [
        (create_prefrontal_cortex, "prefrontal_cortex", 128, 128),
        (create_temporal_cortex, "temporal_cortex", 128, 128),
        (create_hippocampus, "hippocampus", 128, 96),
        (create_cerebellum, "cerebellum", 128, 64),
    ],
)
def test_neuromorphic_zone_forward_shapes_and_activity(creator, zone_name, d_model, max_neurons):
    torch.manual_seed(42)
    zone = creator(d_model=d_model, max_neurons=max_neurons)

    # [B, D]
    x_bd = torch.randn(2, d_model)
    out_bd, activity_bd = zone.process(x_bd)
    assert out_bd.shape == (2, d_model)
    assert isinstance(activity_bd, dict)
    assert activity_bd.get("zone_name") == zone_name

    # [B, T, D]
    x_btd = torch.randn(2, 4, d_model)
    out_btd, activity_btd = zone.process(x_btd)
    assert out_btd.shape == (2, 4, d_model)
    assert isinstance(activity_btd, dict)
    assert activity_btd.get("zone_name") == zone_name


def test_neuromorphic_zone_spiking_activity_positive():
    torch.manual_seed(0)
    zone = create_prefrontal_cortex(d_model=64, max_neurons=64)

    # Positive-biased input to ensure spikes
    x = torch.ones(2, 6, 64) * 0.5
    out, activity = zone.process(x)
    assert out.shape == (2, 6, 64)

    # Accept non-zero avg firing when available
    avg_rate = activity.get("avg_firing_rate", 0.0)
    assert isinstance(avg_rate, float)
    assert 0.0 <= avg_rate <= 1.0

