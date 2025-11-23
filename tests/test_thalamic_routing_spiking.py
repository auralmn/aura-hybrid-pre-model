import torch

from base.snn_processor import NeuromorphicProcessor
from base.brain_zone_factory import create_cerebellum


def test_thalamic_keyword_routing_and_zone_spiking():
    d_model = 64
    proc = NeuromorphicProcessor(d_model=d_model)

    # Install a cerebellum zone (precision/fine-tuning keywords map here)
    zone = create_cerebellum(d_model=d_model, max_neurons=d_model)
    proc.set_zone_processors({'cerebellum': zone})

    text = "please calculate this sum quickly"  # should route to cerebellum
    zones = proc.content_router.route_text_to_zones(text)
    assert 'cerebellum' in zones

    # Positive-biased stimulus to ensure spikes
    x = torch.ones(2, 10, d_model) * 0.5
    out, activity = zone.process(x, context={'text': text})
    assert out.shape == (2, 10, d_model)
    assert isinstance(activity, dict)
    avg_rate = float(activity.get('avg_firing_rate', 0.0))
    assert 0.0 <= avg_rate <= 1.0
    # Expect some spiking under positive drive
    assert avg_rate > 0.0


