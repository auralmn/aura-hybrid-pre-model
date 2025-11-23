import torch

from base.neuron import IzhikevichNeuron, load_izhikevich_presets, load_izhikevich_patterns_json, create_izhikevich_from_pattern, simulate_izhikevich


def test_izhikevich_tonic_spiking_minimal():
    # Tonic spiking preset: a=0.02, b=0.2, c=-65, d=6, I ~ 14
    izh = IzhikevichNeuron(a=0.02, b=0.2, c=-65, d=6, dt=0.2)
    T = 200
    I = torch.full((T,), 14.0)
    spk = izh(I)
    # Expect some spikes
    assert spk.sum().item() > 0


def test_izhikevich_load_presets():
    presets = load_izhikevich_presets('pattern.csv')
    assert isinstance(presets, dict)
    assert 'regular spiking (rs)' in presets or 'regular spiking pyramidal' in presets


def test_izhikevich_json_patterns():
    patterns = load_izhikevich_patterns_json('izhikevich_23_firing_patterns.json')
    assert isinstance(patterns, dict) and len(patterns) > 0
    # Try to instantiate a known pattern name if present
    name = next(iter(patterns.keys()))
    izh = create_izhikevich_from_pattern(name, patterns)
    spk = simulate_izhikevich(izh, T=100, I=patterns[name].get('I', 10.0))
    assert spk.numel() == 100

