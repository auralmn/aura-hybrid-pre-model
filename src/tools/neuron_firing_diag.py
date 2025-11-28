#!/usr/bin/env python3
"""
Neuron firing diagnostic tool.
Run manually to verify spiking and end-to-end routing with adjusted parameters.
"""

import torch

from base.neuron import AdaptiveLIFNeuron
from base.events import EventBus

# Optional brain-level test
try:
    from core.brain import create_aura_brain
except Exception:
    create_aura_brain = None  # type: ignore


def main() -> None:
    print("TESTING NEURON FIRING WITH OPTIMIZED PARAMETERS...")

    event_bus = EventBus()
    counter = {"n": 0}

    def count_firings(event):
        counter["n"] += 1
        print(f"NEURON FIRED #{counter['n']}: {event.data}")

    event_bus.subscribe('neuron_fired', count_firings)

    # Lower threshold, reduced leakage, gentler slope
    neuron = AdaptiveLIFNeuron(beta=0.8, threshold=0.3, init_slope=15.0, event_bus=event_bus, name="diag_neuron")

    print("Testing single neuron with strong inputs...")
    strong_input = torch.tensor([[5.0]])

    for step in range(10):
        spikes, membrane = neuron.forward(strong_input)
        s = float(spikes.item())
        m = float(membrane.item())
        print(f"Step {step}: Input={strong_input.item():.1f}, Membrane={m:.3f}, Spike={s:.3f}")
        if s > 0.5:
            print(f"  Fired at step {step}!")

    print(f"\nTotal firings detected (single neuron): {counter['n']}")

    # Optional: end-to-end brain routing test with amplified inputs
    if create_aura_brain is not None:
        print("\nTesting end-to-end brain routing with amplified inputs...")
        try:
            brain = create_aura_brain(d_model=128, use_neuromorphic=True)
            # Subscribe to neuron_fired at the brain level
            zone_counter = {"n": 0}
            def on_zone_fire(ev):
                zone_counter["n"] += 1
            brain.event_bus.subscribe('neuron_fired', on_zone_fire)

            x = torch.randn(2, 7, 128) * 1.5  # amplify inputs
            print(f"Input amplitude approx: mean={float(x.abs().mean()):.3f}, max={float(x.abs().max()):.3f}")
            out, info = brain.process_input(x, text_context='analyze memory language')
            print(f"Routing mode: {info.get('mode')}")
            print(f"Zone events observed: {zone_counter['n']}")
            print(f"Output shape: {tuple(out.shape)}")
        except Exception as e:
            print(f"End-to-end test error: {e}")


if __name__ == "__main__":
    main()
