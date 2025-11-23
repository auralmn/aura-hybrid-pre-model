#!/usr/bin/env python3
"""
Continuous Learning Runner

- Creates a neuromorphic brain
- Attaches RSS-based continuous learning (if available)
- Starts the learning loop and logs key events

Run:
  python -m src.tools.continuous_learning_runner
"""

import asyncio
import logging

from core.brain import create_aura_brain


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    brain = create_aura_brain(d_model=512, use_neuromorphic=True)

    # Attach orchestrator and subscribe to interesting events
    brain.attach_continuous_learning()

    # Sanity checks
    orc = getattr(brain, "learning_orchestrator", None)
    proc = getattr(brain, "neuromorphic_processor", None)
    print("Orchestrator present:", bool(orc))
    print("Neuromorphic processor present:", bool(proc))
    if proc:
        print("Zones wired:", list((proc.zone_processors or {}).keys()))

    # Enable local corpus if available (guard for older versions)
    if orc and hasattr(orc, "set_vocab_dir"):
        orc.set_vocab_dir("vocab_src")

    print("Starting continuous learning (Ctrl+C to stop)...")
    await brain.start_continuous_learning()

    # Force an immediate save so brain_states/ appears without waiting
    import os, asyncio
    await asyncio.sleep(2)
    try:
        os.makedirs("brain_states", exist_ok=True)
        # Try orchestrator helper first
        if orc and hasattr(orc, "_save_homeostasis_all"):
            orc._save_homeostasis_all()
        # Also save directly via processor zones in case orchestrator hasn't started yet
        if proc and getattr(proc, "zone_processors", None):
            for zname, zone in proc.zone_processors.items():
                neu = getattr(zone, "neuromorphic_processor", None)
                if neu and hasattr(neu, "save_homeostasis_state"):
                    neu.save_homeostasis_state(os.path.join("brain_states", f"{zname}_homeostasis.json"))
    except Exception as e:
        print("Immediate save error:", e)
    print("brain_states dir:", os.listdir("brain_states") if os.path.isdir("brain_states") else "not created")

    # Optional: subscribe simple logging handlers
    def on_content_processed(ev):
        data = ev.data
        #print(data)
        print(f"CONTENT: {data.get('category','?')} time={data.get('processing_time',0):.3f}s total_activation={data.get('brain_response',{}).get('total_activation',0):.3f}")

    def on_neuron_fired(ev):
       #print(ev.data)
        print(f"FIRE: zone={ev.data.get('zone','?')} rate={ev.data.get('firing_rate_ema',0):.3f}")

    brain.event_bus.subscribe('content_processed', on_content_processed)
    brain.event_bus.subscribe('neuron_fired', on_neuron_fired)

    print("Continuous learning running... (Ctrl+C to stop)")
    try:
        await brain.start_continuous_learning()
        # Keep running
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await brain.stop_continuous_learning()


if __name__ == "__main__":
    asyncio.run(main())
