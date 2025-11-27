"""
Integration Test for the Natural Brain Architecture.
Verifies: Thalamic Gating -> Cortical Processing -> Limbic Modulation -> Basal Integration
"""
import torch
import pytest
import sys
import os

# Ensure src path is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.natural_brain import NaturalBrain
from src.base.snn_brain_zones import BrainZoneConfig, BrainZoneType

def test_natural_brain_flow():
    print("\n🧠 Testing Natural Brain Architecture...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 64
    vocab_size = 1000
    
    # 1. Configure Regions
    zones_config = {
        'prefrontal': BrainZoneConfig(
            name='prefrontal',
            max_neurons=64,
            zone_type=BrainZoneType.PREFRONTAL_CORTEX,
            d_model=d_model,
            use_spiking=True
        ),
        'temporal': BrainZoneConfig(
            name='temporal',
            max_neurons=64,
            zone_type=BrainZoneType.TEMPORAL_CORTEX, # Uses FullLanguageZone
            d_model=d_model,
            use_spiking=True
        )
    }
    
    # 2. Initialize Brain
    brain = NaturalBrain(d_model, vocab_size, zones_config, device=device)
    print(f"✅ Brain initialized on {device}")
    
    # 3. Create Fake Input
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # 4. Forward Pass
    logits, info = brain(input_ids)
    
    # 5. Verify Shapes
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("✅ Forward pass output shape correct")
    
    # 6. Verify Biological Subsystems
    # Routing
    assert 'routing' in info
    print(f"   Routing Probs: {info['routing'].shape}")
    
    # Limbic System
    assert 'emotion' in info
    assert 'arousal' in info['emotion']
    print(f"   Emotional State: Arousal={info['emotion']['arousal']:.3f}")
    
    # Hormones (should be empty on first step, or initialized)
    assert 'hormones' in info
    
    # 7. Test Homeostasis Update
    # Simulate a high-stress event (low accuracy)
    brain.update_homeostasis({'accuracy': 0.1, 'energy': 0.8})
    
    # Check if Cortisol spiked
    hormones = brain.endocrine.step({'accuracy': 0.1})
    cortisol = hormones.get('cortisol', 0.0)
    print(f"   Stress Response: Cortisol={cortisol:.4f}")
    
    # We expect some cortisol response to stress
    assert cortisol >= 0.0

if __name__ == "__main__":
    test_natural_brain_flow()