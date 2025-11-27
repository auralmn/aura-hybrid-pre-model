import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.hippocampal import HippocampalFormation
from src.core.brain import EnhancedBrain

class TestHippocampalBrainIntegration(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_brain_with_hippocampus(self):
        # Create Brain without zones config to test manual hippo injection if needed,
        # or rely on internal initialization
        brain = EnhancedBrain(d_model=64, device=self.device)
        
        # Check internal router exists
        self.assertIsNotNone(brain.global_router)
        
    def test_process_input_flow(self):
        # Setup Brain
        brain = EnhancedBrain(d_model=64, device=self.device)
        
        # Create fake input
        x = torch.randn(2, 10, 64, device=self.device) # [Batch, Seq, Dim]
        
        # Process
        out, info = brain.process_input(x, text_context="testing memory")
        
        # Check output shape
        self.assertEqual(out.shape, x.shape)
        
        # Check info contains routing
        self.assertIn('routing', info)
        self.assertIn('mode', info)
        self.assertEqual(info['mode'], 'neuromorphic_gpu')

if __name__ == '__main__':
    unittest.main()