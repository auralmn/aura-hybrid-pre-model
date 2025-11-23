import unittest
import sys
import os
import numpy as np
import asyncio

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core.liquid_moe import LiquidMoERouter, LiquidGatingNetwork, BanditGating, LiquidCell
from core.experts import ExpertNLMSHead, NLMSExpertAdapter

class TestLiquidMoE(unittest.TestCase):
    def setUp(self):
        self.in_dim = 10
        self.hidden_dim = 8
        self.n_experts = 3
        
        # Create mock experts
        self.experts = {}
        for i in range(self.n_experts):
            head = ExpertNLMSHead(
                n_features=self.in_dim,
                vocab_size=100,
                attention_config={},
                initial_bias=float(i)
            )
            self.experts[f"expert_{i}"] = NLMSExpertAdapter(head)

    def test_liquid_cell(self):
        cell = LiquidCell(in_dim=self.in_dim, hidden_dim=self.hidden_dim)
        x = np.random.randn(self.in_dim)
        h = cell.step(x)
        self.assertEqual(h.shape, (self.hidden_dim,))
        self.assertFalse(np.allclose(h, 0.0))

    def test_bandit_gating(self):
        # Use low exploration to test exploitation
        bandit = BanditGating(n_experts=self.n_experts, exploration_factor=0.1)
        
        # Update all experts to level the playing field regarding count
        bandit.update(0, error=0.0) # Perfect reward 1.0
        bandit.update(1, error=10.0) # Poor reward ~0.1
        bandit.update(2, error=10.0) # Poor reward ~0.1
        
        scores = bandit.get_ucb_scores()
        self.assertTrue(scores[0] > scores[1])
        self.assertTrue(scores[0] > scores[2])
        
        base_gates = np.ones(self.n_experts) / self.n_experts
        idx, gates = bandit.select_top_k(k=2, base_gates=base_gates)
        
        self.assertEqual(len(idx), 2)
        self.assertEqual(idx[0], 0) # Expert 0 should be top

    async def async_test_router(self):
        router = LiquidMoERouter(
            experts=self.experts,
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            top_k=2
        )
        
        x = np.random.randn(self.in_dim)
        
        # Route
        out = await router.route(x)
        self.assertIn('y_hat', out)
        self.assertEqual(len(out['topk']), 2)
        
        # Learn
        learn_out = await router.learn(x, token_ids=[], y_true=1.0)
        self.assertIn('y_hat', learn_out)

    def test_router(self):
        asyncio.run(self.async_test_router())

if __name__ == '__main__':
    unittest.main()
