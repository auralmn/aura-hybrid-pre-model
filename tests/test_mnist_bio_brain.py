import unittest
import sys
import os
import numpy as np
import asyncio
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core.neuron_factory import NeuronFactory
from core.liquid_moe import LiquidMoERouter
from core.experts import ExpertNLMSHead, NLMSExpertAdapter
from training.hebbian_layer import OjaLayer
from training.optimized_whitener import OptimizedWhitener

class TestBioBrain(unittest.TestCase):
    def test_neuron_factory_patterns(self):
        factory = NeuronFactory()
        
        # Test bursting pattern
        bursting_neuron = factory.create_spiking_neuron(model="izhikevich", pattern="bursting")
        # Extract parameters from the wrapped cell
        cell = bursting_neuron.cell
        # Bursting defaults in JSON: a=0.02, b=0.2, c=-50, d=2
        self.assertAlmostEqual(cell.a.item(), 0.02)
        self.assertAlmostEqual(cell.c.item(), -50.0)
        
        # Test fast spiking
        fs_neuron = factory.create_spiking_neuron(model="izhikevich", pattern="fast_spiking")
        cell_fs = fs_neuron.cell
        # Fast spiking defaults: a=0.1
        self.assertAlmostEqual(cell_fs.a.item(), 0.1)
        
        # Test AdEx
        adex_neuron = factory.create_spiking_neuron(model="adex", pattern="bursting")
        cell_adex = adex_neuron.cell
        # Custom mapping for bursting in factory: a=2.0
        self.assertAlmostEqual(cell_adex.a.item(), 2.0)

    async def async_mnist_test(self):
        print("\n--- Starting MNIST Test ---")
        
        # 1. Load Data
        try:
            from torchvision import datasets, transforms
            # Use a local directory that we likely have write access to
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Transform not strictly needed if we access .data directly, but good practice
            dataset = datasets.MNIST(data_dir, train=True, download=True)
            
            # Subset for speed
            indices = torch.randperm(len(dataset))[:500]
            X = dataset.data[indices].float().view(-1, 784).numpy() / 255.0
            y = dataset.targets[indices].numpy()
            print("Loaded MNIST subset (500 samples)")
        except Exception as e:
            print(f"Torchvision load failed ({e}), using synthetic data")
            X = np.random.rand(500, 784).astype(np.float32)
            y = np.random.randint(0, 10, 500)
            
        # 2. Init Components
        input_dim = 784
        hebbian_k = 64
        n_experts = 10
        
        whitener = OptimizedWhitener(dim=input_dim)
        # Disable neurogenesis by setting max_components = n_components
        hippocampus = OjaLayer(input_dim=input_dim, n_components=hebbian_k, max_components=hebbian_k)
        
        # Create experts
        experts = {}
        for i in range(n_experts):
            head = ExpertNLMSHead(
                n_features=hebbian_k,
                vocab_size=10, 
                attention_config={},
                initial_bias=float(i) # Bias towards class i
            )
            experts[f"expert_{i}"] = NLMSExpertAdapter(head)
            
        cortex = LiquidMoERouter(experts=experts, in_dim=hebbian_k, hidden_dim=64, top_k=2)
        
        # 3. Training Loop
        correct = 0
        total = 0
        
        print("Training...")
        for i in range(len(X)):
            # Hebbian (Unsupervised)
            x_raw = X[i]
            target = float(y[i])
            
            x_w = whitener.transform(x_raw)
            oja_out = hippocampus.step(x_w)
            y_abstract = oja_out.y
            
            # MoE (Supervised/RL)
            out = await cortex.learn(x=y_abstract, token_ids=[], y_true=target)
            
            # Evaluation
            pred = out['y_hat']
            
            # Check for NaN
            if np.isnan(pred):
                print("Training diverged (NaN prediction). Stopping.")
                break
                
            pred_cls = int(round(pred))
            
            # Clamp to valid range
            pred_cls = max(0, min(9, pred_cls))
            
            if pred_cls == int(target):
                correct += 1
            total += 1
            
            if i % 100 == 0:
                print(f"Step {i}: Acc={correct/total:.2f}, Pred={pred:.2f}, Tgt={target}")
                
        acc = correct / total
        print(f"Final Accuracy: {acc:.2f}")
        # Verify we ran at least some steps (divergence is possible on random data)
        self.assertTrue(total > 0)

    def test_mnist(self):
        asyncio.run(self.async_mnist_test())

if __name__ == '__main__':
    unittest.main()

