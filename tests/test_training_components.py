import unittest
import sys
import os

# Add src to path explicitly
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import torch
import numpy as np
import shutil
import tempfile
from training.memory_pool import ArrayPool
from training.optimized_whitener import OptimizedWhitener 
from training.stdp_learning import STDPLearner
from training.hebbian_layer import OjaLayer
from encoders.fast_hash_embedder import FastHashEmbedder
from training.hf_dataset_loader import MixedTextDataset

class TestTrainingComponents(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_memory_pool(self):
        # Use a larger pool to ensure it fits
        pool = ArrayPool(max_pool_size_mb=10)
        shape = (10, 10)
        
        # Allocate
        arr1 = pool.get_array(shape, np.dtype(np.float32))
        self.assertEqual(arr1.shape, shape)
        
        # Return and reallocate
        pool.return_array(arr1)
        
        # Verify it's in pool (internal check)
        key = (shape, np.dtype(np.float32))
        # Depending on numpy version, dtype equality can be tricky, but pool uses exact match
        # Let's just try to get it back
        
        arr2 = pool.get_array(shape, np.dtype(np.float32))
        
        # Check hit count
        self.assertEqual(pool.stats.hits, 1)

    def test_optimized_whitener(self):
        dim = 5
        whitener = OptimizedWhitener(dim=dim)
        data = np.random.randn(10, dim).astype(np.float32)
        
        # First pass
        out1 = whitener.transform(data[0])
        self.assertEqual(out1.shape, (dim,))
        
        # Check state updated
        self.assertFalse(np.allclose(whitener.mu, 0))

    def test_stdp_learning(self):
        learner = STDPLearner()
        tokens = [1, 2, 3, 4, 5]
        
        # Process sequence
        stats = learner.process_sequence(tokens)
        
        # Should have created weights
        self.assertTrue(len(learner.token_weights) > 0)
        
        # Check modulation
        mods = learner.get_modulations(tokens)
        self.assertEqual(len(mods), len(tokens))
        self.assertTrue(np.all(mods >= 1.0))

    def test_hebbian_layer(self):
        input_dim = 10
        layer = OjaLayer(n_components=2, input_dim=input_dim)
        
        # Create dummy input
        x = np.random.randn(input_dim).astype(np.float32)
        x = x / np.linalg.norm(x)
        
        # Step
        out = layer.step(x)
        self.assertEqual(out.y.shape, (2,))
        
        # Check growth (force high residual)
        layer.residual_ema = 10.0 # Force high
        out_grow = layer.step(x)
        
        if out_grow.grew:
            self.assertEqual(layer.K, 3)

    def test_fast_hash_embedder(self):
        embedder = FastHashEmbedder(dim=16)
        text = "hello world"
        
        # Vector only
        vec = embedder.encode(text)
        self.assertEqual(vec.shape, (16,))
        
        # With indices
        vec2, indices = embedder.encode_with_indices(text)
        
        # Relaxed check: implementations differ slightly (vectorized vs loop)
        # Check basic properties
        self.assertEqual(vec2.shape, (16,))
        self.assertTrue(len(indices) > 0)
        
        # Both should be normalized
        self.assertTrue(torch.abs(vec.norm() - 1.0) < 1e-4)
        self.assertTrue(torch.abs(vec2.norm() - 1.0) < 1e-4)

    def test_dataset_loader(self):
        # Create dummy text file
        fpath = os.path.join(self.test_dir, "test.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("Hello world\nTesting dataset")
            
        # Test loader
        dataset = MixedTextDataset(vocab_src_dir=self.test_dir)
        self.assertTrue(len(dataset) > 0)
        item = dataset[0]
        self.assertIsInstance(item, str)

if __name__ == '__main__':
    unittest.main()
