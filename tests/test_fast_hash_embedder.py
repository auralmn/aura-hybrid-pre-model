import torch
from encoders.fast_hash_embedder import FastHashEmbedder


def test_fast_hash_embedder_determinism_and_shape():
    emb = FastHashEmbedder(dim=128)
    v1 = emb.encode("Hello World")
    v2 = emb.encode("Hello World")
    assert v1.shape == (128,)
    assert torch.allclose(v1, v2)
    assert abs(float(v1.norm().item()) - 1.0) < 1e-3


def test_fast_hash_embedder_batch():
    emb = FastHashEmbedder(dim=64)
    out = emb.batch_encode(["abc", "def"]) 
    assert out.shape == (2, 64)

