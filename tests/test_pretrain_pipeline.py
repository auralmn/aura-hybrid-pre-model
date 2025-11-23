import os, json, tempfile
import torch
from encoders.pretrain_pipeline import build_embedding_dataset


def test_build_embedding_dataset_jsonl(tmp_path: tempfile.TemporaryDirectory):
    d = tmp_path / 'data'
    d.mkdir()
    p = d / 'sample.jsonl'
    with open(p, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'text': 'hello world'}) + '\n')
        f.write(json.dumps({'text': 'another line'}) + '\n')

    out = tmp_path / 'emb.pt'
    info = build_embedding_dataset(str(d), str(out), dim=64)
    assert info['num_items'] == 2
    saved = torch.load(out, weights_only=True)
    X = saved['embeddings']
    assert X.shape == (2, 64)

