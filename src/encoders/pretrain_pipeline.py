#!/usr/bin/env python3
"""
Pretraining pipeline to generate and cache FastHash embeddings from text corpora.
Supports jsonl/csv/txt; can save embeddings-only (.pt) plus texts (.json) to avoid pickle warnings.
"""

from typing import Iterable, List, Dict, Optional, Tuple, Callable
import os, json, csv
import torch
from encoders.fast_hash_embedder import FastHashEmbedder
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

def iter_texts_from_dir(path: str, keys: List[str] = ['text','content','sentence','line']) -> Iterable[str]:
    if not os.path.isdir(path):
        return
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        if not os.path.isfile(fpath):
            continue
        low = fname.lower()
        try:
            if low.endswith('.jsonl'):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        for k in keys:
                            if k in obj and isinstance(obj[k], str):
                                yield obj[k]
                                break
            elif low.endswith('.csv'):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:
                            yield str(row[0])
            elif low.endswith('.txt'):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
        except Exception:
            continue

def _encode_text_worker(args: Tuple[int, str, int]) -> Tuple[int, list]:
    """Multiprocessing worker: returns (index, embedding_as_list)."""
    idx, text, dim = args
    emb = FastHashEmbedder(dim=dim)
    v = emb.encode(text)
    return (idx, v.tolist())

def _make_embedder(
    encoder: str,
    dim: int,
    *,
    google_model: str = 'text-embedding-004',
) -> Callable[[str], torch.Tensor]:
    """Return a callable that encodes a single text to a torch vector."""
    enc = (encoder or 'hash').lower()

    if enc == 'google':
        try:
            import os as _os
            import google.generativeai as genai
            api_key = _os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                raise RuntimeError('GOOGLE_API_KEY not set in environment')
            genai.configure(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"google embeddings unavailable: {e}")

        def _encode_google(text: str) -> torch.Tensor:
            try:
                res = genai.embed_content(model=google_model, content=text)
                vec = res.get('embedding') if isinstance(res, dict) else getattr(res, 'embedding', None)
                if vec is None:
                    raise RuntimeError('no embedding field in response')
                v = torch.tensor(vec, dtype=torch.float32)
                if v.numel() == dim:
                    return v
                if v.numel() > dim:
                    return v[:dim]
                out = torch.zeros(dim, dtype=torch.float32)
                out[:v.numel()] = v
                return out
            except Exception:
                return torch.zeros(dim, dtype=torch.float32)

        return _encode_google

    _fh = FastHashEmbedder(dim=dim)
    return lambda text: _fh.encode(text)

def build_embedding_dataset(
    source_dir: str,
    out_path: str,
    dim: int = 1024,
    max_items: Optional[int] = None,
    workers: int = 0,
    use_threads: bool = False,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: Optional[int] = None,
    encoder: str = 'hash',
    google_model: str = 'text-embedding-004',
    separate_texts: bool = False,
    texts_out_path: Optional[str] = None,
) -> Dict[str, object]:
    """Create and save an embedding dataset tensor from a directory of text files."""
    texts: List[str] = []
    for i, text in enumerate(iter_texts_from_dir(source_dir)):
        texts.append(text)
        if max_items is not None and i + 1 >= max_items:
            break

    if not texts:
        return {'num_items': 0, 'path': out_path}

    total = len(texts)
    processed = 0
    last_pct = -1

    def _maybe_report():
        nonlocal last_pct
        if not show_progress and progress_callback is None:
            return
        if progress_callback is not None:
            progress_callback(processed, total)
            return
        pct = int((processed / max(1, total)) * 100)
        if pct != last_pct:
            print(f"Progress: {pct}% ({processed}/{total})")
            last_pct = pct

    embed = _make_embedder(encoder, dim, google_model=google_model)

    def _encode_block(block_indices: List[int], block_texts: List[str]) -> torch.Tensor:
        if workers and workers > 1 and encoder == 'hash' and not use_threads:
            try:
                import os as _os
                cpu = max(1, int((_os.cpu_count() or 1)))
            except Exception:
                cpu = 1
            num_workers = max(1, min(workers, cpu))
            executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
            Xb = torch.zeros((len(block_indices), dim), dtype=torch.float32)
            extra = {}
            if executor_cls is ProcessPoolExecutor:
                try:
                    extra['mp_context'] = mp.get_context('spawn')
                except Exception:
                    pass
            base = block_indices[0] if block_indices else 0
            with executor_cls(max_workers=num_workers, **extra) as ex:
                futures = [ex.submit(_encode_text_worker, (block_indices[i], t, dim)) for i, t in enumerate(block_texts)]
                for fut in as_completed(futures):
                    idx, v_list = fut.result()
                    Xb[idx - base] = torch.tensor(v_list, dtype=torch.float32)
        else:
            if workers and workers > 1:
                with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                    vecs = list(ex.map(embed, block_texts))
            else:
                vecs = [embed(t) for t in block_texts]
            Xb = torch.stack(vecs, dim=0)
        return Xb

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if chunk_size is not None and chunk_size > 0 and total > chunk_size:
        base_dir = os.path.dirname(out_path)
        base_name = os.path.basename(out_path)
        root, ext = os.path.splitext(base_name)
        shards = []
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            block_idx = list(range(start, end))
            block_txt = texts[start:end]
            Xb = _encode_block(block_idx, block_txt)
            processed += len(block_idx)
            _maybe_report()
            shard_stem = os.path.join(base_dir, f"{root}-shard{start//chunk_size:05d}")
            shard_path = f"{shard_stem}{ext or '.pt'}"
            if separate_texts:
                torch.save(Xb, shard_path)
                texts_path = f"{shard_stem}.texts.json"
                with open(texts_path, 'w', encoding='utf-8') as tf:
                    json.dump(block_txt, tf)
                shards.append({'path': shard_path, 'texts_path': texts_path, 'start': start, 'end': end, 'dim': dim})
            else:
                torch.save({'embeddings': Xb, 'texts': block_txt, 'start': start, 'end': end, 'dim': dim}, shard_path)
                shards.append({'path': shard_path, 'start': start, 'end': end, 'dim': dim})
        manifest_path = os.path.join(base_dir, f"{root}.manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({'total': total, 'dim': dim, 'shards': shards}, f)
        return {'num_items': total, 'dim': dim, 'path': manifest_path, 'shards': len(shards)}
    else:
        batch = 10000
        vecs: List[torch.Tensor] = []
        for start in range(0, total, batch):
            end = min(start + batch, total)
            block_idx = list(range(start, end))
            block_txt = texts[start:end]
            Xb = _encode_block(block_idx, block_txt)
            vecs.append(Xb)
            processed += len(block_idx)
            _maybe_report()
        X = torch.cat(vecs, dim=0)
        if separate_texts:
            torch.save(X, out_path)
            tpath = texts_out_path or os.path.splitext(out_path)[0] + '.texts.json'
            with open(tpath, 'w', encoding='utf-8') as tf:
                json.dump(texts, tf)
            return {'num_items': X.shape[0], 'dim': X.shape[1], 'path': out_path, 'texts_path': tpath}
        else:
            torch.save({'embeddings': X, 'texts': texts}, out_path)
            return {'num_items': X.shape[0], 'dim': X.shape[1], 'path': out_path}

def load_embedding_dataset(path: str, texts_path: Optional[str] = None):
    """
    Safe loader that prefers weights_only to avoid pickle code execution.

    Supports:
      - Tensor-only .pt (embeddings) + separate texts JSON
      - Legacy dict .pt with {'embeddings': X, 'texts': [...]}
    Returns: (X: torch.Tensor [N,D], texts: List[str])
    """
    X = None
    texts: List[str] = []
    obj = None
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)  # torch>=2.4
    except TypeError:
        obj = torch.load(path, map_location='cpu')  # older torch fallback

    if isinstance(obj, torch.Tensor):
        X = obj
    elif isinstance(obj, dict) and 'embeddings' in obj:
        X = obj['embeddings']
        if 'texts' in obj:
            texts = obj['texts']

    # texts from JSON if needed
    if not texts:
        if texts_path and os.path.isfile(texts_path):
            try:
                with open(texts_path, 'r', encoding='utf-8') as f:
                    texts = json.load(f)
            except Exception:
                texts = []
        else:
            derived = os.path.splitext(path)[0] + '.texts.json'
            if os.path.isfile(derived):
                try:
                    with open(derived, 'r', encoding='utf-8') as f:
                        texts = json.load(f)
                except Exception:
                    texts = []
    return X, texts

class StreamingTextDataset(torch.utils.data.IterableDataset):
    """Streaming dataset for large-scale training from text files."""
    
    def __init__(self, source_dir: str, dim: int = 1024, encoder: str = 'hash', 
                 batch_size: int = 32, max_len: int = 512):
        self.source_dir = source_dir
        self.dim = dim
        self.encoder_type = encoder
        self.batch_size = batch_size
        self.max_len = max_len
        self.files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) 
                     if os.path.isfile(os.path.join(source_dir, f))]
        
        # Initialize embedder
        if encoder == 'hash':
            self.embedder = FastHashEmbedder(dim=dim)
        else:
            raise NotImplementedError(f"Encoder {encoder} not supported in streaming mode yet")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.files
        
        # Sharding for workers
        if worker_info is not None:
            per_worker = int(len(files) / float(worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker
            if worker_id == worker_info.num_workers - 1:
                end = len(files)
            files = files[start:end]
            
        # Stream content
        for fpath in files:
            try:
                # Naive text reading; for production, use mmap or chunk reading
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # Chunk text into max_len roughly (by words or just char chunks)
                # Here we assume the embedder handles the text chunk
                # For LLM training, usually we yield tokenized windows.
                # Since we use FastHash, we can yield raw text chunks or pre-embedded vectors.
                
                # Yield embeddings directly
                if not text:
                    continue
                    
                # Simple chunking by approximate length
                chunk_size = 2000 # chars
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    if not chunk.strip():
                        continue
                    embedding = self.embedder.encode(chunk)
                    yield embedding
                    
            except Exception:
                continue
