#!/usr/bin/env python3
"""
HuggingFace Dataset Loader for LLM Training
Supports streaming datasets and mixing with vocab_src
"""

from typing import Iterable, List, Optional, Dict, Any, Union
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, Dataset as HFDataset, IterableDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HFDataset = None
    IterableDataset = None
    logger.warning("HuggingFace datasets library not found. Install with `uv pip install datasets`.")


class MixedTextDataset(Dataset):
    """
    Dataset that combines vocab_src files with HuggingFace datasets
    """
    
    def __init__(
        self,
        vocab_src_dir: Optional[str] = None,
        hf_datasets: Optional[List[Dict[str, Any]]] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            vocab_src_dir: Path to vocab_src directory with .txt/.jsonl files
            hf_datasets: List of dicts with 'name' and optional 'config', 'split', 'text_key'
            max_samples: Maximum total samples to load
            shuffle: Whether to shuffle the combined dataset
        """
        self.texts: List[str] = []
        self.vocab_src_dir = vocab_src_dir
        self.hf_datasets = hf_datasets or []
        self.shuffle = shuffle
        
        self._load_data(max_samples)
    
    def _load_data(self, max_samples: Optional[int] = None):
        """Load texts from vocab_src and HF datasets"""
        # Import here to avoid circular imports if any
        from encoders.pretrain_pipeline import iter_texts_from_dir
        
        if self.vocab_src_dir and os.path.isdir(self.vocab_src_dir):
            logger.info(f"Loading local texts from {self.vocab_src_dir}")
            for text in iter_texts_from_dir(self.vocab_src_dir):
                if text.strip():
                    self.texts.append(text.strip())
                    if max_samples and len(self.texts) >= max_samples:
                        return
        
        if HF_AVAILABLE and self.hf_datasets:
            for hf_config in self.hf_datasets:
                if max_samples and len(self.texts) >= max_samples:
                    break
                try:
                    dataset_name = hf_config['name']
                    config = hf_config.get('config')
                    split = hf_config.get('split', 'train')
                    text_key = hf_config.get('text_key', 'text')
                    
                    logger.info(f"Loading HF dataset: {dataset_name}")
                    dataset = load_dataset(dataset_name, config, split=split, streaming=False)
                    
                    for item in dataset:
                        if isinstance(item, dict) and text_key in item:
                            text = item[text_key]
                            if isinstance(text, str) and text.strip():
                                self.texts.append(text.strip())
                                if max_samples and len(self.texts) >= max_samples:
                                    break
                except Exception as e:
                    logger.warning(f"Failed to load HF dataset {hf_config.get('name', 'unknown')}: {e}")
                    continue
        
        if self.shuffle:
            import random
            random.shuffle(self.texts)
            
        logger.info(f"Loaded {len(self.texts)} total texts")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class StreamingMixedDataset(IterableDataset):
    """
    Streaming version that doesn't load all data into memory
    """
    
    def __init__(
        self,
        vocab_src_dir: Optional[str] = None,
        hf_datasets: Optional[List[Dict[str, Any]]] = None,
    ):
        self.vocab_src_dir = vocab_src_dir
        self.hf_datasets = hf_datasets or []
    
    def __iter__(self) -> Iterable[str]:
        from encoders.pretrain_pipeline import iter_texts_from_dir
        
        # 1. Yield from local files first
        if self.vocab_src_dir and os.path.isdir(self.vocab_src_dir):
            for text in iter_texts_from_dir(self.vocab_src_dir):
                if text.strip():
                    yield text.strip()
        
        # 2. Yield from HF datasets
        if HF_AVAILABLE and self.hf_datasets:
            for hf_config in self.hf_datasets:
                try:
                    dataset_name = hf_config['name']
                    config = hf_config.get('config')
                    split = hf_config.get('split', 'train')
                    text_key = hf_config.get('text_key', 'text')
                    
                    dataset = load_dataset(dataset_name, config, split=split, streaming=True)
                    
                    for item in dataset:
                        if isinstance(item, dict) and text_key in item:
                            text = item[text_key]
                            if isinstance(text, str) and text.strip():
                                yield text.strip()
                except Exception as e:
                    logger.warning(f"Failed to stream HF dataset {hf_config.get('name', 'unknown')}: {e}")
                    continue


def create_dataloader(
    vocab_src_dir: Optional[str] = None,
    hf_datasets: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    num_workers: int = 0, # Default 0 for safety, can increase
    shuffle: bool = True,
    streaming: bool = False,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
) -> DataLoader:
    """
    Create optimized DataLoader for training
    """
    if streaming:
        dataset = StreamingMixedDataset(vocab_src_dir, hf_datasets)
        # Shuffle not supported in simple streaming without buffer
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
    else:
        dataset = MixedTextDataset(vocab_src_dir, hf_datasets, max_samples, shuffle)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

