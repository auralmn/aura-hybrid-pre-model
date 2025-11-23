#!/usr/bin/env python3
"""
Colab-specific utilities for GPU detection and configuration
"""

import os
import torch
from typing import Dict, Optional

def detect_colab() -> bool:
    """Detect if running in Google Colab"""
    try:
        from google.colab import drive
        return True
    except ImportError:
        return 'COLAB_GPU' in os.environ

def get_gpu_info() -> Dict[str, any]:
    """Get GPU information and capabilities"""
    info = {
        'available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'memory_total_gb': None,
        'memory_allocated_gb': None,
        'memory_reserved_gb': None,
        'compute_capability': None,
        'is_colab': detect_colab(),
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(info['current_device'])
        props = torch.cuda.get_device_properties(info['current_device'])
        info['memory_total_gb'] = props.total_memory / 1e9
        info['memory_allocated_gb'] = torch.cuda.memory_allocated(info['current_device']) / 1e9
        info['memory_reserved_gb'] = torch.cuda.memory_reserved(info['current_device']) / 1e9
        info['compute_capability'] = f"{props.major}.{props.minor}"
    
    return info

def get_model_preset(gpu_type: Optional[str] = None) -> Dict[str, int]:
    """
    Get model size presets for different GPU types
    """
    if gpu_type is None:
        gpu_info = get_gpu_info()
        if not gpu_info['available']:
            gpu_type = 'cpu'
        else:
            name = gpu_info['device_name'].lower()
            if 'l4' in name or 't4' in name:
                gpu_type = 'l4'
            elif 'a100' in name:
                if gpu_info['memory_total_gb'] and gpu_info['memory_total_gb'] >= 70:
                    gpu_type = 'a100_80gb'
                else:
                    gpu_type = 'a100_40gb'
            else:
                gpu_type = 'default'
    
    presets = {
        'l4': {
            'd_model': 512,
            'd_ff': 2048,
            'num_layers': 6,
            'num_heads': 8,
            'max_batch_size': 8,
        },
        'a100_40gb': {
            'd_model': 1024,
            'd_ff': 4096,
            'num_layers': 12,
            'num_heads': 16,
            'max_batch_size': 16,
        },
        'a100_80gb': {
            'd_model': 1536,
            'd_ff': 6144,
            'num_layers': 24,
            'num_heads': 24,
            'max_batch_size': 32,
        },
        'default': {
            'd_model': 768,
            'd_ff': 3072,
            'num_layers': 8,
            'num_heads': 12,
            'max_batch_size': 4,
        },
        'cpu': {
            'd_model': 256,
            'd_ff': 1024,
            'num_layers': 4,
            'num_heads': 4,
            'max_batch_size': 1,
        },
    }
    
    return presets.get(gpu_type, presets['default'])

def setup_colab_environment():
    """Setup Colab environment optimizations"""
    if not detect_colab():
        return
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

def print_gpu_status():
    """Print current GPU status"""
    info = get_gpu_info()
    print(f"GPU Available: {info['available']}")
    if info['available']:
        print(f"Device: {info['device_name']}")
        print(f"Total Memory: {info['memory_total_gb']:.2f} GB")
        print(f"Allocated: {info['memory_allocated_gb']:.2f} GB")
        print(f"Reserved: {info['memory_reserved_gb']:.2f} GB")
        print(f"Compute Capability: {info['compute_capability']}")
    print(f"Colab: {info['is_colab']}")

