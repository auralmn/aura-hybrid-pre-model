import torch
import gc
import logging
import time

logger = logging.getLogger(__name__)

# Throttle cache clearing to avoid thrashing (min seconds between clears)
_LAST_CLEAR_TIME = 0
_CLEAR_INTERVAL = 60  # seconds

def maybe_empty_cuda_cache(reason: str = "", min_free_ratio: float = 0.10) -> None:
    """
    Smart CUDA cache management.
    Only clears cache if:
    1. Free VRAM is critically low (< min_free_ratio)
    2. Sufficient time has passed since last clear (throttling)
    
    Args:
        reason: Log message explaining why
        min_free_ratio: Threshold (0.10 = 10% free memory)
    """
    global _LAST_CLEAR_TIME
    
    if not torch.cuda.is_available():
        return

    try:
        # Get memory info (fast, non-blocking)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_ratio = free_bytes / float(total_bytes)
        
        # Only clear if we are in the "Danger Zone"
        if free_ratio < min_free_ratio:
            current_time = time.time()
            if current_time - _LAST_CLEAR_TIME < _CLEAR_INTERVAL:
                # Throttled - skip clear to preserve throughput
                return
                
            tag = f" ({reason})" if reason else ""
            logger.warning(f"⚠️ Low VRAM ({free_ratio:.1%}). Clearing cache{tag}...")
            
            # 1. Sync and collect garbage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            
            # 2. Release caching allocator memory
            torch.cuda.empty_cache()
            
            _LAST_CLEAR_TIME = time.time()
            
            # Log result
            new_free, _ = torch.cuda.mem_get_info()
            logger.info(f"   -> Freed. New capacity: {new_free/1e9:.2f} GB")
            
    except Exception as exc:
        # Don't crash training on memory check failure
        pass

def get_memory_stats() -> dict:
    """Get current GPU memory statistics for logging."""
    if not torch.cuda.is_available():
        return {"available": False}
        
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    
    return {
        "available": True,
        "total_gb": total / 1e9,
        "free_gb": free / 1e9,
        "allocated_gb": allocated / 1e9,
        "reserved_gb": reserved / 1e9,
        "utilization": 1.0 - (free / total)
    }