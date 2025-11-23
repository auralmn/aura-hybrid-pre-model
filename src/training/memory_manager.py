import torch
import gc
import logging

logger = logging.getLogger(__name__)

def maybe_empty_cuda_cache(reason: str = "", min_free_ratio: float = 0.12) -> None:
    """
    Smart CUDA cache management.
    Only clears cache if free VRAM drops below a threshold ratio.
    Prevents performance thrashing from frequent empty_cache calls.
    
    Args:
        reason: Log message explaining why cache might be cleared
        min_free_ratio: Minimum ratio of free memory (0.0-1.0) before clearing
    """
    if not torch.cuda.is_available():
        return

    try:
        # Get memory info
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_ratio = free_bytes / float(total_bytes)
        
        if free_ratio < min_free_ratio:
            tag = f" ({reason})" if reason else ""
            logger.warning(f"🔄 Clearing CUDA cache{tag}. Free VRAM ratio: {free_ratio:.3f}")
            
            # Force garbage collection first
            gc.collect()
            torch.cuda.empty_cache()
            
            # Log improvement
            new_free, _ = torch.cuda.mem_get_info()
            new_ratio = new_free / float(total_bytes)
            logger.info(f"   -> Cache cleared. New free ratio: {new_ratio:.3f}")
            
    except Exception as exc:
        logger.warning(f"⚠️ Unable to query CUDA memory for cleanup: {exc}")

def get_memory_stats() -> dict:
    """Get current GPU memory statistics"""
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

