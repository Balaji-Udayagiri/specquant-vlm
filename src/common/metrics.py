"""Shared timing and memory helpers. Use these everywhere so numbers are comparable."""
import time, torch

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed(fn):
    """Return (result, seconds) with CUDA sync if available."""
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t0

def peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)

def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
