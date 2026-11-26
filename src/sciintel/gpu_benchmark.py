from __future__ import annotations

import time


def run_gpu_benchmark(seconds: int = 20, matrix_size: int = 3072) -> dict:
    """A short sustained CUDA workload to verify compute, not just CUDA detection."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    started = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started < seconds:
        c = a @ b
        # Force the result to be consumed so kernels are not optimized away.
        _ = c[0, 0].item()
        iterations += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    props = torch.cuda.get_device_properties(0)
    result = {"gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
              "matrix_size": matrix_size, "iterations": iterations,
              "elapsed_seconds": round(elapsed, 2),
              "iterations_per_second": round(iterations / elapsed, 2),
              "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
              "total_vram_gb": round(props.total_memory / 1024**3, 2)}
    del a, b
    torch.cuda.empty_cache()
    return result
