"""
================================================================================
MODULE 08: MIXED PRECISION INFERENCE (FP32 VS FP16 VS BF16) BENCHMARK
================================================================================

CONCEPT OVERVIEW:
-----------------
Deep learning models were historically trained in FP32 (Single Precision Floating Point - 4 bytes/weight).

MIXED PRECISION INFERENCE SOLUTION:
-----------------------------------
Inference models run using lower bit-width formats:
- FP16 (Half Precision - 2 bytes/weight): 5 exponent bits, 10 mantissa bits.
- BF16 (Bfloat16 - 2 bytes/weight): 8 exponent bits (same dynamic range as FP32!), 7 mantissa bits.
- FP8 (Floating Point 8 - 1 byte/weight): Supported on NVIDIA Hopper (H100) & Ada Lovelace GPUs.

KEY ADVANTAGES:
---------------
1. 50% VRAM Reduction: Cuts memory footprint by 2x (e.g. 70B model drops from 280GB down to 140GB).
2. Tensor Core Speedup: Tensor Cores process matrix math 2x - 4x faster in FP16/BF16 than FP32!

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `benchmark_mixed_precision()`: PyTorch benchmark running GEMM matrix multiplication
   in FP32 vs FP16 vs BF16, measuring execution time and VRAM tensor memory footprint.
2. `ollama_precision_check()`: Displays Ollama model precision flags.
================================================================================
"""

import time
import torch
# pyrefly: ignore [missing-import]
import ollama


def benchmark_mixed_precision():
    """
    Executes matrix multiplication across FP32, FP16, and BF16 floating-point formats in PyTorch.
    """
    print("=" * 70)
    print("1. PyTorch Precision Benchmark (FP32 vs FP16 vs BF16)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Execution Device: {device}")

    dim = 4096            # Matrix dimension [4096 x 4096]
    num_iterations = 20   # Number of benchmark iterations

    # Generate baseline FP32 matrices
    W = torch.randn(dim, dim, device=device)
    X = torch.randn(dim, dim, device=device)

    # -------------------------------------------------------------------------
    # TEST A: FP32 Single Precision (4 Bytes / Element)
    # -------------------------------------------------------------------------
    W_fp32 = W.to(torch.float32)
    X_fp32 = X.to(torch.float32)
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_t = time.perf_counter()
    for _ in range(num_iterations):
        _ = torch.matmul(X_fp32, W_fp32)
    if device == "cuda":
        torch.cuda.synchronize()
    fp32_time_ms = (time.perf_counter() - start_t) * 1000
    fp32_mem_mb = (W_fp32.element_size() * W_fp32.nelement()) / (1024 ** 2)

    # -------------------------------------------------------------------------
    # TEST B: FP16 Half Precision (2 Bytes / Element)
    # -------------------------------------------------------------------------
    W_fp16 = W.to(torch.float16)
    X_fp16 = X.to(torch.float16)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_t = time.perf_counter()
    for _ in range(num_iterations):
        _ = torch.matmul(X_fp16, W_fp16)
    if device == "cuda":
        torch.cuda.synchronize()
    fp16_time_ms = (time.perf_counter() - start_t) * 1000
    fp16_mem_mb = (W_fp16.element_size() * W_fp16.nelement()) / (1024 ** 2)

    # -------------------------------------------------------------------------
    # TEST C: BF16 Bfloat16 (2 Bytes / Element)
    # -------------------------------------------------------------------------
    bf16_supported = device == "cuda" and torch.cuda.is_bf16_supported()
    bf16_time_ms, bf16_mem_mb = 0.0, 0.0
    if bf16_supported:
        W_bf16 = W.to(torch.bfloat16)
        X_bf16 = X.to(torch.bfloat16)
        start_t = time.perf_counter()
        for _ in range(num_iterations):
            _ = torch.matmul(X_bf16, W_bf16)
        if device == "cuda":
            torch.cuda.synchronize()
        bf16_time_ms = (time.perf_counter() - start_t) * 1000
        bf16_mem_mb = (W_bf16.element_size() * W_bf16.nelement()) / (1024 ** 2)

    # -------------------------------------------------------------------------
    # Results Summary
    # -------------------------------------------------------------------------
    print(f"\n[BENCHMARK RESULTS ({dim}x{dim} Matrix, {num_iterations} Iterations)]:")
    print(f" FP32 Time: {fp32_time_ms:.4f} ms | Tensor VRAM Memory: {fp32_mem_mb:.2f} MB")
    print(f" FP16 Time: {fp16_time_ms:.4f} ms | Tensor VRAM Memory: {fp16_mem_mb:.2f} MB (50% VRAM Reduction!)")
    if bf16_supported:
        print(f" BF16 Time: {bf16_time_ms:.4f} ms | Tensor VRAM Memory: {bf16_mem_mb:.2f} MB")
    else:
        print(" BF16: Hardware native BF16 acceleration requires Ampere/Hopper GPU.")
    print("-" * 70 + "\n")


def ollama_precision_check():
    """
    Inspects precision quantization flags in local Ollama models.
    """
    print("=" * 70)
    print("2. Ollama Local Model Precision Details")
    print("=" * 70)

    model_name = "llama3.2:1b"
    try:
        details = ollama.show(model=model_name)
        print(f"Model Name: {model_name}")
        print(f"Quantization / Precision Level: {details.get('details', {}).get('quantization_level', 'Q4_K_M / FP16')}")
        print(f"Model Family Architecture:       {details.get('details', {}).get('family', 'llama')}\n")
    except Exception as e:
        print(f"[Ollama Notice]: Live check skipped ({e}).")


if __name__ == "__main__":
    benchmark_mixed_precision()
    ollama_precision_check()
