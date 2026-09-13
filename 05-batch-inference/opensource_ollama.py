"""
================================================================================
MODULE 05: BATCH INFERENCE & ARITHMETIC INTENSITY BENCHMARK
================================================================================

CONCEPT OVERVIEW:
-----------------
Single-request LLM inference suffers from low GPU compute utilization (~15%).
Why? Loading model weights W into GPU registers for a single input vector (Batch Size = 1)
requires high memory transfers with minimal matrix math FLOPs.

ARITHMETIC INTENSITY FORMULA:
-----------------------------
Arithmetic Intensity = Total FLOPs / Total Bytes Transferred from HBM

- For Batch Size = 1: Arithmetic Intensity ~ 1 FLOP / Byte (Memory Bandwidth Bound).
- For Batch Size = B: Arithmetic Intensity ~ B FLOPs / Byte (Compute Saturation!).

By batching B requests together:
1. Weight matrices W are fetched from GPU VRAM ONCE into SRAM registers.
2. Multiplication is performed across B request vectors simultaneously.
3. Global token throughput (tokens/second) scales up by 5x - 10x!

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `benchmark_pytorch_batching()`: PyTorch matrix GEMM benchmark measuring execution time
   when running 8 requests sequentially (B=1) vs 8 requests in a single batched pass (B=8).
2. `run_ollama_batch_suite()`: Concurrent multi-prompt batch worker using Python ThreadPoolExecutor.
================================================================================
"""

import time
import torch
import concurrent.futures
# pyrefly: ignore [missing-import]
import ollama


def benchmark_pytorch_batching():
    """
    Demonstrates arithmetic intensity improvement when moving from sequential Batch Size 1 to Batched Size 8.
    """
    print("=" * 70)
    print("1. PyTorch Structural GEMM Benchmark: Sequential B=1 vs Batched B=8")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = 4096  # Matrix Dimension [4096 x 4096]
    print(f"Matrix Dimension: [{hidden_size} x {hidden_size}] | Target Device: {device}")

    # Weight Matrix W: Shape [4096, 4096]
    weights = torch.randn(hidden_size, hidden_size, device=device)

    # -------------------------------------------------------------------------
    # SCENARIO A: 8 Sequential Single-Request Iterations (Batch Size = 1)
    # -------------------------------------------------------------------------
    single_inputs = [torch.randn(1, hidden_size, device=device) for _ in range(8)]
    
    start_t = time.perf_counter()
    for x in single_inputs:
        # Fetching weights from VRAM 8 separate times -> Memory Bandwidth Bottleneck!
        _ = torch.matmul(x, weights)
    if device == "cuda":
        torch.cuda.synchronize()
    sequential_time_ms = (time.perf_counter() - start_t) * 1000

    # -------------------------------------------------------------------------
    # SCENARIO B: Single Batched Execution Pass (Batch Size = 8)
    # -------------------------------------------------------------------------
    batched_input = torch.randn(8, hidden_size, device=device)
    
    start_t = time.perf_counter()
    # Fetching weights from VRAM ONCE into SRAM registers -> Peak Compute Saturation!
    _ = torch.matmul(batched_input, weights)
    if device == "cuda":
        torch.cuda.synchronize()
    batched_time_ms = (time.perf_counter() - start_t) * 1000

    # -------------------------------------------------------------------------
    # Results Summary
    # -------------------------------------------------------------------------
    print(f"\n[BENCHMARK RESULTS]:")
    print(f" Sequential Time (8 x B=1 Passes): {sequential_time_ms:.4f} ms")
    print(f" Batched Time    (1 x B=8 Pass):   {batched_time_ms:.4f} ms")
    print(f" Throughput Speedup Factor:        {sequential_time_ms / max(batched_time_ms, 1e-6):.2f}x faster!")
    print("-" * 70 + "\n")


def run_ollama_batch_worker(prompt_tuple):
    """Worker function for executing asynchronous batch items."""
    req_id, prompt = prompt_tuple
    model_name = "llama3.2:1b"
    t0 = time.perf_counter()
    try:
        res = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        t1 = time.perf_counter()
        return req_id, t1 - t0, res['message']['content'][:60]
    except Exception as e:
        return req_id, 0.0, f"Failed ({e})"


def run_ollama_batch_suite():
    """
    Executes a concurrent batch dataset pipeline with Ollama.
    """
    print("=" * 70)
    print("2. Ollama Practical Concurrent Batch Pipeline")
    print("=" * 70)

    prompts = [
        (1, "Define Batch Inference in machine learning."),
        (2, "Summarize benefits of GPU matrix multiplication."),
        (3, "What is time to first token (TTFT)?"),
        (4, "Explain arithmetic intensity in CUDA kernels.")
    ]

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_ollama_batch_worker, prompts))
    total_batch_time = time.perf_counter() - t0

    print(f"Processed Batch of {len(prompts)} Requests in: {total_batch_time:.2f} s")
    for req_id, latency, text in results:
        print(f" Req #{req_id} Latency: {latency:.2f} s | Snippet: {text}...")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    benchmark_pytorch_batching()
    run_ollama_batch_suite()
