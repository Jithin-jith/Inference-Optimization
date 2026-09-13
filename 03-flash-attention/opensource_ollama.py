"""
================================================================================
MODULE 03: FLASHATTENTION TILING & SDPA BENCHMARK
================================================================================

CONCEPT OVERVIEW:
-----------------
Standard Softmax Attention calculates:
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

In standard PyTorch, this allocates intermediate matrices S = Q @ K^T and A = softmax(S)
in High Bandwidth Memory (HBM). For sequence length N, memory reads/writes to HBM scale as O(N^2).
Since HBM bandwidth is far slower than GPU SRAM compute, attention becomes severely memory-bound.

FLASHATTENTION SOLUTION (TILING & ONLINE SOFTMAX):
--------------------------------------------------
FlashAttention divides Q, K, V matrices into smaller block tiles that fit inside GPU SRAM (19MB on A100):
1. Loads block tiles into fast SRAM.
2. Computes attention locally inside SRAM using Online Softmax scaling without ever writing the
   full N x N matrix back to HBM.
3. Reduces memory access from O(N^2) to O(N).

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `benchmark_flash_attention()`: PyTorch benchmark comparing Naive Attention (allocating N x N tensor)
   against PyTorch Scaled Dot Product Attention (SDPA / FlashAttention kernel), measuring speedup and VRAM.
2. `ollama_flash_attention_demo()`: Ollama execution over a long context prompt (~2,000 words).
================================================================================
"""

import time
import torch
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import ollama


def benchmark_flash_attention():
    """
    Compares Naive PyTorch Softmax Attention against PyTorch Scaled Dot Product Attention (SDPA).
    Demonstrates execution latency and peak memory usage on GPU SRAM.
    """
    print("=" * 70)
    print("1. PyTorch Benchmark: Naive O(N^2) Attention vs FlashAttention (SDPA)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing FlashAttention benchmark on device: {device}")

    # Hyperparameters for Long-Context Attention
    batch_size = 2       # Number of sequences in batch
    num_heads = 16       # Number of Attention Heads
    head_dim = 64        # Dimension per head
    seq_len = 4096       # Sequence Length: 4,096 tokens (Highlights O(N^2) memory footprint!)

    print(f"Sequence Length: {seq_len} tokens | Batch: {batch_size} | Heads: {num_heads} | Head Dim: {head_dim}")

    # Initialize synthetic Query, Key, Value Tensors: Shape [Batch=2, Heads=16, Seq_Len=4096, Head_Dim=64]
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)

    # -------------------------------------------------------------------------
    # METHOD A: Naive Attention (Materializes intermediate N x N Attention Matrix in HBM)
    # -------------------------------------------------------------------------
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    start_t = time.perf_counter()
    # Step 1: Compute Q @ K^T -> Shape: [2, 16, 4096, 4096] (MASSIVE 4096 x 4096 Tensor allocated in HBM!)
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (head_dim ** 0.5)
    # Step 2: Softmax normalization in HBM
    attn_weights = F.softmax(scores, dim=-1)
    # Step 3: Compute final Output matrix
    output_naive = torch.matmul(attn_weights, V)
    
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time_ms = (time.perf_counter() - start_t) * 1000
    naive_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0

    # -------------------------------------------------------------------------
    # METHOD B: FlashAttention via PyTorch Scaled Dot Product Attention (SDPA)
    # -------------------------------------------------------------------------
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_t = time.perf_counter()
    # PyTorch automatically selects FlashAttention CUDA kernel when hardware supports Ampere/Hopper
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
        output_flash = F.scaled_dot_product_attention(Q, K, V)
    
    if device == "cuda":
        torch.cuda.synchronize()
    flash_time_ms = (time.perf_counter() - start_t) * 1000
    flash_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0

    # -------------------------------------------------------------------------
    # Benchmark Results Summary
    # -------------------------------------------------------------------------
    print(f"\n[BENCHMARK RESULTS (Sequence Length = {seq_len})]:")
    print(f" Naive Attention Time:   {naive_time_ms:.4f} ms | Peak VRAM Memory: {naive_mem_mb:.2f} MB")
    print(f" FlashAttention Time:    {flash_time_ms:.4f} ms | Peak VRAM Memory: {flash_mem_mb:.2f} MB")
    if flash_time_ms > 0:
        print(f" Compute Speedup Factor: {naive_time_ms / flash_time_ms:.2f}x faster!")
    print("-" * 70 + "\n")


def ollama_flash_attention_demo():
    """
    Demonstrates long-context processing execution with local Ollama engine.
    """
    print("=" * 70)
    print("2. Ollama Practical Demonstration: Long Context Execution")
    print("=" * 70)

    model_name = "llama3.2:1b"
    # Generate synthetic long context prompt (~1500 items)
    long_prompt = "Dataset Items: " + " ".join([f"item_{i}" for i in range(1500)]) + "\nSummarize the sequence above."

    try:
        print(f"Submitting long prompt context ({len(long_prompt.split())} words) to Ollama...")
        t0 = time.perf_counter()
        resp = ollama.chat(model=model_name, messages=[{"role": "user", "content": long_prompt}])
        t1 = time.perf_counter()
        print(f"Ollama Execution Time: {t1 - t0:.2f} s")
        print(f"Response snippet: {resp['message']['content'][:120]}...\n")
    except Exception as e:
        print(f"[Ollama Notice]: Live call skipped ({e}).")


if __name__ == "__main__":
    benchmark_flash_attention()
    ollama_flash_attention_demo()
