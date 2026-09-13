"""
================================================================================
MODULE 13: GRAPH OPTIMIZATION & OPERATOR FUSION BENCHMARK
================================================================================

CONCEPT OVERVIEW:
-----------------
In default PyTorch eager execution, every individual tensor operation (Linear -> BiasAdd -> LayerNorm -> GeLU)
triggers an individual CUDA kernel launch overhead from Host CPU to GPU Device.

For deep networks running single-token decode passes, **Host CPU Python runtime overhead** can account
for up to 30% - 50% of total execution latency!

COMPUTATIONAL GRAPH COMPILATION SOLUTIONS:
------------------------------------------
1. Operator Fusion: Combines adjacent math ops into a single compiled CUDA kernel (e.g. BiasAdd + LayerNorm + GeLU
   computed in 1 SRAM pass without intermediate HBM writes).
2. CUDA Graphs: Captures the entire execution graph once during warm-up. Subsequent iterations launch
   the entire graph with ZERO CPU launch overhead.
3. ONNX Runtime & TensorRT-LLM: Compiles graphs into hardware-optimized binary engines.
4. Torch Compile (`torch.compile`): Uses OpenAI Triton compiler to dynamically generate fused CUDA kernels.

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `UnfusedMLPBlock`: PyTorch multi-layer block.
2. `benchmark_torch_compile()`: Measures execution time of Eager PyTorch vs `torch.compile` graph fusion.
3. `ollama_graph_demo()`: Ollama execution overview.
================================================================================
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import ollama


class UnfusedMLPBlock(nn.Module):
    """
    Simulates a multi-layer MLP block with sequential ops (Linear -> ReLU -> Linear -> LayerNorm).
    """
    def __init__(self, dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Multiple individual ops -> Eager PyTorch launches 4 separate CUDA kernels!
        h = F.relu(self.fc1(x))
        out = self.norm(self.fc2(h) + x)
        return out


def benchmark_torch_compile():
    """
    Benchmarks PyTorch Eager Execution vs Torch.compile Operator Fusion Graph.
    """
    print("=" * 70)
    print("1. PyTorch Eager Mode vs Compiled Graph (`torch.compile`) Operator Fusion")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 1024
    num_iters = 50

    model = UnfusedMLPBlock(dim).to(device)
    x = torch.randn(16, dim, device=device)

    # -------------------------------------------------------------------------
    # MODE A: Eager Execution (Separate CUDA Kernels per Op)
    # -------------------------------------------------------------------------
    # Warmup
    for _ in range(5):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    start_t = time.perf_counter()
    for _ in range(num_iters):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    eager_time_ms = (time.perf_counter() - start_t) * 1000

    # -------------------------------------------------------------------------
    # MODE B: Torch Compile Graph (Fused Triton CUDA Kernels)
    # -------------------------------------------------------------------------
    compiled_model = torch.compile(model, mode="reduce-overhead")
    # Trigger compilation warm-up pass
    for _ in range(5):
        _ = compiled_model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    start_t = time.perf_counter()
    for _ in range(num_iters):
        _ = compiled_model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    compile_time_ms = (time.perf_counter() - start_t) * 1000

    # -------------------------------------------------------------------------
    # Results Summary
    # -------------------------------------------------------------------------
    print(f"\n[BENCHMARK RESULTS ({num_iters} Iterations)]:")
    print(f" Eager Execution Time:    {eager_time_ms:.4f} ms")
    print(f" Compiled Graph Time:     {compile_time_ms:.4f} ms")
    if compile_time_ms > 0:
        print(f" Operator Fusion Speedup: {eager_time_ms / max(compile_time_ms, 1e-6):.2f}x faster!")
    print("-" * 70 + "\n")


def ollama_graph_demo():
    """
    Demonstrates Ollama compiled model execution.
    """
    print("=" * 70)
    print("2. Ollama Practical Demonstration: Compiled Graph Execution")
    print("=" * 70)

    model_name = "llama3.2:1b"
    try:
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Explain operator fusion in computational graphs in 2 bullet points."}]
        )
        t1 = time.perf_counter()
        print(f"Execution Latency: {t1 - t0:.2f} s")
        print(f"Response: {resp['message']['content']}\n")
    except Exception as e:
        print(f"[Ollama Notice]: Live call skipped ({e}).")


if __name__ == "__main__":
    benchmark_torch_compile()
    ollama_graph_demo()
