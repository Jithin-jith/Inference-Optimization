"""
================================================================================
MODULE 10: TENSOR PARALLELISM (TP) SIMULATION
================================================================================

CONCEPT OVERVIEW:
-----------------
When a model (e.g. 70B parameter model requiring 140GB VRAM) exceeds single-GPU capacity,
**Tensor Parallelism (Megatron-style)** shards individual weight matrices INTRA-LAYER across
N GPUs connected by high-speed NVLink interconnects (900 GB/s on H100).

COLUMN PARALLEL LINEAR LAYER:
-----------------------------
Splits weight matrix W [In_Dim, Out_Dim] vertically into N shards:
W = [ W_1 | W_2 | ... | W_N ]
Each GPU i computes local product Y_i = X @ W_i in parallel.

ROW PARALLEL LINEAR LAYER:
--------------------------
Splits weight matrix W horizontally across N GPUs:
W = [ W_1 / W_2 / ... / W_N ]
Each GPU computes Z_i = Y_i @ W_i.
An **All-Reduce Sum** collective operation sums results across GPUs via NVLink:
Z = All-Reduce-Sum( sum(Z_i) )

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `SimulatedColumnParallelLinear`: PyTorch module sharding column projections across 2 virtual GPUs.
2. `SimulatedRowParallelLinear`: PyTorch module implementing row projections with All-Reduce Sum.
3. `run_tensor_parallel_simulation()`: Complete forward pass simulation showing tensor shapes.
4. `ollama_multi_gpu_overview()`: Engine execution.
================================================================================
"""

import time
import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
import ollama


class SimulatedColumnParallelLinear(nn.Module):
    """
    Simulates Column-Parallel Linear Projection (Splits Out_Features vertically across 2 GPUs).
    Self-Attention W_Q, W_K, W_V and MLP Gate/Up matrices use Column Parallelism.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        assert out_features % 2 == 0, "Out_features must be divisible by tensor parallel degree (2)."
        half_out = out_features // 2
        
        # Virtual GPU 0 Weight Shard: Shape [In_Features, Out_Features / 2]
        self.w_gpu0 = nn.Parameter(torch.randn(in_features, half_out))
        # Virtual GPU 1 Weight Shard: Shape [In_Features, Out_Features / 2]
        self.w_gpu1 = nn.Parameter(torch.randn(in_features, half_out))

    def forward(self, x):
        # Parallel GEMM on GPU 0 and GPU 1 simultaneously
        y0 = torch.matmul(x, self.w_gpu0)  # Shape: [Batch, Seq, Out_Features/2]
        y1 = torch.matmul(x, self.w_gpu1)  # Shape: [Batch, Seq, Out_Features/2]
        
        # Concatenate outputs along hidden feature dimension
        return torch.cat([y0, y1], dim=-1)  # Shape: [Batch, Seq, Out_Features]


class SimulatedRowParallelLinear(nn.Module):
    """
    Simulates Row-Parallel Linear Projection (Splits In_Features horizontally across 2 GPUs).
    Self-Attention Output W_O and MLP Down matrices use Row Parallelism.
    Executes NCCL All-Reduce Sum across NVLink to merge GPU results.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        assert in_features % 2 == 0, "In_features must be divisible by tensor parallel degree (2)."
        half_in = in_features // 2
        
        # Virtual GPU 0 Weight Shard: Shape [In_Features / 2, Out_Features]
        self.w_gpu0 = nn.Parameter(torch.randn(half_in, out_features))
        # Virtual GPU 1 Weight Shard: Shape [In_Features / 2, Out_Features]
        self.w_gpu1 = nn.Parameter(torch.randn(half_in, out_features))

    def forward(self, x_split0, x_split1):
        # Local GEMM on partial input tensors on GPU 0 and GPU 1
        z0 = torch.matmul(x_split0, self.w_gpu0)  # Shape: [Batch, Seq, Out_Features]
        z1 = torch.matmul(x_split1, self.w_gpu1)  # Shape: [Batch, Seq, Out_Features]
        
        # Simulated NCCL All-Reduce Sum Collective Operation across NVLink Interconnect
        z_final = z0 + z1                         # Shape: [Batch, Seq, Out_Features]
        return z_final


def run_tensor_parallel_simulation():
    """
    Executes Column + Row Tensor Parallelism simulation in PyTorch.
    """
    print("=" * 70)
    print("1. PyTorch Structural Column & Row Tensor Parallelism (TP=2) Simulation")
    print("=" * 70)

    in_dim = 128
    out_dim = 256
    seq_len = 8

    col_layer = SimulatedColumnParallelLinear(in_dim, out_dim)
    row_layer = SimulatedRowParallelLinear(out_dim, in_dim)

    # Input tensor
    x = torch.randn(1, seq_len, in_dim)
    print(f"Input Tensor Shape: {list(x.shape)}")

    # Step 1: Column Parallel Projection (Splits Out_Features across 2 GPUs)
    y_col = col_layer(x)
    print(f"Column-Parallel Combined Output Shape: {list(y_col.shape)}")

    # Split output tensor to simulate partial row-parallel inputs on 2 GPUs
    y0, y1 = torch.chunk(y_col, chunks=2, dim=-1)

    # Step 2: Row Parallel Projection with All-Reduce Sum
    z_final = row_layer(y0, y1)
    print(f"Row-Parallel Output (Post All-Reduce Sum) Shape: {list(z_final.shape)}")
    print("Simulated TP=2 Execution Completed Successfully!\n")


def ollama_multi_gpu_overview():
    """
    Demonstrates Ollama multi-GPU execution overview.
    """
    print("=" * 70)
    print("2. Ollama Multi-GPU Engine Overview")
    print("=" * 70)

    model_name = "llama3.2:1b"
    try:
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Explain NVLink All-Reduce in 2 bullet points."}]
        )
        t1 = time.perf_counter()
        print(f"Execution Latency: {t1 - t0:.2f} s")
        print(f"Response: {resp['message']['content']}\n")
    except Exception as e:
        print(f"[Ollama Notice]: Live call skipped ({e}).")


if __name__ == "__main__":
    run_tensor_parallel_simulation()
    ollama_multi_gpu_overview()
