"""
================================================================================
MODULE 15: MEMORY OFFLOADING (CPU / GPU / NVME LAYER SWAPPING)
================================================================================

CONCEPT OVERVIEW:
-----------------
When a model exceeds GPU VRAM and purchasing extra GPU hardware is unfeasible,
**Memory Offloading** partitions model weights across a hierarchical storage tier:
1. GPU VRAM (~2.0 TB/s bandwidth - Fastest)
2. Host System CPU RAM (~100 GB/s bandwidth over PCIe Gen4/Gen5)
3. NVMe SSD Storage (~7 GB/s bandwidth - Slowest)

DEEPSPEED-ZERO-OFFLOAD & LLAMA.CPP LAYER OFFLOADING:
----------------------------------------------------
For a model with L layers:
- L_gpu layers reside on GPU VRAM.
- L_cpu layers reside in System CPU RAM.
- During forward pass execution, layers are loaded dynamically from CPU RAM to GPU SRAM
  just-in-time over the PCIe bus interface.

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `OffloadedLayerBlock`: PyTorch layer module storing weights on CPU RAM and transferring to GPU VRAM on-demand.
2. `run_offloading_simulation()`: Benchmarks PCIe layer transfer + GPU execution latency.
3. `ollama_offload_demo()`: Ollama layer offload inspection.
================================================================================
"""

import time
import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
import ollama


class OffloadedLayerBlock(nn.Module):
    """
    Simulates a heavy transformer layer residing on System CPU RAM,
    transferred to GPU VRAM dynamically during forward pass execution.
    """
    def __init__(self, hidden_dim=2048):
        super().__init__()
        # Weight Parameter resides on System CPU RAM!
        self.weight_cpu = nn.Parameter(torch.randn(hidden_dim, hidden_dim, device="cpu"))

    def forward(self, x_gpu):
        # ---------------------------------------------------------------------
        # Step 1: Transfer Weights from CPU RAM -> GPU VRAM across PCIe Bus
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        w_gpu = self.weight_cpu.to(x_gpu.device, non_blocking=True)
        
        # ---------------------------------------------------------------------
        # Step 2: Compute Matrix Multiplication on GPU
        # ---------------------------------------------------------------------
        out_gpu = torch.matmul(x_gpu, w_gpu)
        
        # ---------------------------------------------------------------------
        # Step 3: Explicitly Free GPU VRAM Weight Copy to Prevent OOM
        # ---------------------------------------------------------------------
        del w_gpu
        return out_gpu, (time.perf_counter() - t0) * 1000


def run_offloading_simulation():
    """
    Executes PyTorch CPU-to-GPU dynamic layer swapping simulation.
    """
    print("=" * 70)
    print("1. PyTorch Dynamic CPU-to-GPU Memory Offloading Simulation")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 2048  # Matrix dimension [2048 x 2048]
    print(f"Target Execution Device: {device} | Matrix Dim: [{hidden_dim} x {hidden_dim}]")

    offloaded_layer = OffloadedLayerBlock(hidden_dim)
    input_gpu = torch.randn(1, 128, hidden_dim, device=device)

    # Execute forward pass with CPU-to-GPU weight transfer
    out, transfer_time_ms = offloaded_layer(input_gpu)

    print(f"Output Tensor Shape: {list(out.shape)}")
    print(f"PCIe Layer Transfer + GPU Execution Latency: {transfer_time_ms:.4f} ms")
    print("Offloading Forward Cycle Completed Successfully!\n")


def ollama_offload_demo():
    """
    Demonstrates Ollama layer offload inspection.
    """
    print("=" * 70)
    print("2. Ollama GPU/CPU Layer Offload Inspection")
    print("=" * 70)

    model_name = "llama3.2:1b"
    try:
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Explain CPU RAM memory offloading in 2 bullet points."}]
        )
        t1 = time.perf_counter()
        print(f"Execution Latency: {t1 - t0:.2f} s")
        print(f"Response: {resp['message']['content']}\n")
    except Exception as e:
        print(f"[Ollama Notice]: Live call skipped ({e}).")


if __name__ == "__main__":
    run_offloading_simulation()
    ollama_offload_demo()
