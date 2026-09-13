"""
================================================================================
MODULE 11: PIPELINE PARALLELISM (PP) MICRO-BATCH SCHEDULER
================================================================================

CONCEPT OVERVIEW:
-----------------
While Tensor Parallelism (TP) splits weight matrices INTRA-LAYER within a single node,
**Pipeline Parallelism (PP)** divides total transformer layers INTER-LAYER across multiple GPU nodes.

FOR A MODEL WITH L LAYERS AND P PIPELINE STAGES:
------------------------------------------------
Layers per GPU Stage = L / P
For example, a 96-layer model split across 4 nodes:
- Stage 0 (Node 0): Layers 1 to 24
- Stage 1 (Node 1): Layers 25 to 48
- Stage 2 (Node 2): Layers 49 to 72
- Stage 3 (Node 3): Layers 73 to 96

PIPELINE BUBBLE & MICRO-BATCHING:
---------------------------------
To prevent GPUs from idling while waiting for upstream nodes, input batches are split into
M micro-batches. As Stage 0 finishes micro-batch 1, it passes intermediate activations to Stage 1
and immediately starts processing micro-batch 2.

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `PipelineStage1`, `PipelineStage2`, `PipelineStage3`: PyTorch modules simulating 3 pipeline stages.
2. `run_pipeline_parallel_simulation()`: Execution loop demonstrating micro-batch activation passing.
3. `ollama_pipeline_demo()`: Ollama execution overview.
================================================================================
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import ollama


class PipelineStage1(nn.Module):
    """Simulates GPU 0: Layers 1 to 4"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        return F.relu(self.layer(x))


class PipelineStage2(nn.Module):
    """Simulates GPU 1: Layers 5 to 8"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        return F.relu(self.layer(x))


class PipelineStage3(nn.Module):
    """Simulates GPU 2: Layers 9 to 12"""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.head = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        return self.head(x)


def run_pipeline_parallel_simulation():
    """
    Executes PyTorch 3-Stage Pipeline Parallelism with Micro-Batches.
    """
    print("=" * 70)
    print("1. PyTorch 3-Stage Pipeline Parallelism with Micro-Batches")
    print("=" * 70)

    hidden_dim = 128
    vocab_size = 500
    num_micro_batches = 4

    # Initialize 3 Pipeline Stages
    stage1 = PipelineStage1(hidden_dim)
    stage2 = PipelineStage2(hidden_dim)
    stage3 = PipelineStage3(hidden_dim, vocab_size)

    # Input batch split into 4 micro-batches
    batch_input = torch.randn(num_micro_batches, 2, hidden_dim)
    print(f"Total Input Payload: {num_micro_batches} Micro-Batches")

    pipeline_outputs = []

    # Simulate Pipelined Forward Execution across stages
    for mb_idx in range(num_micro_batches):
        mb = batch_input[mb_idx : mb_idx + 1]
        
        # Step 1: Stage 1 Execution (GPU 0) -> Produces Activation Tensor 1
        act1 = stage1(mb)
        
        # Step 2: Stage 2 Execution (GPU 1 receives Activation Tensor 1) -> Produces Activation Tensor 2
        act2 = stage2(act1)
        
        # Step 3: Stage 3 Execution (GPU 2 receives Activation Tensor 2) -> Computes Final Vocabulary Logits
        logits = stage3(act2)
        pipeline_outputs.append(logits)

        print(f" Micro-Batch #{mb_idx + 1}: Passed Stage 1 -> Stage 2 -> Stage 3 (Final Logits Shape: {list(logits.shape)})")

    print("Pipeline Micro-Batch Execution Completed Successfully!\n")


def ollama_pipeline_demo():
    """
    Demonstrates Ollama multi-stage execution concept.
    """
    print("=" * 70)
    print("2. Ollama Practical Demonstration: Pipeline Parallel Concept")
    print("=" * 70)

    model_name = "llama3.2:1b"
    try:
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Explain pipeline bubble idle time in 2 bullet points."}]
        )
        t1 = time.perf_counter()
        print(f"Execution Latency: {t1 - t0:.2f} s")
        print(f"Response: {resp['message']['content']}\n")
    except Exception as e:
        print(f"[Ollama Notice]: Live call skipped ({e}).")


if __name__ == "__main__":
    run_pipeline_parallel_simulation()
    ollama_pipeline_demo()
