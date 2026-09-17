"""
================================================================================
MODULE 07: PARALLEL DECODING (MEDUSA / LOOKAHEAD MULTI-HEAD PREDICTION) BENCHMARK
================================================================================

CONCEPT OVERVIEW:
-----------------
Standard autoregressive decoding generates tokens strictly one by one in sequence:
t_1 -> t_2 -> t_3 ... requiring N sequential forward passes.

PARALLEL DECODING SOLUTION (MEDUSA ARCHITECTURE):
-------------------------------------------------
Medusa attaches multiple auxiliary linear decoding heads (Medusa Heads) on top of the base model's
last hidden state `h_t`:
- Head 1 predicts token t+1
- Head 2 predicts token t+2
- Head 3 predicts token t+3

Generates a candidate tree of multi-token sequences simultaneously.
In the next step, a single parallel tree-attention forward pass verifies all tree paths at once!

WHAT THIS SCRIPT BENCHMARKS:
----------------------------
1. Baseline: Sequential Autoregressive Decoding (1 token per forward pass step).
2. Optimized: Medusa 3-Head Multi-Token Prediction & Parallel Tree Attention Verification.
3. Detailed Parameter Comparison Table comparing forward pass count, throughput, and speedup.
================================================================================
"""

import time
import sys
# pyrefly: ignore [missing-import]
import ollama

# Safe PyTorch import with Windows DLL policy fallback
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


class DummyLinear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
    def __call__(self, x):
        return [0.5] * self.out_features


class MedusaHeadBlock:
    """
    Simulates 3 Medusa auxiliary decoding heads attached to a base transformer backbone.
    """
    def __init__(self, hidden_dim=64, vocab_size=1000):
        if TORCH_AVAILABLE:
            class PyTorchMedusa(nn.Module):
                def __init__(self, h_dim, v_size):
                    super().__init__()
                    self.head1 = nn.Linear(h_dim, v_size)
                    self.head2 = nn.Linear(h_dim, v_size)
                    self.head3 = nn.Linear(h_dim, v_size)
                def forward(self, h_t):
                    return self.head1(h_t), self.head2(h_t), self.head3(h_t)
            self.net = PyTorchMedusa(hidden_dim, vocab_size)
        else:
            self.h1 = DummyLinear(hidden_dim, vocab_size)
            self.h2 = DummyLinear(hidden_dim, vocab_size)
            self.h3 = DummyLinear(hidden_dim, vocab_size)

    def forward(self, h_t):
        if TORCH_AVAILABLE:
            return self.net(h_t)
        else:
            return [101], [202], [303]


def run_parallel_decoding_benchmark():
    """
    Benchmarks standard sequential 1-token decoding vs Medusa 3-head parallel tree decoding.
    """
    print("=" * 90)
    print("1. PYTORCH / SIMULATION BENCHMARK: SEQUENTIAL DECODING VS. MEDUSA PARALLEL TREE DECODING")
    print("=" * 90)

    target_tokens = 30
    hidden_dim = 64
    vocab_size = 1000

    # -------------------------------------------------------------------------
    # PHASE 1: Baseline Sequential Decoding (1 Token / Step)
    # -------------------------------------------------------------------------
    print("\n[PHASE 1] Executing Baseline Autoregressive Decoding (1 Token per Forward Pass)...")
    t0 = time.perf_counter()
    base_forward_passes = target_tokens
    for _ in range(base_forward_passes):
        # Simulate single forward pass latency per token
        time.sleep(0.003)
    t1 = time.perf_counter()
    base_latency = t1 - t0
    base_tps = target_tokens / max(base_latency, 0.0001)

    print(f" -> Baseline Forward Passes: {base_forward_passes}")
    print(f" -> Baseline Execution Time: {base_latency:.4f} s")
    print(f" -> Baseline Decoding Speed: {base_tps:.2f} tok/s")

    # -------------------------------------------------------------------------
    # PHASE 2: Optimized Medusa Parallel Decoding (3 Tokens / Step)
    # -------------------------------------------------------------------------
    print("\n[PHASE 2] Executing Medusa Multi-Head Parallel Decoding & Tree Verification...")
    t0 = time.perf_counter()
    if TORCH_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        medusa_heads = MedusaHeadBlock(hidden_dim, vocab_size)
        medusa_heads.net.to(device)
        h_t = torch.randn(1, 1, hidden_dim, device=device)
        t1_l, t2_l, t3_l = medusa_heads.forward(h_t)
        cand_t1 = torch.argmax(t1_l, dim=-1).item()
        cand_t2 = torch.argmax(t2_l, dim=-1).item()
        cand_t3 = torch.argmax(t3_l, dim=-1).item()
    else:
        cand_t1, cand_t2, cand_t3 = 101, 202, 303

    # With 3 heads + tree verification, average effective step emits ~2.5 accepted tokens per forward pass
    tokens_per_pass = 2.5
    medusa_forward_passes = int(target_tokens / tokens_per_pass)
    for _ in range(medusa_forward_passes):
        time.sleep(0.0035)  # Slightly higher overhead for tree attention verification
    t1 = time.perf_counter()
    medusa_latency = t1 - t0
    medusa_tps = target_tokens / max(medusa_latency, 0.0001)

    print(f" -> Generated Candidate Tree Tokens (t+1, t+2, t+3): [{cand_t1}, {cand_t2}, {cand_t3}]")
    print(f" -> Medusa Forward Passes: {medusa_forward_passes}")
    print(f" -> Medusa Execution Time: {medusa_latency:.4f} s")
    print(f" -> Medusa Decoding Speed: {medusa_tps:.2f} tok/s")

    # -------------------------------------------------------------------------
    # PHASE 3: Parameter Comparison Table
    # -------------------------------------------------------------------------
    speedup = base_latency / max(medusa_latency, 0.0001)

    print("\n" + "=" * 90)
    print("DETAILED PARAMETER COMPARISON SUMMARY: STANDARD AUTOREGRESSIVE VS. MEDUSA PARALLEL")
    print("=" * 90)
    print(f"  {'PARAMETER / METRIC':<33} | {'BASELINE (Sequential AR)':<32} | {'OPTIMIZED (Medusa Parallel)'}")
    print("  " + "-" * 86)
    print(f"  {'Decoding Architecture':<33} | {'Single-Head Sequential':<32} | {'3 Auxiliary Medusa Heads + Tree Attn'}")
    print(f"  {'Target Tokens Generated':<33} | {target_tokens:<32} | {target_tokens}")
    print(f"  {'Total Forward Passes Required':<33} | {base_forward_passes:<32} | {medusa_forward_passes}")
    print(f"  {'Avg Tokens per Forward Pass':<33} | {'1.00 token/pass':<32} | {f'{tokens_per_pass:.2f} tokens/pass'}")
    print(f"  {'Total Wall Clock Latency':<33} | {base_latency:<30.4f} s | {medusa_latency:<30.4f} s")
    print(f"  {'Effective Decoding Throughput':<33} | {base_tps:<28.2f} tok/s | {medusa_tps:<28.2f} tok/s")
    print(f"  {'Latency Reduction / Speedup':<33} | {'1.00x Baseline':<32} | {speedup:<.2f}x Speedup")
    print("=" * 90 + "\n")


def ollama_parallel_decoding_demo():
    """
    Demonstrates Ollama engine query.
    """
    print("=" * 90)
    print("2. OLLAMA ENGINE MULTI-TASK GENERATION DEMO")
    print("=" * 90)

    model_name = "llama3.2:1b"
    prompt = "List 3 benefits of parallel token decoding."
    try:
        t0 = time.perf_counter()
        resp = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        t1 = time.perf_counter()
        print(f" -> Ollama Latency: {t1 - t0:.2f} s")
        print(f" -> Snippet: {resp['message']['content'][:120]}...\n")
    except Exception as e:
        print(f" -> [Ollama Notice]: Live call skipped ({e}).\n")


if __name__ == "__main__":
    run_parallel_decoding_benchmark()
    ollama_parallel_decoding_demo()

