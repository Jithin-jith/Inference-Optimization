"""
================================================================================
MODULE 01: KEY-VALUE (KV) CACHING IMPLEMENTATION & BENCHMARK
================================================================================

CONCEPT OVERVIEW:
-----------------
In autoregressive Large Language Models (LLMs), generation proceeds token-by-token.
When generating token `t_i`, the attention mechanism calculates attention scores between `t_i`
and ALL previous tokens `t_1, ..., t_{i-1}`.

PROBLEM WITHOUT KV-CACHE:
-------------------------
At step `i`, a naive implementation recomputes Key (K) and Value (V) projections for all
preceding `i-1` tokens from scratch. Over a sequence of length N:
- Computational Complexity: O(N^3)
- Wasted FLOPs: Recomputing fixed historical activations repeatedly.

SOLUTION WITH KV-CACHE:
-----------------------
Key (K) and Value (V) activation vectors of historical tokens are saved in GPU VRAM (KV-Cache).
For each new token `t_i`:
1. Compute Q_i, K_i, V_i ONLY for the new incoming token.
2. Append K_i and V_i to the cached K and V tensors.
3. Compute attention using the updated K_cache and V_cache.
- Computational Complexity: O(N^2)
- Latency Reduction: 10x - 50x faster generation per token (slashes Inter-Token Latency - ITL).

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `torch_kv_cache_demo()`: Pure PyTorch benchmark comparing manual attention execution
   WITH vs WITHOUT KV-caching, measuring execution time and demonstrating tensor shapes.
2. `ollama_kv_cache_demo()`: Practical demonstration using local Ollama model to show
   how KV cache retention across conversation turns speeds up multi-turn responses.
================================================================================
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import ollama


def torch_kv_cache_demo():
    """
    Simulates multi-head self-attention token generation step-by-step in PyTorch.
    Compares naive re-computation against KV-cache tensor concatenation.
    """
    print("=" * 70)
    print("1. PyTorch Structural Benchmark: Manual KV-Cache vs Non-Cached Generation")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Hyperparameters & Tensor Dimensions Setup
    # -------------------------------------------------------------------------
    batch_size = 1       # Single request sequence
    num_heads = 8        # Number of Attention Heads
    head_dim = 64        # Dimension per head (hidden_size = num_heads * head_dim = 512)
    seq_len = 128        # Initial prompt context token length
    num_steps = 10       # Number of new tokens to generate autoregressively
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Prompt Length: {seq_len} tokens | Decoding Steps: {num_steps}")

    # Simulated Weight Matrices for Linear Projections: Q, K, V
    # Weight shape: [Hidden_Dim, Hidden_Dim] -> [512, 512]
    W_q = torch.randn(num_heads * head_dim, num_heads * head_dim, device=device)
    W_k = torch.randn(num_heads * head_dim, num_heads * head_dim, device=device)
    W_v = torch.randn(num_heads * head_dim, num_heads * head_dim, device=device)

    # Simulated initial prompt input embeddings: Shape [Batch=1, Seq_Len=128, Hidden_Dim=512]
    prompt_inputs = torch.randn(batch_size, seq_len, num_heads * head_dim, device=device)

    # =========================================================================
    # SCENARIO A: Without KV-Cache (Recompute full sequence at every step)
    # =========================================================================
    print("\nExecuting Scenario A: WITHOUT KV-Cache...")
    start_time = time.perf_counter()
    
    # Track sequence growth: starts at 128 tokens, grows by 1 at each decoding step
    for step in range(num_steps):
        # Full current sequence: [1, 128 + step, 512]
        current_seq = prompt_inputs[:, : seq_len + step, :]
        
        # Recompute Query, Key, Value for ALL tokens from step 0 to (128+step) -> WASTED COMPUTATION!
        Q = torch.matmul(current_seq, W_q)  # [1, 128+step, 512]
        K = torch.matmul(current_seq, W_k)  # [1, 128+step, 512]
        V = torch.matmul(current_seq, W_v)  # [1, 128+step, 512]
        
        # Calculate Attention Scores: Q @ K^T / sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (head_dim ** 0.5)  # [1, 128+step, 128+step]
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)                             # [1, 128+step, 512]

    no_cache_time_ms = (time.perf_counter() - start_time) * 1000

    # =========================================================================
    # SCENARIO B: With KV-Cache (Cache past K,V; compute ONLY new 1 token)
    # =========================================================================
    print("Executing Scenario B: WITH KV-Cache...")
    start_time = time.perf_counter()

    # Step 1: Prefill Phase - Compute initial K & V cache for prompt tokens [1, 128, 512]
    K_cache = torch.matmul(prompt_inputs, W_k)  # Shape: [1, 128, 512]
    V_cache = torch.matmul(prompt_inputs, W_v)  # Shape: [1, 128, 512]

    # Step 2: Decode Phase - For each step, project ONLY the 1 new single incoming token!
    for step in range(num_steps):
        # New incoming token embedding: Shape [1, 1, 512]
        new_token_input = torch.randn(batch_size, 1, num_heads * head_dim, device=device)
        
        # Compute Q, K, V for ONLY the new 1 token (NOT historical 128 tokens!)
        Q_new = torch.matmul(new_token_input, W_q)  # [1, 1, 512]
        K_new = torch.matmul(new_token_input, W_k)  # [1, 1, 512]
        V_new = torch.matmul(new_token_input, W_v)  # [1, 1, 512]

        # Update KV-Cache: Concatenate new K_new and V_new along sequence length dimension (dim=1)
        K_cache = torch.cat([K_cache, K_new], dim=1)  # Grows to [1, 128+step+1, 512]
        V_cache = torch.cat([V_cache, V_new], dim=1)  # Grows to [1, 128+step+1, 512]

        # Compute Attention of Query_new [1, 1, 512] against full Cached K [1, 128+step+1, 512]
        scores = torch.matmul(Q_new, K_cache.transpose(-1, -2)) / (head_dim ** 0.5)  # [1, 1, 128+step+1]
        attn_weights = F.softmax(scores, dim=-1)
        output_new = torch.matmul(attn_weights, V_cache)                             # [1, 1, 512]

    cache_time_ms = (time.perf_counter() - start_time) * 1000

    # -------------------------------------------------------------------------
    # Benchmark Results Summary
    # -------------------------------------------------------------------------
    print(f"\n[BENCHMARK RESULTS ({num_steps} Decoding Steps)]:")
    print(f" Time WITHOUT KV-Cache (Recomputing): {no_cache_time_ms:.4f} ms")
    print(f" Time WITH KV-Cache    (Concatenating): {cache_time_ms:.4f} ms")
    if cache_time_ms > 0:
        print(f" Performance Speedup Factor:          {no_cache_time_ms / cache_time_ms:.2f}x faster!")
    print("-" * 70 + "\n")


def ollama_kv_cache_demo():
    """
    Demonstrates multi-turn conversation KV-cache retention in Ollama client.
    First turn builds prompt KV cache; second turn reuses cached history for instant response.
    """
    print("=" * 70)
    print("2. Ollama Practical Benchmark: Conversation Multi-Turn KV-Cache Retention")
    print("=" * 70)

    model_name = "llama3.2:1b"
    try:
        # Turn 1: Send large system/initial prompt
        prompt_1 = "Write a detailed 3-paragraph summary on how KV-caching optimizes transformer memory bandwidth."
        print(f"Turn 1: Submitting prompt to '{model_name}' (Building initial KV cache)...")

        t0 = time.perf_counter()
        resp1 = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt_1}]
        )
        t1 = time.perf_counter()
        print(f"Turn 1 Response Time: {t1 - t0:.2f} s")
        print(f"Response snippet: {resp1['message']['content'][:120]}...\n")

        # Turn 2: Follow-up question leveraging existing history
        print("Turn 2: Follow-up query leveraging previously stored conversation KV cache...")
        t2 = time.perf_counter()
        resp2 = ollama.chat(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt_1},
                {"role": "assistant", "content": resp1['message']['content']},
                {"role": "user", "content": "Summarize your previous response into 2 bullet points."}
            ]
        )
        t3 = time.perf_counter()
        print(f"Turn 2 Response Time: {t3 - t2:.2f} s")
        print(f"Response: {resp2['message']['content']}\n")

    except Exception as e:
        print(f"[Ollama Notice]: Live call skipped ({e}). Ensure Ollama is running and '{model_name}' is pulled.")


if __name__ == "__main__":
    torch_kv_cache_demo()
    ollama_kv_cache_demo()
