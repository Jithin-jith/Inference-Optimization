"""
================================================================================
MODULE 16: STREAMING GENERATION & PERCEIVED LATENCY REDUCTION BENCHMARK
================================================================================

CONCEPT OVERVIEW:
-----------------
In standard non-streaming HTTP REST endpoints, the user interface waits passively while the server
generates the full response payload (e.g. 500 tokens).

If generation speed is 30 tokens/sec, the user waits:
Total User Wait Time = TTFT + (500 tokens / 30 tokens/sec) = 1.5s + 16.6s = 18.1 seconds!

STREAMING GENERATION SOLUTION (SERVER-SENT EVENTS / SSE):
---------------------------------------------------------
Utilizes Server-Sent Events (SSE) or WebSockets to transmit generated tokens to the client interface
token-by-token as soon as each individual token is sampled!

PERCEIVED LATENCY (TTFT) IMPROVEMENT:
-------------------------------------
Perceived User Wait Time = Time to First Token (TTFT) ~ 100ms - 300ms!
Reduces perceived user wait time by 90% - 98%, providing an instantaneous human-like typing UX!

WHAT THIS SCRIPT BENCHMARKS:
----------------------------
1. Baseline: Non-Streaming Blocking Mode (`stream=False`).
2. Optimized: Real-Time Server-Sent Event Token Streaming (`stream=True`).
3. Detailed Parameter Comparison Table showing TTFT, total generation latency, and perceived speedup.
================================================================================
"""

import sys
import time
# pyrefly: ignore [missing-import]
import ollama


def run_simulated_stream_benchmark():
    """
    Executes a benchmark comparing non-streaming blocking delivery vs streaming generator delivery.
    """
    print("=" * 90)
    print("1. SIMULATION BENCHMARK: NON-STREAMING BLOCKING VS REAL-TIME TOKEN STREAMING")
    print("=" * 90)

    tokens = ["Streaming ", "generation ", "slashes ", "perceived ", "latency ", "to ", "near ", "zero!"]
    inter_token_delay = 0.08  # 80ms ITL

    # Baseline: Non-Streaming Blocking Call (Waits for all 8 tokens before showing ANY text)
    t0 = time.perf_counter()
    time.sleep(inter_token_delay * len(tokens))  # Simulate server generating full payload
    blocking_text = "".join(tokens)
    t1 = time.perf_counter()
    block_total_latency = t1 - t0
    block_ttft = block_total_latency  # User sees nothing until end!

    print(f" -> Non-Streaming Total Blocking Time: {block_total_latency:.3f} s")
    print(f" -> Non-Streaming Time to First Token (TTFT): {block_ttft:.3f} s")

    # Optimized: Real-Time Token Streaming
    print("\n -> Streaming Token Output: ", end="")
    t0 = time.perf_counter()
    first_token = True
    stream_ttft = 0.0
    for tok in tokens:
        time.sleep(inter_token_delay)
        if first_token:
            stream_ttft = time.perf_counter() - t0
            first_token = False
        sys.stdout.write(tok)
        sys.stdout.flush()
    t1 = time.perf_counter()
    stream_total_latency = t1 - t0
    print("\n")

    # -------------------------------------------------------------------------
    # PARAMETER COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("=" * 90)
    print("DETAILED PARAMETER COMPARISON SUMMARY: RESPONSE GENERATION MODE")
    print("=" * 90)
    print(f"  {'PARAMETER / METRIC':<30} | {'BASELINE (Non-Streaming Blocking)':<22} | {'OPTIMIZED (Real-Time Streaming)'}")
    print("  " + "-" * 86)
    print(f"  {'Output Delivery Protocol':<30} | {'Single Blocking Payload':<22} | {'Token-by-Token SSE Stream'}")
    print(f"  {'Perceived Time to 1st Token (TTFT)':<30} | {f'{block_ttft:.3f} s (Full Delay)':<22} | {f'{stream_ttft:.3f} s (Near Instant)'}")
    print(f"  {'Perceived User Wait Reduction':<30} | {'0.0% (Full Wait)':<22} | {f'{((block_ttft - stream_ttft) / block_ttft) * 100:.1f}% Faster First Token'}")
    print(f"  {'Total Generation Wall Clock':<30} | {block_total_latency:<20.3f} s | {stream_total_latency:<20.3f} s")
    print(f"  {'User Engagement Rating':<30} | {'High Abort Rate (Bored User)':<22} | {'High Engagement (Active Typing)'}")
    print("=" * 90 + "\n")


def ollama_streaming_demo():
    """
    Demonstrates real-time token streaming API call with Ollama client (`stream=True`).
    """
    print("=" * 90)
    print("2. OLLAMA REAL-TIME TOKEN STREAMING API (stream=True)")
    print("=" * 90)

    model_name = "llama3.2:1b"
    prompt = "Explain streaming generation advantages in 3 short bullet points."

    try:
        print(f" -> Connecting to Ollama stream ('{model_name}')...\n -> Stream: ", end="")
        t0 = time.perf_counter()
        first_token = True
        ttft_s = 0.0

        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream:
            if first_token:
                ttft_s = time.perf_counter() - t0
                first_token = False
            token_text = chunk['message']['content']
            sys.stdout.write(token_text)
            sys.stdout.flush()

        print(f"\n -> Perceived TTFT: {ttft_s:.3f} s\n")

    except Exception as e:
        print(f" -> [Ollama Notice]: Live streaming call skipped ({e}).\n")


if __name__ == "__main__":
    run_simulated_stream_benchmark()
    ollama_streaming_demo()

