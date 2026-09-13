"""
================================================================================
MODULE 16: STREAMING GENERATION & PERCEIVED LATENCY REDUCTION
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

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `simulated_token_stream()`: Python Generator function (`yield`) simulating real-time SSE token delivery.
2. `run_simulated_stream_demo()`: Measures Time to First Token (TTFT) vs total generation time.
3. `ollama_streaming_demo()`: Ollama real-time token streaming (`stream=True`).
================================================================================
"""

import sys
import time
# pyrefly: ignore [missing-import]
import ollama


def simulated_token_stream():
    """
    Python generator function simulating token-by-token streaming with 80ms Inter-Token Latency (ITL).
    """
    tokens = ["Streaming ", "generation ", "slashes ", "perceived ", "latency ", "to ", "near ", "zero!"]
    for token in tokens:
        time.sleep(0.08)  # Simulate 80ms ITL (Inter-Token Latency)
        yield token


def run_simulated_stream_demo():
    """
    Executes token generator stream and measures perceived TTFT.
    """
    print("=" * 70)
    print("1. Simulated Token-by-Token Streaming Generator Output")
    print("=" * 70)

    print("Stream Output: ", end="")
    t0 = time.perf_counter()
    first_token = True
    ttft_ms = 0.0

    for token in simulated_token_stream():
        if first_token:
            ttft_ms = (time.perf_counter() - t0) * 1000
            first_token = False
        sys.stdout.write(token)
        sys.stdout.flush()

    total_time_s = time.perf_counter() - t0
    print(f"\n\n[STREAM METRICS]:")
    print(f" Perceived Time to First Token (TTFT): {ttft_ms:.2f} ms")
    print(f" Total Response Generation Time:      {total_time_s:.2f} s\n")


def ollama_streaming_demo():
    """
    Demonstrates real-time token streaming API call with Ollama client (`stream=True`).
    """
    print("=" * 70)
    print("2. Ollama Real-Time Token Streaming API (stream=True)")
    print("=" * 70)

    model_name = "llama3.2:1b"
    prompt = "Explain streaming generation advantages in 3 short bullet points."

    try:
        print(f"Connecting to Ollama stream ('{model_name}')...\nResponse Stream: ")
        t0 = time.perf_counter()
        first_token = True
        ttft_s = 0.0

        # Initiate streaming response using stream=True
        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        # Loop over streaming event chunks
        for chunk in stream:
            if first_token:
                ttft_s = time.perf_counter() - t0
                first_token = False
            token_text = chunk['message']['content']
            sys.stdout.write(token_text)
            sys.stdout.flush()

        print(f"\n\n -> Perceived Time to First Token (TTFT): {ttft_s:.3f} s\n")

    except Exception as e:
        print(f"\n[Ollama Notice]: Live streaming call skipped ({e}).")


if __name__ == "__main__":
    run_simulated_stream_demo()
    ollama_streaming_demo()
