"""
================================================================================
MODULE 04: GEMINI ASYNCHRONOUS HIGH-CONCURRENCY SERVING & PAGEDATTENTION BENCHMARK
================================================================================

CONCEPT OVERVIEW:
-----------------
Google Gemini Cloud infrastructure serves millions of concurrent user sessions across global TPU Pods.
Under the hood, serverless KV page block allocators assign physical TPU memory pages dynamically as
requests arrive and stream output tokens.

HOW DEVELOPERS UTILIZE HIGH CONCURRENCY IN GEMINI SDK:
------------------------------------------------------
By using Python `asyncio` paired with the modern `google-genai` async client (`client.aio`),
developers can dispatch tens or hundreds of requests concurrently over persistent HTTP/2 connections.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Compares:
1. Sequential Execution (Traditional single-worker queue / linear KV cache allocation bottleneck)
2. Concurrent Asynchronous Execution (PagedAttention-enabled parallel dynamic block allocation)

Measures speedup ratio, latency per request, and total wall-clock time across identical prompts.
================================================================================
"""

import os
import sys
import time
import asyncio
import warnings
import logging
from typing import TypedDict
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Filter out lower-level SDK warnings written directly to sys.stderr
class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, msg):
        if "automatic function calling" in msg or "AFC" in msg:
            return
        self.original_stderr.write(msg)

    def flush(self):
        if hasattr(self.original_stderr, "flush"):
            self.original_stderr.flush()

sys.stderr = StderrFilter(sys.stderr)

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
warnings.showwarning = lambda *args, **kwargs: None

logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

load_dotenv()



def send_sync_gemini_request(client, request_id: int, prompt: str):
    """
    Sends a synchronous generate content request to Google Gemini (Sequential mode).
    """
    model_name = "gemini-2.5-flash"
    t0 = time.perf_counter()
    print(f"[Sync Req #{request_id}] Launched sequential request: '{prompt[:40]}...'")
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=150,
        )
    )
    t1 = time.perf_counter()
    print(f"[Sync Req #{request_id}] Completed in {t1 - t0:.2f} s | Snippet: {response.text[:60]}...")
    return response.text, t1 - t0


async def send_async_gemini_request(client, request_id: int, prompt: str):
    """
    Sends an asynchronous generate content request to Google Gemini (PagedAttention Concurrent mode).
    """
    model_name = "gemini-2.5-flash"
    t0 = time.perf_counter()
    print(f"[Async Req #{request_id}] Launched concurrent request: '{prompt[:40]}...'")
    
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=150,
        )
    )
    t1 = time.perf_counter()
    print(f"[Async Req #{request_id}] Completed in {t1 - t0:.2f} s | Snippet: {response.text[:60]}...")
    return response.text, t1 - t0


async def gemini_concurrent_paged_demo():
    """
    Executes a benchmark comparison between Sequential Execution and Concurrent PagedAttention Async Execution.
    """
    print("=" * 75)
    print("Google Gemini API: Sequential vs. Concurrent PagedAttention Benchmark")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # API Key & Client Setup
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Warning] GOOGLE_API_KEY environment variable is not set.")
        print("To run live calls against Google Cloud, set export GOOGLE_API_KEY='your_api_key'.\n")

    client = genai.Client()

    # Prompts to be executed in benchmark
    prompts = [
        "Explain memory virtual paging in operating systems.",
        "What is the difference between internal and external memory fragmentation?",
        "How does vLLM PagedAttention optimize GPU VRAM utilization?",
        "Compare contiguous linear memory allocation with block page tables."
    ]

    try:
        # =====================================================================
        # Phase 1: Sequential Execution (Simulating Monolithic Unpaged Queuing)
        # =====================================================================
        print("\n--- PHASE 1: SEQUENTIAL EXECUTION (Single Queue / Unpaged Bottleneck) ---")
        seq_start = time.perf_counter()
        seq_latencies = []
        for idx, p in enumerate(prompts):
            _, req_time = send_sync_gemini_request(client, idx + 1, p)
            seq_latencies.append(req_time)
        seq_end = time.perf_counter()
        seq_total_time = seq_end - seq_start

        # =====================================================================
        # Phase 2: Concurrent Asynchronous Execution (PagedAttention Parallel Serving)
        # =====================================================================
        print("\n--- PHASE 2: CONCURRENT ASYNC EXECUTION (PagedAttention Page Table Serving) ---")
        async_start = time.perf_counter()
        tasks = [send_async_gemini_request(client, idx + 1, p) for idx, p in enumerate(prompts)]
        async_results = await asyncio.gather(*tasks)
        async_end = time.perf_counter()
        async_total_time = async_end - async_start
        async_latencies = [res[1] for res in async_results]

        # =====================================================================
        # Phase 3: Benchmark Summary & Speedup Analysis
        # =====================================================================
        speedup = seq_total_time / async_total_time if async_total_time > 0 else 0

        print("\n" + "=" * 75)
        print("BENCHMARK RESULTS & SPEEDUP COMPARISON")
        print("=" * 75)
        print(f"Total Prompts Tested:               {len(prompts)}")
        print(f"Sequential Mode Total Wall Time:    {seq_total_time:.2f} s")
        print(f"Concurrent Async Mode Total Time:   {async_total_time:.2f} s")
        print(f"Execution Speedup Factor:           {speedup:.2f}x Faster")
        print("-" * 75)
        print(f"Avg Latency per Request (Seq):      {sum(seq_latencies)/len(seq_latencies):.2f} s")
        print(f"Avg Latency per Request (Async):    {sum(async_latencies)/len(async_latencies):.2f} s")
        print(f"Throughput (Seq):                   {len(prompts)/seq_total_time:.2f} req/s")
        print(f"Throughput (Async):                 {len(prompts)/async_total_time:.2f} req/s")
        print("=" * 75 + "\n")

    except Exception as e:
        print(f"\n[SDK Execution Note]: API call skipped or failed ({e}).")


if __name__ == "__main__":
    asyncio.run(gemini_concurrent_paged_demo())

