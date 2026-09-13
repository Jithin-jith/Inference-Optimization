"""
================================================================================
MODULE 04: GEMINI ASYNCHRONOUS HIGH-CONCURRENCY SERVING
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
Using `client.aio.models.generate_content(...)` to execute 4 parallel asynchronous requests to Gemini 2.5 Flash,
measuring total execution time vs individual request latency.
================================================================================
"""

import os
import time
import asyncio
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


async def send_async_gemini_request(client, request_id: int, prompt: str):
    """
    Sends an asynchronous generate content request to Google Gemini.
    """
    model_name = "gemini-2.5-flash"
    t0 = time.perf_counter()
    print(f"[Req #{request_id}] Launched async request: '{prompt[:40]}...'")
    
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=150,
        )
    )
    t1 = time.perf_counter()
    print(f"[Req #{request_id}] Completed in {t1 - t0:.2f} s | Snippet: {response.text[:60]}...")
    return response.text


async def gemini_concurrent_paged_demo():
    """
    Executes multiple parallel asynchronous prompts using Python asyncio.
    """
    print("=" * 70)
    print("Google Gemini API Async High-Concurrency Processing")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # API Key & Client Setup
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] GEMINI_API_KEY environment variable is not set.")
        print("To run live calls against Google Cloud, set export GEMINI_API_KEY='your_api_key'.\n")

    client = genai.Client()

    # Prompts to be executed in parallel
    prompts = [
        "Explain memory virtual paging in operating systems.",
        "What is the difference between internal and external memory fragmentation?",
        "How does vLLM PagedAttention optimize GPU VRAM utilization?",
        "Compare contiguous linear memory allocation with block page tables."
    ]

    try:
        print(f"Launching {len(prompts)} concurrent asynchronous requests to Gemini...")
        t0 = time.perf_counter()
        
        # Dispatch all requests concurrently using asyncio.gather
        tasks = [send_async_gemini_request(client, idx + 1, p) for idx, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        
        t1 = time.perf_counter()

        print("\n[CONCURRENCY SUMMARY]:")
        print(f" Total Parallel Execution Time: {t1 - t0:.2f} s")
        print(f" Average Latency per Request:   {(t1 - t0) / len(prompts):.2f} s\n")

    except Exception as e:
        print(f"\n[SDK Execution Note]: API call skipped or failed ({e}).")


if __name__ == "__main__":
    asyncio.run(gemini_concurrent_paged_demo())
