"""
================================================================================
MODULE 14: GEMINI DYNAMIC THROUGHPUT MULTIPLEXING
================================================================================

CONCEPT OVERVIEW:
-----------------
Google Gemini endpoints serve high-QPS traffic using continuous iteration scheduling across TPU Pods.
Incoming API requests are dynamically packed into running TPU micro-batches.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Using the `google-genai` SDK with `client.aio` to simulate high-concurrency dynamic request multiplexing.
================================================================================
"""

import os
import time
import asyncio
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


async def send_dynamic_batch_req(client, req_id: int, prompt: str):
    """Sends asynchronous request to Gemini model."""
    model_name = "gemini-2.5-flash"
    t0 = time.perf_counter()
    print(f"[Req #{req_id}] Sent prompt: '{prompt[:35]}...'")
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=100)
    )
    t1 = time.perf_counter()
    print(f"[Req #{req_id}] Completed in {t1 - t0:.2f} s | Text snippet: {response.text[:50]}...")
    return t1 - t0


async def gemini_dynamic_batching_demo():
    """
    Dispatches parallel dynamic batch requests using asyncio.
    """
    print("=" * 70)
    print("Google Gemini API High-Throughput Dynamic Batch Pipeline")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # API Key & Client Setup
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] GEMINI_API_KEY environment variable is not set.")
        print("To run live, set export GEMINI_API_KEY='your_api_key'.\n")

    client = genai.Client()

    prompts = [
        "Explain iteration-level scheduling.",
        "What is time to first token?",
        "Define inter-token latency.",
        "Why is static batching inefficient for LLMs?"
    ]

    try:
        t0 = time.perf_counter()
        latencies = await asyncio.gather(*[send_dynamic_batch_req(client, i+1, p) for i, p in enumerate(prompts)])
        t1 = time.perf_counter()

        print("\n[DYNAMIC BATCH SUMMARY]:")
        print(f" Total Parallel Batch Runtime: {t1 - t0:.2f} s")
        print(f" Average Individual Latency:   {sum(latencies) / len(latencies):.2f} s\n")

    except Exception as e:
        print(f"\n[SDK Execution Note]: API call skipped or failed ({e}).")


if __name__ == "__main__":
    asyncio.run(gemini_dynamic_batching_demo())
