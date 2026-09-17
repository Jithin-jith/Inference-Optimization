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
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Warning] GOOGLE_API_KEY environment variable is not set.")
        print("To run live, set export GOOGLE_API_KEY='your_api_key'.\n")

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
