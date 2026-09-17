"""
================================================================================
MODULE 01: GEMINI PROPRIETARY CONTEXT CACHING (KV-CACHE API)
================================================================================

CONCEPT OVERVIEW:
-----------------
In cloud-managed LLM APIs (like Google Gemini), hardware GPU VRAM allocation is abstracted
away from the user. However, when repeatedly passing large prompts (system instructions,
massive PDF files, API documentation, or codebases > 32,768 tokens), recomputing prompt attention
at every request incurs high costs and latency.

GEMINI CONTEXT CACHING SOLUTION:
--------------------------------
Google Gemini provides explicit **Context Caching**. By creating a cached content resource,
Google's backend persists the pre-calculated KV-Cache for the system prompt on TPU server memory
for a user-specified Time-to-Live (TTL).

KEY ADVANTAGES:
---------------
1. Cost Savings: Up to 75% - 80% discount on input token costs for cached prompts.
2. Latency Reduction: Dramatically slashes Time to First Token (TTFT).
3. Session Efficiency: Shared KV cache can be queried across multiple API calls simultaneously.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Using the official modern `google-genai` SDK to:
1. Initialize `genai.Client()`.
2. Allocate an explicit Cached Content resource using `client.caches.create()`.
3. Perform an **Uncached** query (recomputing full system instruction context).
4. Perform a **Cached** query referencing pre-computed KV state via `types.GenerateContentConfig(cached_content=...)`.
5. Compare latency and usage metadata across both requests.
6. Clean up cache resources via `client.caches.delete()`.
================================================================================
"""

import os
import sys
import time
import warnings
import logging
from typing import TypedDict
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
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


def gemini_context_caching_demo():
    """
    Demonstrates Google Gemini Context Caching API workflow using google-genai SDK.
    """
    print("=" * 70)
    print("Google Gemini API: Explicit Context Caching (Serverless KV-Cache)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # API Key & Client Setup
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Warning] GOOGLE_API_KEY environment variable is not set.")
        print("To execute live API calls against Google Cloud:")
        print(" -> PowerShell: $env:GOOGLE_API_KEY='your_api_key'")
        print(" -> Bash/Linux: export GOOGLE_API_KEY='your_api_key'\n")

    # Initialize the modern official Google GenAI SDK Client
    client = genai.Client()
    model_name = "gemini-2.5-flash"

    # Construct a large system instruction context (simulated here)
    # Production context caching requires minimum token lengths (e.g. 32,768 tokens)
    large_system_instruction = """
    You are an expert Enterprise Solutions Architect AI. You provide technical blueprints
    for LLM inference optimization, cloud GPU cluster design, and high-throughput microservices.
    """ + ("Reference Blueprint Line: Always output structured latency metrics. " * 160)

    try:
        # ---------------------------------------------------------------------
        # Step 1: Create explicit Context Cache resource on Google Cloud
        # ---------------------------------------------------------------------
        print("1. Allocating explicit Context Cache on Google Cloud Infrastructure...")
        
        cache_resource = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                contents=[large_system_instruction],
                ttl="300s",  # Time to Live: Cache persists on Google TPUs for 5 minutes (300s)
                display_name="enterprise_arch_kb_cache",
            )
        )
        print(f" -> Cache Created Successfully!")
        print(f" -> Resource Name ID:  {cache_resource.name}")
        print(f" -> Display Name:      {cache_resource.display_name}")

        # ---------------------------------------------------------------------
        # Step 2: Query Model WITHOUT Caching (Full Attention Re-computation)
        # ---------------------------------------------------------------------
        print("\n2. Querying Gemini Model WITHOUT Cache (Full Context Re-computation)...")
        user_query = "What are the primary tradeoffs between Tensor Parallelism and Pipeline Parallelism?"
        
        t0_uncached = time.perf_counter()
        response_uncached = client.models.generate_content(
            model=model_name,
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=large_system_instruction,  # Passing full context directly
                temperature=0.2
            )
        )
        t1_uncached = time.perf_counter()
        latency_uncached = t1_uncached - t0_uncached
        
        print(f" -> Latency (WITHOUT Cache): {latency_uncached:.2f} seconds")
        print(f" -> Usage Metadata:          {response_uncached.usage_metadata}")
        print(f" -> Response Snippet:\n{response_uncached.text[:180]}...\n")

        # ---------------------------------------------------------------------
        # Step 3: Query Model WITH Cached Content Reference
        # ---------------------------------------------------------------------
        print("3. Querying Gemini Model WITH Cached KV State...")
        
        t0_cached = time.perf_counter()
        response_cached = client.models.generate_content(
            model=model_name,
            contents=user_query,
            config=types.GenerateContentConfig(
                cached_content=cache_resource.name,  # Attaches pre-computed KV cache!
                temperature=0.2
            )
        )
        t1_cached = time.perf_counter()
        latency_cached = t1_cached - t0_cached
        
        print(f" -> Latency (WITH Cache):    {latency_cached:.2f} seconds")
        print(f" -> Usage Metadata:          {response_cached.usage_metadata}")
        print(f" -> Response Snippet:\n{response_cached.text[:180]}...\n")

        # ---------------------------------------------------------------------
        # Step 4: Comparison Summary
        # ---------------------------------------------------------------------
        print("=" * 70)
        print("LATENCY & PERFORMANCE COMPARISON SUMMARY")
        print("=" * 70)
        print(f" -> Uncached Call Latency: {latency_uncached:.2f} s")
        print(f" -> Cached Call Latency:   {latency_cached:.2f} s")
        if latency_uncached > 0 and latency_cached > 0:
            saved = latency_uncached - latency_cached
            speedup = latency_uncached / latency_cached
            print(f" -> Time Saved:            {saved:.2f} s ({speedup:.2f}x speedup)")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # Step 5: Delete Cached Content Resource after session completion
        # ---------------------------------------------------------------------
        print("\n5. Deleting Context Cache Resource from Google Cloud...")
        client.caches.delete(name=cache_resource.name)
        print(" -> Cache resource deleted successfully.")

    except Exception as e:
        print(f"\n[SDK Execution Note]: Live call skipped or failed ({e}).")
        print("Ensure valid GOOGLE_API_KEY and minimum token requirements are satisfied.")


if __name__ == "__main__":
    gemini_context_caching_demo()
