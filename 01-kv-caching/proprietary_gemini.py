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
3. Execute prompts referencing the cached KV state via `types.GenerateContentConfig(cached_content=...)`.
4. Clean up cache resources via `client.caches.delete()`.
================================================================================
"""

import os
import time
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] GEMINI_API_KEY environment variable is not set.")
        print("To execute live API calls against Google Cloud:")
        print(" -> PowerShell: $env:GEMINI_API_KEY='your_api_key'")
        print(" -> Bash/Linux: export GEMINI_API_KEY='your_api_key'\n")

    # Initialize the modern official Google GenAI SDK Client
    client = genai.Client()
    model_name = "gemini-2.5-flash"

    # Construct a large system instruction context (simulated here)
    # Production context caching requires minimum token lengths (e.g. 32,768 tokens)
    large_system_instruction = """
    You are an expert Enterprise Solutions Architect AI. You provide technical blueprints
    for LLM inference optimization, cloud GPU cluster design, and high-throughput microservices.
    """ + ("Reference Blueprint Line: Always output structured latency metrics. " * 600)

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
        # Step 2: Query Model passing the Cached Content reference
        # ---------------------------------------------------------------------
        print("\n2. Querying Gemini Model referencing Cached KV State...")
        user_query = "What are the primary tradeoffs between Tensor Parallelism and Pipeline Parallelism?"
        
        t0 = time.perf_counter()
        response_cached = client.models.generate_content(
            model=model_name,
            contents=user_query,
            config=types.GenerateContentConfig(
                cached_content=cache_resource.name,  # Attaches pre-computed KV cache!
                temperature=0.2
            )
        )
        t1 = time.perf_counter()
        
        print(f" -> Latency (With Cached KV Context): {t1 - t0:.2f} seconds")
        print(f" -> Usage Metadata: {response_cached.usage_metadata}")
        print(f" -> Response Snippet:\n{response_cached.text[:180]}...\n")

        # ---------------------------------------------------------------------
        # Step 3: Delete Cached Content Resource after session completion
        # ---------------------------------------------------------------------
        print("3. Deleting Context Cache Resource from Google Cloud...")
        client.caches.delete(name=cache_resource.name)
        print(" -> Cache resource deleted successfully.")

    except Exception as e:
        print(f"\n[SDK Execution Note]: Live call skipped or failed ({e}).")
        print("Ensure valid GEMINI_API_KEY and minimum token requirements are satisfied.")


if __name__ == "__main__":
    gemini_context_caching_demo()