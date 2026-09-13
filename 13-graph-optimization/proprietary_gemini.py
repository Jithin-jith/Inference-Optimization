"""
================================================================================
MODULE 13: GEMINI XLA COMPILER GRAPH OPTIMIZATION
================================================================================

CONCEPT OVERVIEW:
-----------------
Google Gemini models run on TPU architectures compiled via **XLA (Accelerated Linear Algebra)** inside JAX.

HOW XLA GRAPH COMPILATION WORKS:
--------------------------------
1. Graph Inspection: XLA analyzes the entire high-level JAX computation graph for a Gemini layer.
2. Kernel Fusion: Merges element-wise math operations, attention matrices, and activations into single
   unified TPU HBM/SRAM memory access loops.
3. Memory Pre-allocation: Pre-allocates fixed memory buffers, eliminating runtime host-device overhead.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Using `google-genai` SDK to query Gemini Flash regarding XLA graph compiler optimizations.
================================================================================
"""

import os
import time
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


def gemini_xla_graph_demo():
    """
    Queries Google Gemini API regarding XLA graph compilation architecture.
    """
    print("=" * 70)
    print("Google Gemini Infrastructure: XLA Graph Compilation Architecture")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # API Key & Client Setup
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] GEMINI_API_KEY environment variable is not set.")
        print("To run live, set export GEMINI_API_KEY='your_api_key'.\n")

    client = genai.Client()
    model_name = "gemini-2.5-flash"

    prompt = "Explain how Google XLA (Accelerated Linear Algebra) compiles computational graphs for TPU execution."

    try:
        print("Querying Gemini Cloud API...")
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=200
            )
        )
        t1 = time.perf_counter()
        print(f"Execution Latency: {t1 - t0:.2f} s")
        print(f"Response Snippet:\n{response.text[:250]}...\n")

    except Exception as e:
        print(f"\n[SDK Execution Note]: API call skipped or failed ({e}).")


if __name__ == "__main__":
    gemini_xla_graph_demo()
